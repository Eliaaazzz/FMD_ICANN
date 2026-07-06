# -*- coding: utf-8 -*-
"""FinFact SOTA pipeline (budget edition): DeepSeek V4-Flash + hybrid-RAG kNN
+ adaptive self-consistency escalation + cached-prefix arbitration
+ NEI consistency gate + zero-cost extractive explanation.

  python run_pipeline.py --split dev  --limit 50 --tag dev50
  python run_pipeline.py --split test --tag test_final --budget-usd 1.1
Resume: rerun the same command; finished IDs are skipped (JSONL checkpoint).
"""
import argparse
import hashlib
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from fmd_common import (COLING, DATA, DEV_GOLD, HERE, RESOLUTION_TO_LABEL,
                        TEST_GOLD, canon_label, classification_metrics,
                        load_jsonl, nei_cue_in_tail, parse_verdict_json,
                        rating_to_label, truncate_words)

# USD per 1M tokens (DeepSeek V4 preview flat pricing).
PRICES = {
    "deepseek-v4-flash": {"miss": 0.14, "hit": 0.0028, "out": 0.28},
    "deepseek-v4-pro": {"miss": 0.435, "hit": 0.003625, "out": 0.87},
}


class BudgetExceeded(RuntimeError):
    pass


# --------------------------------------------------------------------------
# retrieval
# --------------------------------------------------------------------------

class HybridRetriever:
    """BM25 + dense (precomputed bge embeddings) fused with RRF."""

    def __init__(self) -> None:
        from rank_bm25 import BM25Okapi
        self.corpus = load_jsonl(COLING / "train.json")
        self.emb = np.load(DATA / "train_emb.npy")
        toks = [self._tok(self._doc_text(e)) for e in self.corpus]
        self.bm25 = BM25Okapi(toks)

    @staticmethod
    def _doc_text(e: dict) -> str:
        digest = e.get("sci_digest")
        if isinstance(digest, list):
            digest = " ".join(str(x) for x in digest)
        return f"{e.get('claim', '')} {digest or ''}".strip()

    @staticmethod
    def _tok(s: str) -> list[str]:
        return re.findall(r"[a-z0-9]+", s.lower())

    def topk(self, query_text: str, query_emb: np.ndarray, k: int,
             exclude_id: str | None = None,
             cover_labels: bool = True) -> list[dict]:
        bm = self.bm25.get_scores(self._tok(query_text))
        dn = self.emb @ query_emb
        rrf = np.zeros(len(self.corpus))
        for rank_arr in (np.argsort(-bm), np.argsort(-dn)):
            for r, i in enumerate(rank_arr[:60]):
                rrf[i] += 1.0 / (60 + r + 1)
        ranked = []
        for i in np.argsort(-rrf):
            e = self.corpus[int(i)]
            if exclude_id is not None and e.get("ID") == exclude_id:
                continue
            ranked.append(e)
            if len(ranked) >= 200:
                break
        out = ranked[:k]
        if cover_labels and len(out) >= 3:
            # swap tail slots so every label has >=1 exemplar (slot 0 kept)
            slot = len(out) - 1
            for lab in ("NEI", "True", "False"):
                present = {canon_label(e.get("label")) for e in out}
                if lab in present or slot < 1:
                    continue
                cand = next((e for e in ranked
                             if canon_label(e.get("label")) == lab
                             and e not in out), None)
                if cand:
                    out[slot] = cand
                    slot -= 1
        return out


# --------------------------------------------------------------------------
# LLM client with live cost tracking + budget guard
# --------------------------------------------------------------------------

class LLM:
    def __init__(self, model: str, mock: bool = False,
                 budget_usd: float = 0.0) -> None:
        self.model = model
        self.mock = mock
        self.budget_usd = budget_usd
        self.lock = threading.Lock()
        self.usage = {"prompt": 0, "completion": 0, "cache_hit": 0, "calls": 0}
        self._thinking_param_ok = True
        if not mock:
            from openai import OpenAI
            key = os.environ.get("DEEPSEEK_API_KEY", "")
            if not key:
                raise SystemExit("DEEPSEEK_API_KEY not set")
            self.client = OpenAI(api_key=key,
                                 base_url="https://api.deepseek.com",
                                 timeout=600, max_retries=0)

    def spent_usd(self) -> float:
        p = PRICES.get(self.model, PRICES["deepseek-v4-flash"])
        with self.lock:
            miss = self.usage["prompt"] - self.usage["cache_hit"]
            return (miss * p["miss"] + self.usage["cache_hit"] * p["hit"]
                    + self.usage["completion"] * p["out"]) / 1e6

    def chat(self, messages: list[dict], temperature: float = 0.0,
             max_tokens: int = 500, json_mode: bool = True,
             thinking: bool = False) -> str:
        if self.budget_usd and self.spent_usd() >= self.budget_usd:
            raise BudgetExceeded(
                f"budget {self.budget_usd} USD reached "
                f"(spent ~{self.spent_usd():.3f})")
        if self.mock:
            return self._mock_reply(messages, json_mode)
        kwargs: dict = dict(model=self.model, messages=messages,
                            temperature=temperature, max_tokens=max_tokens)
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        if self._thinking_param_ok:
            kwargs["extra_body"] = {
                "thinking": {"type": "enabled" if thinking else "disabled"}}
        last_err: Exception | None = None
        for attempt in range(6):
            try:
                resp = self.client.chat.completions.create(**kwargs)
                u = resp.usage
                with self.lock:
                    self.usage["calls"] += 1
                    self.usage["prompt"] += getattr(u, "prompt_tokens", 0) or 0
                    self.usage["completion"] += (
                        getattr(u, "completion_tokens", 0) or 0)
                    self.usage["cache_hit"] += (
                        getattr(u, "prompt_cache_hit_tokens", 0) or 0)
                return resp.choices[0].message.content or ""
            except Exception as e:  # noqa: BLE001 - API surface is broad
                last_err = e
                msg = str(e)
                if "thinking" in msg.lower() and self._thinking_param_ok:
                    self._thinking_param_ok = False
                    kwargs.pop("extra_body", None)
                    continue
                if ("response_format" in msg or "json" in msg.lower()) \
                        and kwargs.pop("response_format", None) is not None:
                    continue
                time.sleep(min(2 ** attempt * 2, 45))
        raise RuntimeError(f"LLM call failed after retries: {last_err}")

    def _mock_reply(self, messages: list[dict], json_mode: bool) -> str:
        h = int(hashlib.md5(
            messages[-1]["content"].encode()).hexdigest(), 16)
        lab = ["False", "True", "NEI"][h % 3]
        with self.lock:
            self.usage["calls"] += 1
        if json_mode:
            return json.dumps({
                "key_assertion": "mock", "evidence_for": "mock",
                "evidence_against": "mock",
                "sufficiency": "sufficient" if h % 4 else "insufficient",
                "prediction": lab, "confidence": 0.5 + (h % 50) / 100})
        return ("Mock explanation. " * 40).strip()


# --------------------------------------------------------------------------
# prompts
# --------------------------------------------------------------------------

VERDICT_SYSTEM = (
    "You are a meticulous financial fact-checking expert. You verify claims "
    "strictly against the provided contextual information and follow the "
    "labeling conventions of the reference corpus. Always answer with a "
    "single valid JSON object."
)

LABEL_GUIDE = """# Label definitions (this corpus derives labels from professional fact-check rulings)
- "True": the fact-check confirms the core assertion. This INCLUDES rulings like "True" and "Mostly True" (minor imprecision is fine if the core assertion holds).
- "False": the fact-check refutes the core assertion: fabricated, debunked, miscaptioned, scam, or shown baseless.
- "NEI": the fact-check does NOT cleanly confirm or refute the core assertion. This INCLUDES rulings like "Mixture", "Half True", "Unproven", "Unverified", "Research in progress", claims partly true and partly false, and claims the checker could not resolve either way.

# Decision procedure
1. If the context contains the fact-checker's own rating language (e.g. "we rate this claim Mixture", "Mostly True", "Unproven", "this is false"), quote it in editor_rating_verbatim - it is authoritative and your prediction MUST follow its mapping ("Mostly True" -> True, NOT NEI; "Mixture"/"Half True"/"Unproven" -> NEI).
2. Otherwise judge from the reported evidence: confirmed -> True; refuted/baseless -> False; mixed or unresolved -> NEI.
3. Distinguish carefully: "no evidence for the claim AND actively debunked" -> False, but "could not be verified either way" -> NEI.
4. Nuance, caveats, or minor imprecision alone do NOT make a claim mixed: use mixed/unresolved only when the checker's bottom line is genuinely mixed or they could not verify the core assertion. If the checker's overall conclusion validates the core of the claim -> True.
The reference examples below come from the same corpus and demonstrate its labeling conventions."""

VERDICT_SCHEMA = """# Output format (strict JSON, no other text)
{"key_assertion": "<the claim's single central factual assertion, one sentence>",
 "editor_rating_verbatim": "<exact rating phrase quoted from the context (e.g. 'we rate this claim Mixture.', 'Mostly True'), or 'none'>",
 "evidence_for": "<strongest context evidence supporting the assertion, or 'none'>",
 "evidence_against": "<strongest context evidence contradicting/undermining it, or 'none'>",
 "resolution": "<confirmed | refuted | mixed | unresolved>",
 "prediction": "<True | False | NEI>  (confirmed->True, refuted->False, mixed/unresolved->NEI)",
 "confidence": <0.0-1.0>}"""


def exemplar_block(exemplars: list[dict]) -> str:
    parts = []
    for i, e in enumerate(exemplars, 1):
        parts.append(
            f"Example {i}:\n"
            f"Claim: {e.get('claim', '')}\n"
            f"Context (excerpt): {truncate_words(e.get('justification', ''), 50)}\n"
            f"Label: {canon_label(e.get('label'))}\n"
            f"Editor's rationale (excerpt): "
            f"{truncate_words(e.get('evidence', ''), 40)}")
    return "\n\n".join(parts)


def item_block(item: dict, max_ctx_words: int) -> str:
    digest = item.get("sci_digest")
    if isinstance(digest, list):
        digest = " ".join(str(x) for x in digest)
    return (f"Claim: {item.get('claim', '')}\n"
            f"Posted: {item.get('posted', '')}\n"
            f"Claim summaries: {digest or '(none)'}\n"
            f"Contextual information: "
            f"{truncate_words(item.get('justification', ''), max_ctx_words)}")


def verdict_messages(item: dict, exemplars: list[dict],
                     max_ctx_words: int) -> list[dict]:
    user = (
        "# Task\nDetermine whether the claim below is True, False, or NEI "
        "(Not Enough Information) based ONLY on the provided contextual "
        "information.\n\n"
        f"{LABEL_GUIDE}\n\n"
        f"# Reference examples (similar verified claims)\n"
        f"{exemplar_block(exemplars)}\n\n"
        f"# Claim to verify\n{item_block(item, max_ctx_words)}\n\n"
        f"{VERDICT_SCHEMA}")
    return [{"role": "system", "content": VERDICT_SYSTEM},
            {"role": "user", "content": user}]


ADJUDICATE_TURN = """The analyses above may be wrong or in conflict. You are now the senior adjudicator.

# Additional analyst opinions
{opinions}

# Adjudication procedure
1. Re-scan the context for the fact-checker's own rating language, especially near the end (e.g. "we rate this claim ...", "Mixture", "Half True", "Unproven", "Mostly True"). If present it is authoritative: Mixture/Half True/Unproven/unresolved -> NEI; True/Mostly True -> True; False/debunked -> False.
2. Otherwise weigh each analyst's cited evidence against the context you already read.
3. Counterfactual test: if the claim were False, what would the context look like? If it were True? Which world matches?
4. "No evidence + actively debunked" -> False; "could not be verified either way" or partly true/partly false -> NEI.

Output the final verdict as the same strict JSON schema as before."""


def arbitration_messages(base_msgs: list[dict], samples: list[dict]) -> list[dict]:
    """Multi-turn continuation of the verdict prompt → prefix cache hit."""
    opinions = json.dumps(
        [{k: s.get(k) for k in ("prediction", "confidence", "sufficiency",
                                "evidence_for", "evidence_against")}
         for s in samples[1:]], ensure_ascii=False, indent=1)
    return base_msgs + [
        {"role": "assistant", "content": samples[0].get("raw", "")},
        {"role": "user", "content": ADJUDICATE_TURN.format(opinions=opinions)},
    ]


# --------------------------------------------------------------------------
# per-item pipeline
# --------------------------------------------------------------------------

def majority(labels: list[str]) -> tuple[str | None, int]:
    from collections import Counter
    c = Counter(l for l in labels if l)
    if not c:
        return None, 0
    lab, n = c.most_common(1)[0]
    return lab, n


def needs_escalation(s: dict, item: dict, conf_thresh: float) -> bool:
    pred, conf = s.get("prediction"), s.get("confidence")
    if pred is None:
        return True
    if conf is None or conf < conf_thresh:
        return True
    res = str(s.get("resolution") or "").lower()
    if res in RESOLUTION_TO_LABEL and RESOLUTION_TO_LABEL[res] != pred:
        return True  # resolution/prediction inconsistency
    if pred != "NEI" and nei_cue_in_tail(item.get("justification", "")):
        return True  # corpus NEI-cue present but not predicted NEI
    return False


def process_item(item: dict, query_emb: np.ndarray, retriever: HybridRetriever,
                 llm: LLM, cfg: argparse.Namespace) -> dict:
    exclude = item.get("ID") if cfg.split == "dev" else None
    exemplars = retriever.topk(
        f"{item.get('claim', '')}", query_emb, cfg.k, exclude_id=exclude)
    msgs = verdict_messages(item, exemplars, cfg.max_ctx_words)

    samples: list[dict] = []
    raw = llm.chat(msgs, temperature=0.0, max_tokens=500, json_mode=True)
    s1 = parse_verdict_json(raw)
    s1["raw"] = raw
    samples.append(s1)

    path = "single"
    final = dict(s1)
    if needs_escalation(s1, item, cfg.conf_thresh):
        path = "escalated"
        for t in (0.8, 0.8):
            raw = llm.chat(msgs, temperature=t, max_tokens=500, json_mode=True)
            s = parse_verdict_json(raw)
            s["raw"] = raw
            samples.append(s)
        labels = [s["prediction"] for s in samples]
        maj, votes = majority(labels)
        if maj is not None and votes >= 2:
            final = next(dict(s) for s in samples if s["prediction"] == maj)
        if maj is None or votes < cfg.n_agree or cfg.always_arbitrate:
            araw = llm.chat(arbitration_messages(msgs, samples),
                            temperature=0.0, max_tokens=cfg.arb_max_tokens,
                            json_mode=True, thinking=True)
            arb = parse_verdict_json(araw)
            arb["raw"] = araw
            if arb["prediction"]:
                final = arb
                path = "arbitrated"
            samples.append({**arb, "role": "arbiter"})

    # authoritative editor-rating mapping (verified quote only)
    mapped = rating_to_label(final.get("editor_rating_verbatim"),
                             item.get("justification", ""))
    if mapped is None and samples:
        mapped = rating_to_label(samples[0].get("editor_rating_verbatim"),
                                 item.get("justification", ""))
    if mapped and mapped != final.get("prediction"):
        final = dict(final)
        final["prediction"] = mapped
        path += "+rating_map"
    elif cfg.nei_gate and final.get("prediction") in ("True", "False"):
        nei_res = sum(1 for s in samples[:3]
                      if str(s.get("resolution") or "").lower()
                      in ("mixed", "unresolved"))
        if nei_res >= cfg.nei_gate_min:
            final = dict(final)
            final["prediction"] = "NEI"
            path += "+nei_gate"

    if cfg.explanation_mode == "extractive":
        explanation = truncate_words(item.get("justification", ""),
                                     cfg.extractive_words)
    else:
        explanation = ""

    return {
        "ID": item["ID"],
        "claim": item.get("claim", ""),
        "gold": item.get("label"),
        "prediction": final.get("prediction") or "NEI",
        "confidence": final.get("confidence"),
        "resolution": final.get("resolution"),
        "editor_rating": final.get("editor_rating_verbatim"),
        "path": path,
        "votes": [s.get("prediction") for s in samples],
        "confs": [s.get("confidence") for s in samples],
        "resolution_votes": [s.get("resolution") for s in samples],
        "retrieved_ids": [e.get("ID") for e in exemplars],
        "retrieved_labels": [canon_label(e.get("label")) for e in exemplars],
        "explanation": explanation.strip(),
    }


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["dev", "test"], default="dev")
    ap.add_argument("--model", default="deepseek-v4-flash")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--conf-thresh", type=float, default=0.7)
    ap.add_argument("--n-agree", type=int, default=2,
                    help="votes needed to accept majority without arbitration")
    ap.add_argument("--always-arbitrate", action="store_true")
    ap.add_argument("--arb-max-tokens", type=int, default=2500)
    ap.add_argument("--nei-gate", action="store_true", default=True)
    ap.add_argument("--no-nei-gate", dest="nei_gate", action="store_false")
    ap.add_argument("--nei-gate-min", type=int, default=2)
    ap.add_argument("--explanation-mode", choices=["extractive", "none"],
                    default="extractive")
    ap.add_argument("--extractive-words", type=int, default=400)
    ap.add_argument("--max-ctx-words", type=int, default=1000)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--budget-usd", type=float, default=1.15)
    ap.add_argument("--mock", action="store_true")
    cfg = ap.parse_args()

    gold_path = DEV_GOLD if cfg.split == "dev" else TEST_GOLD
    items = load_jsonl(gold_path)
    q_emb = np.load(DATA / f"{cfg.split}_emb.npy")
    assert len(items) == len(q_emb), "embeddings out of sync with gold file"
    if cfg.offset:
        items, q_emb = items[cfg.offset:], q_emb[cfg.offset:]
    if cfg.limit:
        items, q_emb = items[:cfg.limit], q_emb[:cfg.limit]

    outdir = HERE / "results" / cfg.tag
    outdir.mkdir(parents=True, exist_ok=True)
    pred_path = outdir / "predictions.jsonl"
    done = {r["ID"] for r in load_jsonl(pred_path)} if pred_path.exists() else set()
    todo = [(it, q_emb[i]) for i, it in enumerate(items)
            if it["ID"] not in done]
    print(f"[{cfg.tag}] split={cfg.split} model={cfg.model} total={len(items)} "
          f"done={len(done)} todo={len(todo)} budget=${cfg.budget_usd}")
    (outdir / "config.json").write_text(
        json.dumps(vars(cfg), indent=2, default=str), encoding="utf-8")

    retriever = HybridRetriever()
    llm = LLM(cfg.model, mock=cfg.mock, budget_usd=cfg.budget_usd)
    wlock = threading.Lock()
    t0 = time.time()
    from tqdm import tqdm

    def work(pair):
        item, qe = pair
        return process_item(item, qe, retriever, llm, cfg)

    errors, budget_stop = 0, False
    with ThreadPoolExecutor(max_workers=cfg.workers) as ex, \
            open(pred_path, "a", encoding="utf-8") as f:
        futures = [ex.submit(work, p) for p in todo]
        for fut in tqdm(as_completed(futures), total=len(futures)):
            try:
                rec = fut.result()
            except BudgetExceeded:
                budget_stop = True
                continue
            except Exception as e:  # noqa: BLE001
                errors += 1
                print("ITEM ERROR:", repr(e)[:300])
                continue
            with wlock:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()

    rows = load_jsonl(pred_path)
    metrics = classification_metrics([r["gold"] for r in rows],
                                     [r["prediction"] for r in rows])
    metrics["usage"] = llm.usage
    metrics["est_cost_usd"] = round(llm.spent_usd(), 4)
    metrics["errors"] = errors
    metrics["budget_stop"] = budget_stop
    metrics["minutes"] = round((time.time() - t0) / 60, 1)
    from collections import Counter
    metrics["paths"] = dict(Counter(r.get("path", "?") for r in rows))
    (outdir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps({k: metrics[k] for k in
                      ("n", "accuracy", "macro_f1", "micro_f1",
                       "unparseable_pred", "est_cost_usd", "paths")},
                     indent=2))
    print("per-class:", json.dumps(metrics["per_class"], indent=2))
    if budget_stop:
        print("!! BUDGET STOP: rerun same command later to resume remaining items")


if __name__ == "__main__":
    main()
