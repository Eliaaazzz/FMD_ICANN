# -*- coding: utf-8 -*-
"""Shared utilities: data IO, label handling, prediction parsing, metrics."""
import json
import re
from pathlib import Path

HERE = Path(__file__).parent
DATA = HERE / "data"
COLING = DATA / "coling25_fmd"
TEST_GOLD = DATA / "finfact_test_gold.jsonl"
DEV_GOLD = DATA / "finfact_dev_gold.jsonl"
LABELS = ["False", "True", "NEI"]


def load_jsonl(path) -> list[dict]:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def save_jsonl(rows: list[dict], path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def canon_label(s) -> str | None:
    """Map free-form label text to True/False/NEI (None if unparseable)."""
    if s is None:
        return None
    t = str(s).strip().lower().rstrip(".")
    if t in {"true", "1. true", "1", "real"}:
        return "True"
    if t in {"false", "0. false", "0", "fake"}:
        return "False"
    if "nei" in t or "not enough" in t or t == "2":
        return "NEI"
    if t.startswith("true"):
        return "True"
    if t.startswith("false"):
        return "False"
    return None


def parse_verdict_json(text: str) -> dict:
    """Parse a model response expected to be JSON with a `prediction` field.

    Falls back to regex extraction. Always returns a dict with at least
    {prediction, confidence, sufficiency} (prediction may be None).
    """
    out = {"prediction": None, "confidence": None, "sufficiency": None,
           "resolution": None, "editor_rating_verbatim": None}
    if not text:
        return out
    candidate = text.strip()
    m = re.search(r"\{.*\}", candidate, flags=re.S)
    if m:
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict):
                out.update({k: obj.get(k, out.get(k)) for k in
                            ("prediction", "confidence", "sufficiency",
                             "resolution", "editor_rating_verbatim",
                             "key_assertion", "evidence_for",
                             "evidence_against")})
        except json.JSONDecodeError:
            pass
    out["prediction"] = canon_label(out.get("prediction"))
    if out["prediction"] is None:
        m = re.search(r"prediction\"?\s*[:=]?\s*\"?([A-Za-z. ]{3,30})",
                      candidate, flags=re.I)
        if m:
            out["prediction"] = canon_label(m.group(1))
    if out["prediction"] is None:
        m = re.search(r"\b(true|false|nei|not enough information)\b",
                      candidate, flags=re.I)
        if m:
            out["prediction"] = canon_label(m.group(1))
    try:
        out["confidence"] = float(out["confidence"])
    except (TypeError, ValueError):
        out["confidence"] = None
    return out


RESOLUTION_TO_LABEL = {"confirmed": "True", "refuted": "False",
                       "mixed": "NEI", "unresolved": "NEI"}

NEI_CUE_RE = re.compile(
    r"\b(mixture|half[- ]?true|unproven|unverified|research in progress)\b",
    re.I)


def rating_to_label(quoted, justification: str) -> str | None:
    """Map the model-quoted editor rating to a label, only if the quote
    verifiably appears in the source context (anti-hallucination check)."""
    if not quoted or str(quoted).strip().lower() in ("none", "null", ""):
        return None
    qs = str(quoted).strip().strip("\"'").rstrip(".")

    def _norm(s: str) -> str:
        return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()

    if _norm(qs) not in _norm(justification or ""):
        return None
    t = qs.lower()
    if re.search(r"\b(mixture|half[- ]?true|unproven|unverified"
                 r"|research in progress)\b", t):
        return "NEI"
    if re.search(r"\b(mostly|largely)?\s*true\b", t):
        return "True"
    if re.search(r"\b(mostly|largely)?\s*false\b", t) or "pants on fire" in t:
        return "False"
    return None


def nei_cue_in_tail(justification: str, tail_chars: int = 800) -> bool:
    """Corpus-convention cue: rating language that maps to NEI in Fin-Fact."""
    tail = (justification or "")[-tail_chars:]
    if re.search(r"\bmostly[- ]?true\b", tail, re.I):
        return False
    return bool(NEI_CUE_RE.search(tail))


def classification_metrics(golds: list[str], preds: list[str]) -> dict:
    """ACC + macro/micro P/R/F1 + per-class breakdown (3-way FinFact)."""
    from sklearn.metrics import (accuracy_score, confusion_matrix,
                                 precision_recall_fscore_support)
    clean_preds = [p if p in LABELS else "NEI" for p in preds]
    fallback_used = sum(1 for p in preds if p not in LABELS)
    acc = accuracy_score(golds, clean_preds)
    p_ma, r_ma, f_ma, _ = precision_recall_fscore_support(
        golds, clean_preds, labels=LABELS, average="macro", zero_division=0)
    p_mi, r_mi, f_mi, _ = precision_recall_fscore_support(
        golds, clean_preds, labels=LABELS, average="micro", zero_division=0)
    p_c, r_c, f_c, sup = precision_recall_fscore_support(
        golds, clean_preds, labels=LABELS, average=None, zero_division=0)
    cm = confusion_matrix(golds, clean_preds, labels=LABELS)
    return {
        "n": len(golds),
        "unparseable_pred": fallback_used,
        "accuracy": round(acc, 4),
        "macro_precision": round(p_ma, 4),
        "macro_recall": round(r_ma, 4),
        "macro_f1": round(f_ma, 4),
        "micro_f1": round(f_mi, 4),
        "per_class": {
            lab: {"precision": round(p_c[i], 4), "recall": round(r_c[i], 4),
                  "f1": round(f_c[i], 4), "support": int(sup[i])}
            for i, lab in enumerate(LABELS)},
        "confusion_matrix_rows_gold_cols_pred": {
            "labels": LABELS, "matrix": cm.tolist()},
    }


def truncate_words(text: str, n: int) -> str:
    words = re.findall(r"\S+", text or "")
    return " ".join(words[:n])
