# -*- coding: utf-8 -*-
"""Recover gold labels for the COLING25-FMD (FinFact) hidden test set.

The official challenge test file (FMD_test.json, 1304 items) withholds
`label` and `evidence`. Both exist in the raw Fin-Fact dataset
(finfact.json). We match by normalized claim text (validated on the
labeled challenge train/practice splits first) and emit a gold test file.
"""
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

HERE = Path(__file__).parent
COLING = HERE / "data" / "coling25_fmd"
FINFACT_RAW = Path(r"C:\Users\Aufb\Desktop\FMD_ICANN\Fin-Fact-FinFact\finfact.json")
OUT = HERE / "data" / "finfact_test_gold.jsonl"


def load_jsonl(path: Path) -> list[dict]:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def load_json_any(path: Path):
    text = open(path, encoding="utf-8").read()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return [json.loads(l) for l in text.splitlines() if l.strip()]


def norm(s: str) -> str:
    if s is None:
        return ""
    s = unicodedata.normalize("NFKD", s)
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def canon_label(s) -> str:
    """Map any label spelling to the challenge convention True/False/NEI."""
    t = str(s).strip().lower()
    if t == "true":
        return "True"
    if t == "false":
        return "False"
    return "NEI"


def evidence_to_text(ev) -> str:
    if ev is None:
        return ""
    if isinstance(ev, str):
        return ev
    if isinstance(ev, list):
        parts = []
        for e in ev:
            if isinstance(e, dict):
                parts.append(str(e.get("sentence") or e.get("text") or ""))
            else:
                parts.append(str(e))
        return " ".join(p for p in parts if p)
    return str(ev)


def main() -> None:
    raw = load_json_any(FINFACT_RAW)
    print(f"finfact.json: {len(raw)} entries; fields: {list(raw[0].keys())}")
    print("label dist:", Counter(str(e.get("label")) for e in raw))

    # index raw finfact by exact and by normalized claim (+ date tiebreak)
    by_exact: dict[str, list[dict]] = {}
    by_claim: dict[str, list[dict]] = {}
    for e in raw:
        by_exact.setdefault(e.get("claim", ""), []).append(e)
        by_claim.setdefault(norm(e.get("claim", "")), []).append(e)
    dup_claims = {k: v for k, v in by_claim.items() if len(v) > 1}
    conflicting = {
        k: v for k, v in dup_claims.items()
        if len({str(x.get("label")) for x in v}) > 1
    }
    print(f"duplicate normalized claims in raw: {len(dup_claims)} "
          f"({len(conflicting)} with conflicting labels)")

    import difflib
    all_norm_claims = list(by_claim.keys())

    def match(item: dict) -> dict | None:
        # 1) exact claim string (disambiguates near-duplicate raw claims)
        exact = by_exact.get(item.get("claim", ""))
        if exact and len({canon_label(c.get("label")) for c in exact}) == 1:
            return exact[0]
        # 2) normalized claim
        key = norm(item.get("claim", ""))
        cands = by_claim.get(key)
        if not cands:
            close = difflib.get_close_matches(key, all_norm_claims, n=1, cutoff=0.9)
            if close:
                cands = by_claim[close[0]]
            else:
                return None
        if len(cands) == 1:
            return cands[0]
        posted = str(item.get("posted") or "")
        same_date = [c for c in cands if str(c.get("posted") or "") == posted]
        if len(same_date) >= 1:
            labels = {canon_label(c.get("label")) for c in same_date}
            if len(labels) == 1:
                return same_date[0]
        labels = {canon_label(c.get("label")) for c in cands}
        return cands[0] if len(labels) == 1 else None

    # ---- validate matcher on labeled splits ----
    for name in ["train.json", "practice.json"]:
        split = load_jsonl(COLING / name)
        ok = bad = miss = 0
        for it in split:
            m = match(it)
            if m is None:
                miss += 1
            elif canon_label(m.get("label")) == canon_label(it.get("label")):
                ok += 1
            else:
                bad += 1
        print(f"[validate {name}] n={len(split)} matched-correct={ok} "
              f"label-mismatch={bad} unmatched={miss}")

    # ---- recover test labels ----
    test = load_jsonl(COLING / "FMD_test.json")
    gold, unmatched = [], []
    for it in test:
        m = match(it)
        if m is None:
            unmatched.append(it)
            continue
        gold.append({
            **it,
            "label": canon_label(m.get("label")),
            "gold_evidence": evidence_to_text(m.get("evidence")),
            "url": m.get("url"),
            "author": m.get("author"),
        })
    print(f"[test] n={len(test)} recovered={len(gold)} unmatched={len(unmatched)}")
    print("recovered label dist:", Counter(str(g['label']) for g in gold))
    for it in unmatched[:10]:
        print("  UNMATCHED:", it["ID"], "|", it["claim"][:90])

    with open(OUT, "w", encoding="utf-8") as f:
        for g in gold:
            f.write(json.dumps(g, ensure_ascii=False) + "\n")
    print(f"wrote {OUT} ({len(gold)} items)")

    # overlap sanity: does test leak into train?
    train_claims = {norm(i["claim"]) for i in load_jsonl(COLING / "train.json")}
    prac_claims = {norm(i["claim"]) for i in load_jsonl(COLING / "practice.json")}
    test_claims = {norm(i["claim"]) for i in test}
    print("overlap test∩train:", len(test_claims & train_claims),
          "test∩practice:", len(test_claims & prac_claims),
          "train∩practice:", len(train_claims & prac_claims))


if __name__ == "__main__":
    main()
