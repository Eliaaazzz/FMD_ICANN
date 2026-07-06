# -*- coding: utf-8 -*-
"""Chunked, resumable BERTScore for the test-run explanations.

Reports both raw and baseline-rescaled F1 (FMDLlama's setting is
unspecified, so we report both). Limits torch threads to stay polite.
"""
import json
from pathlib import Path

import torch

from fmd_common import HERE, TEST_GOLD, load_jsonl

TAG = "test_v3_flash"
CHUNK = 64
OUT = HERE / "results" / TAG / "bertscore_partial.jsonl"
FINAL = HERE / "results" / TAG / "bertscore.json"


def main() -> None:
    torch.set_num_threads(6)
    rows = load_jsonl(HERE / "results" / TAG / "predictions.jsonl")
    gold = {g["ID"]: g for g in load_jsonl(TEST_GOLD)}
    pairs = [(r["ID"], r.get("explanation", ""),
              gold[r["ID"]].get("gold_evidence", ""))
             for r in rows
             if r.get("explanation", "").strip()
             and gold[r["ID"]].get("gold_evidence", "").strip()]

    done = {}
    if OUT.exists():
        for r in load_jsonl(OUT):
            done[r["ID"]] = r
    todo = [p for p in pairs if p[0] not in done]
    print(f"pairs={len(pairs)} done={len(done)} todo={len(todo)}")

    if todo:
        from bert_score import BERTScorer
        scorer = BERTScorer(lang="en", rescale_with_baseline=True,
                            batch_size=4)
        b = scorer.baseline_vals  # (3,) tensor: P,R,F baselines
        bf = float(b[2])
        with open(OUT, "a", encoding="utf-8") as f:
            for i in range(0, len(todo), CHUNK):
                chunk = todo[i:i + CHUNK]
                cands = [c for _, c, _ in chunk]
                refs = [r for _, _, r in chunk]
                _, _, F = scorer.score(cands, refs)
                for (pid, _, _), fv in zip(chunk, F.tolist()):
                    rec = {"ID": pid, "f_rescaled": round(fv, 5),
                           "f_raw": round(fv * (1 - bf) + bf, 5)}
                    f.write(json.dumps(rec) + "\n")
                f.flush()
                print(f"chunk {i // CHUNK + 1}/"
                      f"{(len(todo) + CHUNK - 1) // CHUNK} done")

    allr = load_jsonl(OUT)
    n = len(allr)
    res = {
        "n": n,
        "bertscore_f1_raw": round(sum(r["f_raw"] for r in allr) / n, 4),
        "bertscore_f1_rescaled": round(
            sum(r["f_rescaled"] for r in allr) / n, 4),
        "fmdllama3_reported": 0.6756,
        "note": "FMDLlama BERTScore setting unspecified; both variants given",
    }
    FINAL.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
