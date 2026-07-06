# -*- coding: utf-8 -*-
"""Full evaluation of a run: classification + explanation quality,
side-by-side with the FMDLlama paper's Table 3 numbers.

  python evaluate.py --tag test_final [--bertscore]
"""
import argparse
import json

from fmd_common import (DEV_GOLD, HERE, TEST_GOLD, classification_metrics,
                        load_jsonl)

FMDLLAMA3 = {"accuracy": 0.7362, "macro_precision": 0.6733,
             "macro_recall": 0.6700, "macro_f1": 0.6667,
             "rouge1": 0.4524, "rouge2": 0.3498, "rougeL": 0.3773,
             "bertscore": 0.6756}
GPT4O = {"accuracy": 0.6702, "macro_f1": 0.6283, "rouge1": 0.2855}
GPT35 = {"accuracy": 0.7270, "macro_f1": 0.6380, "rouge1": 0.2682}


def rouge_scores(rows: list[dict], gold_by_id: dict) -> dict:
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"],
                                      use_stemmer=True)
    agg = {"rouge1": [], "rouge2": [], "rougeL": []}
    n_empty = 0
    for r in rows:
        ref = (gold_by_id.get(r["ID"]) or {}).get("gold_evidence", "")
        hyp = r.get("explanation", "")
        if not ref.strip() or not hyp.strip():
            n_empty += 1
            continue
        s = scorer.score(ref, hyp)
        for k in agg:
            agg[k].append(s[k].fmeasure)
    out = {k: round(sum(v) / max(len(v), 1), 4) for k, v in agg.items()}
    out["skipped_empty"] = n_empty
    return out


def bert_scores(rows: list[dict], gold_by_id: dict) -> dict:
    from bert_score import score as bscore
    refs, hyps = [], []
    for r in rows:
        ref = (gold_by_id.get(r["ID"]) or {}).get("gold_evidence", "")
        hyp = r.get("explanation", "")
        if ref.strip() and hyp.strip():
            refs.append(ref)
            hyps.append(hyp)
    p, r_, f = bscore(hyps, refs, lang="en", batch_size=16, verbose=True)
    return {"bertscore_p": round(p.mean().item(), 4),
            "bertscore_r": round(r_.mean().item(), 4),
            "bertscore_f1": round(f.mean().item(), 4)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--split", choices=["dev", "test"], default=None)
    ap.add_argument("--bertscore", action="store_true")
    args = ap.parse_args()

    outdir = HERE / "results" / args.tag
    rows = load_jsonl(outdir / "predictions.jsonl")
    cfg = json.loads((outdir / "config.json").read_text(encoding="utf-8"))
    split = args.split or cfg.get("split", "test")
    gold = load_jsonl(TEST_GOLD if split == "test" else DEV_GOLD)
    gold_by_id = {g["ID"]: g for g in gold}

    metrics = classification_metrics([r["gold"] for r in rows],
                                     [r["prediction"] for r in rows])
    have_expl = any(r.get("explanation") for r in rows)
    if have_expl:
        metrics["rouge"] = rouge_scores(rows, gold_by_id)
        if args.bertscore:
            metrics["bertscore"] = bert_scores(rows, gold_by_id)

    from collections import Counter
    metrics["paths"] = dict(Counter(r.get("path", "?") for r in rows))
    (outdir / "eval_full.json").write_text(json.dumps(metrics, indent=2),
                                           encoding="utf-8")

    print(f"== {args.tag} (split={split}, n={metrics['n']}) ==")
    hdr = f"{'metric':18s} {'ours':>8s} {'FMDLlama3':>10s} {'GPT-4o':>8s} {'GPT-3.5':>8s}"
    print(hdr)
    for key in ("accuracy", "macro_precision", "macro_recall", "macro_f1"):
        print(f"{key:18s} {metrics[key]:8.4f} {FMDLLAMA3.get(key, float('nan')):10.4f} "
              f"{GPT4O.get(key, float('nan')):8.4f} {GPT35.get(key, float('nan')):8.4f}")
    print("micro_f1          ", f"{metrics['micro_f1']:.4f}",
          " (challenge winner Dunamu: 0.8467 micro-F1 on private split)")
    if have_expl and "rouge" in metrics:
        for key in ("rouge1", "rouge2", "rougeL"):
            print(f"{key:18s} {metrics['rouge'][key]:8.4f} "
                  f"{FMDLLAMA3[key]:10.4f}")
        if "bertscore" in metrics:
            print(f"{'bertscore_f1':18s} {metrics['bertscore']['bertscore_f1']:8.4f} "
                  f"{FMDLLAMA3['bertscore']:10.4f}")
    print("per-class:", json.dumps(metrics["per_class"]))
    print("paths:", metrics["paths"])
    print("confusion (rows=gold F/T/NEI):",
          metrics["confusion_matrix_rows_gold_cols_pred"]["matrix"])


if __name__ == "__main__":
    main()
