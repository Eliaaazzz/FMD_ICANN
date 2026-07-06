# -*- coding: utf-8 -*-
"""FinGuard binary task via zero-API-cost embedding kNN.

Uses the locally reproduced 2900/600/1500 split (the FMDLlama paper's exact
FinGuard split is unpublished, so results carry a 'reproduced split' caveat).
Extracts raw news text from the Llama-formatted jsonl, embeds with
bge-small-en-v1.5 (CPU), and classifies by cosine-kNN majority vote.
"""
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np

SRC = Path(r"C:\Users\Aufb\Desktop\FMD_ICANN\FMD_ICANN-paper-\processed_data\finguard")
OUT = Path(__file__).parent / "results" / "finguard_knn"


def extract_raw(formatted: str) -> str:
    m = re.search(r"Text:\s*(.*?)<\|eot_id\|>", formatted, flags=re.S)
    return (m.group(1) if m else formatted).strip()


def load_split(name: str) -> tuple[list[str], list[str]]:
    texts, labels = [], []
    with open(SRC / f"{name}.jsonl", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            texts.append(extract_raw(r["text"]))
            labels.append("Fake" if str(r["label"]).lower() in
                          ("fake", "false", "0") else "True")
    return texts, labels


def main() -> None:
    train_x, train_y = load_split("train")
    test_x, test_y = load_split("test")
    print(f"train={len(train_x)} {Counter(train_y)}  "
          f"test={len(test_x)} {Counter(test_y)}")

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("BAAI/bge-small-en-v1.5", device="cpu")
    emb_tr = model.encode(train_x, batch_size=64, show_progress_bar=True,
                          normalize_embeddings=True)
    emb_te = model.encode(test_x, batch_size=64, show_progress_bar=True,
                          normalize_embeddings=True)

    sims = emb_te @ emb_tr.T
    results = {}
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    for k in (1, 3, 5, 9, 15):
        idx = np.argsort(-sims, axis=1)[:, :k]
        preds = []
        for row_i, row in enumerate(idx):
            votes = Counter(train_y[j] for j in row)
            top = votes.most_common()
            if len(top) > 1 and top[0][1] == top[1][1]:
                preds.append(train_y[idx[row_i][0]])  # tie -> nearest
            else:
                preds.append(top[0][0])
        acc = accuracy_score(test_y, preds)
        p, r, f1, _ = precision_recall_fscore_support(
            test_y, preds, average="macro", zero_division=0)
        results[k] = {"accuracy": round(acc, 4), "macro_precision": round(p, 4),
                      "macro_recall": round(r, 4), "macro_f1": round(f1, 4)}
        print(f"k={k:2d}  acc={acc:.4f}  macroF1={f1:.4f}")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "metrics.json").write_text(json.dumps(results, indent=2),
                                      encoding="utf-8")
    print("paper reference: FMDLlama3 0.9947 / RoBERTa 0.9961 "
          "(official split unpublished; ours is a reproduced split)")


if __name__ == "__main__":
    main()
