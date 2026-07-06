# -*- coding: utf-8 -*-
"""Build the retrieval index (dense embeddings) and the dev split.

- Corpus: COLING25-FMD train.json (1953 labeled claims = FMDLlama train+val).
- Dense: BAAI/bge-small-en-v1.5 (CPU-friendly); BM25 is rebuilt at runtime.
- Dev split: stratified 200-claim sample of the corpus (excluded from
  retrieval at query time by ID, so no self-leak during tuning).
"""
import json
import random

import numpy as np

from fmd_common import COLING, DATA, DEV_GOLD, load_jsonl, save_jsonl

EMB_MODEL = "BAAI/bge-small-en-v1.5"
DEV_SIZE = 200
SEED = 42


def doc_text(e: dict) -> str:
    digest = e.get("sci_digest")
    if isinstance(digest, list):
        digest = " ".join(str(x) for x in digest)
    return f"{e.get('claim', '')} {digest or ''}".strip()


def main() -> None:
    train = load_jsonl(COLING / "train.json")
    print(f"corpus: {len(train)} labeled claims")

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(EMB_MODEL, device="cpu")
    texts = [doc_text(e) for e in train]
    emb = model.encode(texts, batch_size=64, show_progress_bar=True,
                       normalize_embeddings=True)
    np.save(DATA / "train_emb.npy", emb.astype(np.float32))

    test = load_jsonl(DATA / "finfact_test_gold.jsonl")
    test_emb = model.encode([doc_text(e) for e in test], batch_size=64,
                            show_progress_bar=True, normalize_embeddings=True)
    np.save(DATA / "test_emb.npy", test_emb.astype(np.float32))

    # ---- stratified dev split from the corpus ----
    rng = random.Random(SEED)
    by_label: dict[str, list[int]] = {}
    for i, e in enumerate(train):
        by_label.setdefault(str(e["label"]), []).append(i)
    dev_idx: list[int] = []
    for lab, idxs in sorted(by_label.items()):
        k = round(DEV_SIZE * len(idxs) / len(train))
        dev_idx += rng.sample(idxs, k)
    dev_idx = sorted(dev_idx)
    dev = []
    for i in dev_idx:
        e = dict(train[i])
        e["gold_evidence"] = e.pop("evidence", "")
        dev.append(e)
    save_jsonl(dev, DEV_GOLD)
    np.save(DATA / "dev_emb.npy", emb[dev_idx].astype(np.float32))
    from collections import Counter
    print(f"dev split: {len(dev)}", Counter(str(e['label']) for e in dev))
    print("saved: train_emb.npy, test_emb.npy, dev_emb.npy,", DEV_GOLD.name)


if __name__ == "__main__":
    main()
