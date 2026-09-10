#!/usr/bin/env python3
"""Reproducible v0.2 data pipeline for SHNU-LLM.

Acquires an openly-licensed corpus, cleans and deduplicates it, splits it
deterministically into train/validation with no overlap, trains (or reuses) the
from-scratch byte-level BPE tokenizer, tokenizes, computes exact token counts,
and writes the tokenized streams plus a provenance/statistics metadata file to a
durable directory (a Google Drive path on Colab).

No pretrained model weights, no proprietary datasets, no paid APIs.

Sources (all legally clear):
  wikitext103  WikiText-103-raw-v1  (CC BY-SA 3.0)  via HuggingFace `Salesforce/wikitext`, non-streaming
  wikitext2    WikiText-2 raw text  (CC BY-SA 3.0)  via a public mirror URL (small; for CI / low-compute)
  shakespeare  Tiny Shakespeare     (public domain) fallback only

Usage:
  python scripts/prepare_data.py --dataset wikitext103 --out-dir /content/drive/MyDrive/SHNU_LLM \
      --vocab-size 16000 --val-ratio 0.01
"""
from __future__ import annotations

import os
import sys
import json
import time
import hashlib
import argparse
import datetime

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import shnu_llm as S

WIKITEXT2_URLS = {
    "train": "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/train.txt",
    "valid": "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/valid.txt",
}
SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

LICENSES = {
    "wikitext103": "CC BY-SA 3.0",
    "wikitext2": "CC BY-SA 3.0",
    "shakespeare": "public domain",
}


def _http_text(url: str, timeout: int = 120) -> str:
    import urllib.request
    return urllib.request.urlopen(url, timeout=timeout).read().decode("utf-8")


def acquire(dataset: str, max_chars: int | None):
    """Return (documents, provenance_dict). Documents are raw text records."""
    if dataset == "wikitext103":
        # Current canonical id is Salesforce/wikitext; use NON-streaming to avoid the
        # `hf://` streaming-URI bug seen with the deprecated `wikitext` id.
        from datasets import load_dataset
        ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train")
        docs, total = [], 0
        for row in ds:
            t = row["text"]
            if t and t.strip():
                docs.append(t); total += len(t)
            if max_chars and total >= max_chars:
                break
        prov = {"hf_dataset": "Salesforce/wikitext", "config": "wikitext-103-raw-v1", "streaming": False}
        return docs, prov
    if dataset == "wikitext2":
        train = _http_text(WIKITEXT2_URLS["train"])
        # WikiText mirror uses <unk> and @-@ artifacts; restore readable text.
        train = train.replace(" @-@ ", "-").replace(" @.@ ", ".").replace(" @,@ ", ",")
        docs = [d for d in train.split(" \n \n ") if d.strip()] or train.split("\n\n")
        prov = {"url": WIKITEXT2_URLS["train"]}
        return docs, prov
    if dataset == "shakespeare":
        txt = _http_text(SHAKESPEARE_URL)
        return txt.split("\n\n"), {"url": SHAKESPEARE_URL}
    raise ValueError(f"unknown dataset {dataset!r}")


def main():
    ap = argparse.ArgumentParser(description="Prepare a SHNU-LLM v0.2 training corpus.")
    ap.add_argument("--dataset", default="wikitext103", choices=list(LICENSES))
    ap.add_argument("--out-dir", required=True, help="durable output dir (e.g. a Google Drive path)")
    ap.add_argument("--vocab-size", type=int, default=16000)
    ap.add_argument("--val-ratio", type=float, default=0.01)
    ap.add_argument("--max-chars", type=int, default=None, help="optional cap on corpus size (chars)")
    ap.add_argument("--min-chars", type=int, default=64, help="drop records shorter than this")
    ap.add_argument("--tokenizer-train-records", type=int, default=200_000)
    args = ap.parse_args()

    t0 = time.time()
    os.makedirs(args.out_dir, exist_ok=True)
    for sub in ["tokenizer", "datasets", "logs"]:
        os.makedirs(os.path.join(args.out_dir, sub), exist_ok=True)

    print(f"[1/6] acquiring dataset={args.dataset}")
    raw_docs, provenance = acquire(args.dataset, args.max_chars)
    raw_chars = sum(len(d) for d in raw_docs)
    raw_sha = hashlib.sha256("".join(raw_docs).encode("utf-8")).hexdigest()

    print(f"[2/6] cleaning + deduplicating ({len(raw_docs):,} raw records)")
    records, clean_stats = S.clean_records(raw_docs, min_chars=args.min_chars)

    print(f"[3/6] training/loading tokenizer (vocab={args.vocab_size})")
    tok_path = os.path.join(args.out_dir, "tokenizer", "shnu_bpe.json")
    if os.path.exists(tok_path):
        tok = S.load_tokenizer(tok_path)
    else:
        tok = S.train_tokenizer(iter(records[:args.tokenizer_train_records]), args.vocab_size, tok_path)
    vocab_size = tok.get_vocab_size()
    eos_id = tok.token_to_id("<eos>")
    probe = "SHNU-LLM v0.2 tokenizer round-trip."
    roundtrip_ok = S.decode(tok, S.encode(tok, probe)).strip() == probe.strip()
    specials_ok = all(tok.token_to_id(t) is not None for t in S.SPECIAL_TOKENS)
    tok_sha = hashlib.sha256(open(tok_path, "rb").read()).hexdigest()

    print("[4/6] tokenizing + packing")
    stream = S.build_token_stream(tok, records, eos_id)
    train_ids, val_ids = S.train_val_split(stream, val_ratio=args.val_ratio)

    print("[5/6] saving tokenized streams to Drive")
    ver = datetime.date.today().isoformat()
    np.save(os.path.join(args.out_dir, "datasets", f"{args.dataset}_train_v{vocab_size}.npy"), train_ids)
    np.save(os.path.join(args.out_dir, "datasets", f"{args.dataset}_val_v{vocab_size}.npy"), val_ids)

    meta = {
        "dataset": args.dataset,
        "license": LICENSES[args.dataset],
        "provenance": provenance,
        "prepared_date": ver,
        "raw_records": len(raw_docs),
        "raw_chars": raw_chars,
        "raw_sha256": raw_sha,
        "clean_stats": clean_stats,
        "documents_kept": len(records),
        "tokenizer": {"vocab_size": vocab_size, "path": tok_path, "sha256": tok_sha,
                      "roundtrip_ok": roundtrip_ok, "special_tokens_ok": specials_ok,
                      "special_tokens": S.SPECIAL_TOKENS},
        "tokens": {"total": int(len(stream)), "train": int(len(train_ids)), "val": int(len(val_ids)),
                   "val_ratio": args.val_ratio},
        "chars_per_token": round(raw_chars / max(1, len(stream)), 3),
        "prepare_seconds": round(time.time() - t0, 1),
        "no_pretrained_weights": True,
        "no_paid_apis": True,
    }
    meta_path = os.path.join(args.out_dir, "datasets", f"{args.dataset}_metadata.json")
    json.dump(meta, open(meta_path, "w"), indent=2)

    print("[6/6] done")
    print(json.dumps(meta, indent=2))
    print("metadata ->", meta_path)


if __name__ == "__main__":
    main()
