#!/usr/bin/env python3
"""Train the SHNU-LLM byte-level BPE tokenizer from scratch on a text corpus.

The tokenizer is trained on the project corpus itself; it is not downloaded
from any pretrained model.

Usage:
    python tokenizer/train_tokenizer.py --input corpus.txt --vocab-size 16000 \
        --out tokenizer/shnu_bpe.json
"""
from __future__ import annotations

import os
import sys
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import shnu_llm as S


def main():
    ap = argparse.ArgumentParser(description="Train the SHNU-LLM BPE tokenizer.")
    ap.add_argument("--input", required=True, help="UTF-8 text file (one document per blank-line block)")
    ap.add_argument("--vocab-size", type=int, default=16000)
    ap.add_argument("--out", default="tokenizer/shnu_bpe.json")
    args = ap.parse_args()

    text = open(args.input, encoding="utf-8").read()
    records, stats = S.clean_records(text.split("\n\n"), min_chars=1)
    print(f"corpus: {stats}")
    tok = S.train_tokenizer(iter(records), args.vocab_size, args.out)
    probe = "SHNU-LLM tokenizer round-trip check."
    ok = S.decode(tok, S.encode(tok, probe)).strip() == probe.strip()
    print(f"trained vocab={tok.get_vocab_size()} -> {args.out}; roundtrip_ok={ok}")


if __name__ == "__main__":
    main()
