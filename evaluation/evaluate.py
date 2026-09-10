#!/usr/bin/env python3
"""Evaluate a trained SHNU-LLM checkpoint: validation loss, perplexity, samples.

Usage:
    python evaluation/evaluate.py --model runs/shnu_t4/final/shnu_llm_v0.1.pt \
        --tokenizer runs/shnu_t4/final/tokenizer/shnu_bpe.json --corpus corpus.txt
"""
from __future__ import annotations

import os
import sys
import json
import math
import argparse

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import shnu_llm as S


def main():
    ap = argparse.ArgumentParser(description="Evaluate a SHNU-LLM checkpoint.")
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--corpus", required=True, help="held-out text for perplexity")
    ap.add_argument("--eval-iters", type=int, default=200)
    ap.add_argument("--prompts", nargs="*", default=["The ", "In the "])
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.model, map_location=device, weights_only=False)
    cfg = S.ShnuConfig(**ckpt["config"])
    model = S.ShnuLM(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    tok = S.load_tokenizer(args.tokenizer)
    eos_id = tok.token_to_id("<eos>")

    text = open(args.corpus, encoding="utf-8").read()
    records, _ = S.clean_records(text.split("\n\n"), min_chars=1)
    stream = S.build_token_stream(tok, records, eos_id)
    ds = {"val": S.PackedDataset(stream, cfg.block_size, device)}
    cfg.eval_iters = args.eval_iters
    gen = torch.Generator().manual_seed(0)
    from contextlib import nullcontext
    metrics = S.estimate_loss(model, ds, cfg, gen, nullcontext())
    val = metrics["val"]

    report = {"parameters": model.num_params(), "vocabulary": cfg.vocab_size,
              "context_length": cfg.block_size, "val_loss": round(val, 4),
              "perplexity": round(math.exp(val), 2), "gpu": S.env_report(device).get("gpu_name", "CPU")}
    print(json.dumps(report, indent=2))
    for p in args.prompts:
        print(f"[{p!r}] ->",
              repr(S.sample_text(model, tok, cfg, device, p, 80, temperature=0.8, top_k=50, eos_id=eos_id)))


if __name__ == "__main__":
    main()
