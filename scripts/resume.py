#!/usr/bin/env python3
"""Resume SHNU-LLM pretraining from the latest checkpoint on durable storage.

Discovers the most advanced checkpoint under `<out-dir>/checkpoints`, verifies it
matches the requested architecture, and continues training from the exact saved
step. Intended to point `--out-dir` at a Google Drive path so state survives
Colab runtime recycling.

Usage:
    python scripts/resume.py --config configs/t4.json \
        --out-dir /content/drive/MyDrive/SHNU_LLM --corpus /path/to/corpus.txt
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
    ap = argparse.ArgumentParser(description="Resume SHNU-LLM training from a checkpoint.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--out-dir", required=True, help="durable dir (e.g. a Google Drive path)")
    ap.add_argument("--corpus", required=True, help="text corpus for tokenizer/data")
    ap.add_argument("--max-steps", type=int, default=None)
    args = ap.parse_args()

    cfg_dict = json.load(open(args.config))
    cfg_dict["out_dir"] = args.out_dir
    if args.max_steps is not None:
        cfg_dict["max_steps"] = args.max_steps
    cfg = S.ShnuConfig(**cfg_dict)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # tokenizer: reuse the one saved on durable storage if present
    tok_path = os.path.join(cfg.out_dir, "tokenizer", "shnu_bpe.json")
    text = open(args.corpus, encoding="utf-8").read()
    records, _ = S.clean_records(text.split("\n\n"), min_chars=1)
    if os.path.exists(tok_path):
        tok = S.load_tokenizer(tok_path)
    else:
        tok = S.train_tokenizer(iter(records[:200_000]), cfg.vocab_size, tok_path)
    cfg.vocab_size = tok.get_vocab_size()
    eos_id = tok.token_to_id("<eos>")
    stream = S.build_token_stream(tok, records, eos_id)
    tr, va = S.train_val_split(stream, val_ratio=0.02)
    datasets = {"train": S.PackedDataset(tr, cfg.block_size, device),
                "val": S.PackedDataset(va, cfg.block_size, device)}

    ckpt_dir = os.path.join(cfg.out_dir, "checkpoints")
    latest = S.find_latest_checkpoint(ckpt_dir)
    if latest is None:
        print(f"[resume] no checkpoint under {ckpt_dir}; starting a fresh run")
    else:
        ckpt = torch.load(latest, map_location=device, weights_only=False)
        ok, mism = S.verify_checkpoint_compatible(ckpt, cfg)
        if not ok:
            raise SystemExit(f"[resume] checkpoint architecture mismatch: {mism}")
        print(f"[resume] latest={os.path.basename(latest)} step={S.checkpoint_step(latest)} (compatible)")

    S.set_seed(cfg.seed)
    model = S.ShnuLM(cfg).to(device)
    result = S.train(model, datasets, cfg, device, resume_from=latest,
                     tokenizer=tok, eos_id=eos_id, sample_prompt="The ", verbose=True)
    print(f"[resume] started at step {result['start_step']} -> finished at step {result['steps']} "
          f"(val loss {result['final_val_loss']:.4f}, ppl {math.exp(result['final_val_loss']):.2f})")


if __name__ == "__main__":
    main()
