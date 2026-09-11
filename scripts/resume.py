#!/usr/bin/env python3
"""Autonomously resume SHNU-LLM training from durable (Drive) state.

Fresh runtime -> this script loads the tokenizer and pre-tokenized streams already
on Drive, discovers the newest checkpoint, verifies architecture compatibility,
restores model+optimizer+step+RNG, and continues from the exact saved step. No
step numbers are ever edited by hand.

Fail-safe: if NO checkpoint is found it STOPS by default (so an interrupted long
run is never silently replaced by a brand-new run). Pass --allow-fresh to start a
run from scratch on purpose.

Usage:
    python scripts/resume.py --config configs/v0.2.json \
        --out-dir /content/drive/MyDrive/SHNU_LLM --dataset wikitext103
"""
from __future__ import annotations

import os
import sys
import json
import math
import argparse

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import shnu_llm as S


def load_data(out_dir: str, dataset: str, vocab_size: int, block_size: int, device: str):
    ds_dir = os.path.join(out_dir, "datasets")
    tr = np.load(os.path.join(ds_dir, f"{dataset}_train_v{vocab_size}.npy"))
    va = np.load(os.path.join(ds_dir, f"{dataset}_val_v{vocab_size}.npy"))
    return {"train": S.PackedDataset(tr, block_size, device),
            "val": S.PackedDataset(va, block_size, device)}, int(len(tr)), int(len(va))


def resume_training(cfg, out_dir: str, dataset: str, allow_fresh: bool = False,
                    device: str | None = None, verbose: bool = True):
    """Discover + verify + resume. Returns the train() result dict. Raises SystemExit
    on a missing checkpoint unless allow_fresh=True."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    cfg.out_dir = out_dir

    tok = S.load_tokenizer(os.path.join(out_dir, "tokenizer", "shnu_bpe.json"))
    cfg.vocab_size = tok.get_vocab_size()
    eos_id = tok.token_to_id("<eos>")
    datasets, ntr, nva = load_data(out_dir, dataset, cfg.vocab_size, cfg.block_size, device)
    if verbose:
        print(f"[resume] tokenizer vocab={cfg.vocab_size} | train_tokens={ntr:,} val_tokens={nva:,}")

    ckpt_dir = os.path.join(out_dir, "checkpoints")
    latest = S.find_latest_checkpoint(ckpt_dir)
    if latest is None:
        if not allow_fresh:
            raise SystemExit("[resume] FAIL-SAFE: no checkpoint found and --allow-fresh not set. "
                             "Refusing to start a new run automatically.")
        if verbose:
            print("[resume] no checkpoint; --allow-fresh set -> starting a fresh run at step 0")
    else:
        ck = torch.load(latest, map_location=device, weights_only=False)
        ok, mism = S.verify_checkpoint_compatible(ck, cfg)
        if not ok:
            raise SystemExit(f"[resume] checkpoint architecture mismatch: {mism}")
        if verbose:
            print(f"[resume] found {os.path.basename(latest)} @ step {S.checkpoint_step(latest)} (compatible)")

    S.set_seed(cfg.seed)
    model = S.ShnuLM(cfg).to(device)
    result = S.train(model, datasets, cfg, device, resume_from=latest,
                     tokenizer=tok, eos_id=eos_id, sample_prompt="The ", verbose=verbose)
    if verbose:
        print(f"[resume] step {result['start_step']} -> {result['steps']} "
              f"| val_loss {result['final_val_loss']:.4f} ppl {math.exp(result['final_val_loss']):.2f}")
    return result


def main():
    ap = argparse.ArgumentParser(description="Resume SHNU-LLM training from durable state.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--out-dir", required=True, help="durable dir (e.g. a Google Drive path)")
    ap.add_argument("--dataset", default="wikitext103")
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--allow-fresh", action="store_true",
                    help="permit starting from scratch when no checkpoint exists")
    args = ap.parse_args()

    cfg_dict = json.load(open(args.config))
    if args.max_steps is not None:
        cfg_dict["max_steps"] = args.max_steps
    cfg = S.ShnuConfig(**cfg_dict)
    resume_training(cfg, args.out_dir, args.dataset, allow_fresh=args.allow_fresh)


if __name__ == "__main__":
    main()
