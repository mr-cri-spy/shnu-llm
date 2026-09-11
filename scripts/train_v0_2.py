#!/usr/bin/env python3
"""Guarded entrypoint for the SHNU-LLM v0.2 LONG training run.

This script will NOT train unless every safety gate passes AND you pass
--confirm-long-run. It exists specifically to prevent accidentally launching a
long run in an unsafe environment (no GPU / not a T4 / wrong data / no Drive /
low disk / incompatible checkpoint). It never falls back to CPU.

Dry run (default) — runs pre-flight checks and STOPS:
    python scripts/train_v0_2.py --out-dir /content/drive/MyDrive/SHNU_LLM

Real run — only after a passing pilot and a deliberate decision:
    python scripts/train_v0_2.py --out-dir /content/drive/MyDrive/SHNU_LLM --confirm-long-run
"""
from __future__ import annotations

import os
import sys
import json
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import shnu_llm as S
from shnu_llm.preflight import PreflightError

# Verified v0.2 baseline integrity values (see docs/DATASET.md).
EXPECTED = {
    "dataset_sha256_prefix": "1f9f77ec26e7477b7bf3c80c72a2e365749731f",
    "tokenizer_sha256_prefix": "1bf9397a6213d1a303c9d149d028fa2c456dc",
    "vocab_size": 16000,
    "train_tokens": 116995555,
    "val_tokens": 1181773,
}
REPO = os.path.join(os.path.dirname(__file__), "..")


def main():
    ap = argparse.ArgumentParser(description="Guarded SHNU-LLM v0.2 long-training entrypoint.")
    ap.add_argument("--config", default=os.path.join(REPO, "configs", "v0.2.json"))
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--dataset", default="wikitext103")
    ap.add_argument("--require-t4", action="store_true", default=True)
    ap.add_argument("--allow-non-t4", dest="require_t4", action="store_false",
                    help="permit a non-T4 GPU (still never CPU)")
    ap.add_argument("--min-free-gb", type=float, default=3.0)
    ap.add_argument("--confirm-long-run", action="store_true",
                    help="REQUIRED to actually start long training")
    args = ap.parse_args()

    cfg = S.ShnuConfig(**json.load(open(args.config)))
    cfg.out_dir = args.out_dir
    ds_meta = os.path.join(args.out_dir, "datasets", f"{args.dataset}_metadata.json")
    tok_path = os.path.join(args.out_dir, "tokenizer", "shnu_bpe.json")
    ckpt = S.find_latest_checkpoint(os.path.join(args.out_dir, "checkpoints"))

    print("== SHNU-LLM v0.2 pre-flight ==")
    try:
        report = S.run_preflight(
            cfg, require_t4=args.require_t4, min_free_gb=args.min_free_gb,
            dataset_metadata=ds_meta, tokenizer_path=tok_path,
            expected=EXPECTED, checkpoint_path=ckpt,
        )
    except PreflightError as e:
        print(f"PREFLIGHT FAILED: {e}")
        print("Long training will NOT start. Fix the issue above and retry.")
        sys.exit(2)

    print(json.dumps(report, indent=2, default=str))
    print("PREFLIGHT PASSED.")

    if not args.confirm_long_run:
        print("\nDRY RUN ONLY — pass --confirm-long-run to actually start long training.")
        print("Long training NOT started.")
        return

    # All gates passed and long run explicitly confirmed.
    from resume import resume_training
    print("\n== starting LONG training run (confirmed) ==")
    resume_training(cfg, args.out_dir, args.dataset, allow_fresh=True)


if __name__ == "__main__":
    main()
