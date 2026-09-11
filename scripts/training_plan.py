#!/usr/bin/env python3
"""Compute a realistic SHNU-LLM v0.2 training plan for a free Colab T4.

Everything printed is derived deterministically from the config and the verified
dataset size, or measured on CPU (checkpoint size). Throughput / wall-clock / GPU
memory are T4-dependent and are reported as UNKNOWN until actually measured by the
pilot — this script never invents them.

Usage:
  python scripts/training_plan.py --config configs/v0.2.json --train-tokens 116995555
"""
from __future__ import annotations

import os
import sys
import json
import argparse
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import shnu_llm as S
import torch


def measured_checkpoint_mb(cfg) -> float:
    """Build the model + AdamW state and save one checkpoint on CPU to measure real size."""
    m = S.ShnuLM(cfg)
    opt = S.configure_optimizer(m, cfg)
    x = torch.randint(0, cfg.vocab_size, (2, cfg.block_size))
    y = torch.randint(0, cfg.vocab_size, (2, cfg.block_size))
    _, loss = m(x, y); loss.backward(); opt.step()      # populate optimizer moments
    d = tempfile.mkdtemp()
    p = os.path.join(d, "ck.pt")
    S.save_checkpoint(p, m, opt, cfg, step=1, best_val=1.0,
                      log={"step": [], "train_loss": [], "val_loss": [], "lr": []})
    return round(os.path.getsize(p) / 1e6, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/v0.2.json")
    ap.add_argument("--train-tokens", type=int, default=116_995_555)
    ap.add_argument("--val-tokens", type=int, default=1_181_773)
    ap.add_argument("--keep-last-checkpoints", type=int, default=2)
    args = ap.parse_args()

    cfg = S.ShnuConfig(**json.load(open(args.config)))
    tps = cfg.batch_size * cfg.grad_accum_steps * cfg.block_size
    steps_per_epoch = args.train_tokens / tps
    params = S.ShnuLM(cfg).num_params()
    ckpt_mb = measured_checkpoint_mb(cfg)

    # Recommendation: v0.2 has ~117M train tokens (~3.5 tokens/param). A practical,
    # resumable target on free T4 is 2-3 passes; we recommend ~2 epochs as the default
    # v0.2 baseline (enough to clearly beat v0.1's regime without over-committing free compute).
    rec_epochs = 2
    rec_tokens = rec_epochs * args.train_tokens
    rec_steps = round(rec_tokens / tps)
    # bounded checkpoint storage: latest + best + keep_last numbered step checkpoints
    peak_ckpt_files = 2 + max(0, args.keep_last_checkpoints)

    plan = {
        "config": args.config,
        "model_parameters": params,
        "sequence_length": cfg.block_size,
        "batch_size": cfg.batch_size,
        "gradient_accumulation": cfg.grad_accum_steps,
        "effective_batch_size": cfg.batch_size * cfg.grad_accum_steps,
        "learning_rate": cfg.lr,
        "warmup_steps": cfg.warmup_steps,
        "tokens_per_optimizer_step": tps,
        "train_tokens_available": args.train_tokens,
        "val_tokens_available": args.val_tokens,
        "steps_per_epoch": round(steps_per_epoch, 1),
        "recommended_epochs": rec_epochs,
        "recommended_training_tokens": rec_tokens,
        "recommended_total_steps": rec_steps,
        "checkpoint_interval_steps": 500,
        "validation_interval_steps": 500,
        "keep_last_checkpoints": args.keep_last_checkpoints,
        "measured_checkpoint_size_mb": ckpt_mb,
        "expected_peak_checkpoint_storage_mb": round(ckpt_mb * peak_ckpt_files, 1),
        "config_max_steps": cfg.max_steps,
        # ---- T4-dependent: not known until the pilot measures them ----
        "T4_tokens_per_second": "UNKNOWN (measure with pilot)",
        "T4_gpu_memory_gb": "UNKNOWN (measure with pilot; analytic estimate ~5-7 GB fp16 at this config)",
        "T4_estimated_runtime": "UNKNOWN (= recommended_training_tokens / measured_tokens_per_second)",
        "runtime_note": "free T4 sessions time out / disconnect; the run MUST be resumable and "
                        "checkpoint often (every 500 steps here).",
    }
    print(json.dumps(plan, indent=2))


if __name__ == "__main__":
    main()
