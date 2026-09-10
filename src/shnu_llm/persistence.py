"""Checkpoint discovery and compatibility verification for resumable training.

Permanent training state (checkpoints, optimizer/scheduler state, tokenizer,
datasets, logs, samples) is meant to live on durable storage such as Google
Drive — never on the ephemeral Colab `/content` filesystem. These helpers locate
the latest checkpoint under a given directory and verify that it matches the
current architecture before resuming.
"""
from __future__ import annotations

import os
import re
import glob
from typing import Optional, Tuple

import torch

from .config import ShnuConfig

# Architecture fields that must match for a checkpoint to be safe to resume.
_ARCH_KEYS = ("n_layers", "n_heads", "n_kv_heads", "d_model", "d_ff",
              "block_size", "vocab_size", "tie_embeddings")


def checkpoint_step(path: str) -> int:
    """Best-effort training step for a checkpoint path (filename, then payload)."""
    m = re.search(r"step_(\d+)", os.path.basename(path))
    if m:
        return int(m.group(1))
    try:
        return int(torch.load(path, map_location="cpu", weights_only=False).get("step", -1))
    except Exception:
        return -1


def find_latest_checkpoint(ckpt_dir: str) -> Optional[str]:
    """Return the most advanced checkpoint in `ckpt_dir`, or None if there is none.

    Prefers `checkpoint_latest.pt`; otherwise picks the highest `checkpoint_step_*.pt`.
    """
    if not ckpt_dir or not os.path.isdir(ckpt_dir):
        return None
    latest = os.path.join(ckpt_dir, "checkpoint_latest.pt")
    if os.path.exists(latest):
        return latest
    candidates = glob.glob(os.path.join(ckpt_dir, "checkpoint_step_*.pt"))
    if not candidates:
        return None
    return max(candidates, key=checkpoint_step)


def verify_checkpoint_compatible(ckpt: dict, cfg: ShnuConfig) -> Tuple[bool, dict]:
    """Check that a checkpoint's saved architecture matches `cfg`.

    Returns (is_compatible, mismatches) where mismatches maps each differing
    field to (checkpoint_value, config_value).
    """
    saved = ckpt.get("config", {}) or {}
    mismatches = {
        k: (saved.get(k), getattr(cfg, k))
        for k in _ARCH_KEYS
        if k in saved and saved.get(k) != getattr(cfg, k)
    }
    return (len(mismatches) == 0, mismatches)
