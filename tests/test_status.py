"""Tests for the persistent training-status monitor (CPU-only, no GPU, no training).

Covers: status creation, update, atomic replacement, missing status, malformed
status, checkpoint/status reconciliation, stale-status recovery, completed state,
interrupted state, and the mandatory rule that a valid checkpoint is never reset to
step 0 because the status is stale/missing/corrupt.
"""
from __future__ import annotations

import os
import sys
import json
import glob
import tempfile
from dataclasses import asdict

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import shnu_llm as S


def _cfg(max_steps=20000):
    return S.ShnuConfig(vocab_size=16000, n_layers=8, n_heads=8, d_model=512,
                        block_size=512, batch_size=16, grad_accum_steps=4,
                        max_steps=max_steps, dtype="float32", seed=1337)


def _fake_checkpoint(ckpt_dir, step, cfg, name="checkpoint_latest.pt", best_val=1.23):
    """Write a minimal but real .pt checkpoint (no model needed) for reconciliation tests."""
    os.makedirs(ckpt_dir, exist_ok=True)
    path = os.path.join(ckpt_dir, name)
    torch.save({"step": step, "config": asdict(cfg), "best_val": best_val}, path)
    return path


def _new_status(root):
    st = S.TrainingStatus.for_run(root)
    return st


def test_status_creation():
    root = tempfile.mkdtemp()
    st = _new_status(root)
    cfg = _cfg()
    data = st.default(cfg, git_commit="abc123")
    st.write(data)
    loaded = st.load_or_none()
    assert loaded is not None
    # required fields present
    for k in ["project", "version", "config", "git_commit", "state", "current_step",
              "target_steps", "tokens_processed", "target_tokens", "completion_percent",
              "train_loss", "validation_loss", "perplexity", "best_validation_loss",
              "best_validation_step", "latest_checkpoint", "best_checkpoint",
              "checkpoint_timestamp", "session_count", "resume_count", "elapsed_seconds",
              "throughput_tokens_per_second", "device", "gpu_name", "gpu_memory_gb",
              "last_update_timestamp", "last_error", "reconciliation"]:
        assert k in loaded, f"missing field {k}"
    assert loaded["target_steps"] == 20000
    assert loaded["config"]["tokens_per_optimizer_step"] == 16 * 4 * 512 == 32768
    assert loaded["state"] in S.VALID_STATES


def test_status_update_recomputes_derived():
    root = tempfile.mkdtemp()
    st = _new_status(root)
    cfg = _cfg()
    st.write(st.default(cfg))
    d = st.update(cfg, current_step=1000, validation_loss=5.0)
    assert d["tokens_processed"] == 1000 * 32768
    assert abs(d["completion_percent"] - (100.0 * 1000 / 20000)) < 1e-6
    assert d["perplexity"] is not None and d["perplexity"] > 0  # exp(5.0)
    # invalid state rejected
    try:
        st.update(cfg, state="banana")
        assert False, "invalid state should raise"
    except ValueError:
        pass


def test_atomic_replacement_leaves_no_tmp():
    root = tempfile.mkdtemp()
    st = _new_status(root)
    cfg = _cfg()
    for i in range(5):
        st.write(st.default(cfg) if i == 0 else st.load_or_none())
    d = os.path.dirname(st.status_path)
    assert glob.glob(os.path.join(d, "*.tmp")) == []      # no temp litter
    assert st.load_or_none() is not None                   # file is valid JSON


def test_missing_status():
    root = tempfile.mkdtemp()
    st = _new_status(root)
    assert st.load_or_none() is None
    # render must not crash on a missing file
    assert "No status file" in st.render()


def test_malformed_status_treated_as_missing():
    root = tempfile.mkdtemp()
    st = _new_status(root)
    os.makedirs(os.path.dirname(st.status_path), exist_ok=True)
    with open(st.status_path, "w") as f:
        f.write("{ this is not valid json ")
    assert st.load_or_none() is None      # corrupt -> None, never raises


def test_reconciliation_checkpoint_wins():
    root = tempfile.mkdtemp()
    st = _new_status(root)
    cfg = _cfg()
    # status says step 300, checkpoint says 500 -> trust checkpoint
    st.write(st.update(cfg, current_step=300))
    _fake_checkpoint(st.checkpoints_dir, 500, cfg)
    d = st.reconcile(cfg)
    assert d["current_step"] == 500
    assert d["reconciliation"]["occurred"] is True
    assert d["reconciliation"]["checkpoint_step"] == 500
    assert d["reconciliation"]["status_step_before"] == 300


def test_stale_status_recovery():
    root = tempfile.mkdtemp()
    st = _new_status(root)
    cfg = _cfg()
    st.write(st.update(cfg, current_step=100))     # very stale
    _fake_checkpoint(st.checkpoints_dir, 700, cfg)
    d = st.reconcile(cfg)
    assert d["current_step"] == 700                # recovered from checkpoint
    assert d["tokens_processed"] == 700 * 32768


def test_completed_state():
    root = tempfile.mkdtemp()
    st = _new_status(root)
    cfg = _cfg(max_steps=10)
    _fake_checkpoint(st.checkpoints_dir, 10, cfg)   # reached target
    d = st.reconcile(cfg)
    assert d["current_step"] == 10
    assert d["state"] == "completed"


def test_interrupted_state_persists():
    root = tempfile.mkdtemp()
    st = _new_status(root)
    cfg = _cfg()
    st.write(st.default(cfg))
    st.update(cfg, state="interrupted", last_error="colab disconnect")
    d = st.load_or_none()
    assert d["state"] == "interrupted" and d["last_error"] == "colab disconnect"


def test_valid_checkpoint_never_reset_to_zero():
    root = tempfile.mkdtemp()
    st = _new_status(root)
    cfg = _cfg()
    # corrupt status + a valid checkpoint at step 900
    os.makedirs(os.path.dirname(st.status_path), exist_ok=True)
    open(st.status_path, "w").write("not json at all")
    _fake_checkpoint(st.checkpoints_dir, 900, cfg)
    d = st.reconcile(cfg)
    assert d["current_step"] == 900               # NOT reset to 0
    assert d["reconciliation"]["occurred"] is True
    assert "repaired" in (d["reconciliation"]["reason"] or "").lower() \
        or "missing" in (d["reconciliation"]["reason"] or "").lower()


def test_fresh_run_when_no_checkpoint():
    root = tempfile.mkdtemp()
    st = _new_status(root)
    cfg = _cfg()
    d = st.reconcile(cfg)                          # no checkpoint at all
    assert d["current_step"] == 0
    assert d["fresh_run"] is True
    assert d["reconciliation"]["checkpoint_found"] is False


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print("ok:", name)
    print("PASS")
