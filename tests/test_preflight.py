"""Tests for the free-Colab safety pre-flight checks (CPU-only)."""
from __future__ import annotations

import os
import sys
import json
import tempfile

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import shnu_llm as S
from shnu_llm.preflight import PreflightError


def test_require_gpu_refuses_cpu():
    if torch.cuda.is_available():
        return  # this test only asserts the CPU-refusal behavior
    try:
        S.require_gpu(require_t4=True)
        assert False, "require_gpu should raise without a GPU"
    except PreflightError:
        pass


def test_free_disk():
    d = tempfile.mkdtemp()
    assert S.require_free_disk(d, min_gb=0.0001) > 0
    try:
        S.require_free_disk(d, min_gb=10**9)
        assert False
    except PreflightError:
        pass


def test_dataset_and_tokenizer_hash_checks():
    d = tempfile.mkdtemp()
    meta = {"raw_sha256": "abc123def", "tokens": {"train": 100, "val": 10}, "license": "CC BY-SA 3.0"}
    mp = os.path.join(d, "meta.json"); json.dump(meta, open(mp, "w"))
    assert S.verify_dataset(mp, "abc123", 100, 10)["tokens"]["train"] == 100
    for bad in [lambda: S.verify_dataset(mp, "zzzz"),
                lambda: S.verify_dataset(mp, "abc123", 999)]:
        try:
            bad(); assert False
        except PreflightError:
            pass
    tf = os.path.join(d, "tok.json"); open(tf, "w").write("tokenizer-bytes")
    import hashlib
    sha = hashlib.sha256(b"tokenizer-bytes").hexdigest()
    assert S.verify_tokenizer(tf, sha[:8])["tokenizer_sha256"] == sha
    try:
        S.verify_tokenizer(tf, "deadbeef"); assert False
    except PreflightError:
        pass


def test_checkpoint_compat_check():
    d = tempfile.mkdtemp()
    cfg = S.ShnuConfig(vocab_size=200, n_layers=2, n_heads=2, d_model=32, block_size=32, dtype="float32")
    m = S.ShnuLM(cfg); opt = S.configure_optimizer(m, cfg)
    p = os.path.join(d, "ck.pt")
    S.save_checkpoint(p, m, opt, cfg, 5, 1.0, {"step": [], "train_loss": [], "val_loss": [], "lr": []})
    assert S.verify_checkpoint(p, cfg)["checkpoint_step"] == 5
    bad_cfg = S.ShnuConfig(vocab_size=999, n_layers=2, n_heads=2, d_model=32, block_size=32, dtype="float32")
    try:
        S.verify_checkpoint(p, bad_cfg); assert False
    except PreflightError:
        pass


if __name__ == "__main__":
    test_require_gpu_refuses_cpu()
    test_free_disk()
    test_dataset_and_tokenizer_hash_checks()
    test_checkpoint_compat_check()
    print("PASS")
