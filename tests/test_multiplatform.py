"""Tests for the platform abstraction, Kaggle launcher, and safe benchmark.

CPU-only, tempdir scratch. Verifies:
  * platform lookup / detection / GPU description;
  * the Kaggle launcher dry-runs through the SAME core (verify → gate →
    reconcile → stop) and, when confirmed, resumes from the newest checkpoint
    (never from step 0) — all on scratch, GPU gate skipped for CPU;
  * the Kaggle static-asset staging + verified bundle import path;
  * the benchmark measures throughput and writes NOTHING to disk, and provably
    never opens the real checkpoint file.
"""
from __future__ import annotations

import os
import sys
import json
import glob
import time
import types
import hashlib
import random
import tempfile
import importlib.util

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import shnu_llm as S
from shnu_llm.preflight import PreflightError

_SCRIPTS = os.path.join(os.path.dirname(__file__), "..", "scripts")


def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_SCRIPTS, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---- platform abstraction -------------------------------------------------

def test_platform_lookup_and_detect():
    assert S.get_platform("colab").require_t4 is True
    assert S.get_platform("kaggle").require_t4 is False
    assert S.get_platform("kaggle").uses_drive_mount is False
    assert S.detect_platform().name in ("colab", "kaggle", "generic")
    g = S.describe_gpu()
    assert set(("available", "name", "is_t4")).issubset(g)


# ---- scratch run assets (mirrors test_launcher.py) ------------------------

def _scratch(root):
    os.makedirs(root + "/tokenizer"); os.makedirs(root + "/datasets")
    words = [f"w{i:02d}" for i in range(50)]
    docs = [" ".join(random.Random(i).choice(words) for _ in range(40)) for i in range(300)]
    tok = S.train_tokenizer(iter(docs), 300, root + "/tokenizer/shnu_bpe.json")
    V = tok.get_vocab_size()
    eos = tok.token_to_id("<eos>"); stream = S.build_token_stream(tok, docs, eos)
    tr, va = S.train_val_split(stream, 0.1)
    np.save(root + f"/datasets/scratch_train_v{V}.npy", tr)
    np.save(root + f"/datasets/scratch_val_v{V}.npy", va)
    raw_sha = hashlib.sha256(("\n\n".join(docs)).encode()).hexdigest()
    tok_sha = hashlib.sha256(open(root + "/tokenizer/shnu_bpe.json", "rb").read()).hexdigest()
    json.dump({"raw_sha256": raw_sha, "tokens": {"train": int(len(tr)), "val": int(len(va))}},
              open(root + "/datasets/scratch_metadata.json", "w"))
    exp = dict(dataset_sha256_prefix=raw_sha[:20], tokenizer_sha256_prefix=tok_sha[:20],
               vocab_size=V, train_tokens=int(len(tr)), val_tokens=int(len(va)),
               arch={"vocab_size": V, "n_layers": 2, "n_heads": 2, "d_model": 64, "block_size": 64})
    return V, exp


def _cfg_path(root, V, maxs):
    c = dict(vocab_size=V, n_layers=2, n_heads=2, d_model=64, block_size=64, batch_size=8,
             grad_accum_steps=1, max_steps=maxs, warmup_steps=2, eval_interval=3, eval_iters=3,
             sample_interval=10**9, checkpoint_interval=3, log_interval=10**9, dtype="float32", seed=1337)
    p = root + f"/cfg_{maxs}.json"; json.dump(c, open(p, "w"))
    S.set_seed(1337)
    return p, S.ShnuLM(S.ShnuConfig(**c)).num_params()


def test_kaggle_launch_dry_run_then_resume():
    KL = _load("kaggle_launch")
    root = tempfile.mkdtemp()
    V, exp = _scratch(root)
    cfg6, params = _cfg_path(root, V, 6); exp["params"] = params
    common = dict(dataset="scratch", expected=exp, require_gpu_gate=False, require_t4=False,
                  min_free_gb=0.0001, verbose=False)

    # dry run: same core, stops before training
    rA = KL.launch(root, config_path=cfg6, confirm_long_run=False, allow_fresh=True, **common)
    assert rA["trained"] is False and rA["dry_run"] is True and rA["platform"] == "kaggle"
    assert rA["reconciled_step"] == 0 and rA["target_steps"] == 6

    # confirmed fresh run 0 -> 6 on scratch
    rB = KL.launch(root, config_path=cfg6, confirm_long_run=True, allow_fresh=True, **common)
    d = S.TrainingStatus.for_run(root).load_or_none()
    assert rB["trained"] and d["current_step"] == 6

    # resume 6 -> 9 (never from zero)
    cfg9, _ = _cfg_path(root, V, 9)
    rC = KL.launch(root, config_path=cfg9, confirm_long_run=True, allow_fresh=False, **common)
    assert rC["result"]["start_step"] == 6 and rC["result"]["steps"] == 9


def test_kaggle_refuses_fresh_without_checkpoint():
    KL = _load("kaggle_launch")
    root = tempfile.mkdtemp()
    V, exp = _scratch(root)
    cfg6, params = _cfg_path(root, V, 6); exp["params"] = params
    # confirmed run, no checkpoint, allow_fresh=False -> must refuse (SystemExit from resume)
    try:
        KL.launch(root, config_path=cfg6, dataset="scratch", expected=exp, require_gpu_gate=False,
                  require_t4=False, min_free_gb=0.0001, confirm_long_run=True, allow_fresh=False,
                  verbose=False)
        assert False, "should have refused to start a fresh run"
    except SystemExit:
        pass


def test_kaggle_stage_and_import_bundle():
    KL = _load("kaggle_launch")
    # a "Drive-side" run with a checkpoint at step 6, exported as a bundle
    drive = tempfile.mkdtemp()
    V, exp = _scratch(drive)
    cfg6, params = _cfg_path(drive, V, 6); exp["params"] = params
    KL.launch(drive, config_path=cfg6, dataset="scratch", expected=exp, require_gpu_gate=False,
              require_t4=False, min_free_gb=0.0001, confirm_long_run=True, allow_fresh=True, verbose=False)
    cfg = S.ShnuConfig(**json.load(open(cfg6)))
    bundle = os.path.join(tempfile.mkdtemp(), "run.tar.gz")
    S.export_run_bundle(drive, bundle, dataset="scratch")

    # a fresh Kaggle working dir: stage static assets from a "data_root", import the bundle
    data_root = tempfile.mkdtemp()  # pretend attached read-only dataset
    os.makedirs(data_root + "/tokenizer"); os.makedirs(data_root + "/datasets")
    import shutil
    shutil.copy2(drive + "/tokenizer/shnu_bpe.json", data_root + "/tokenizer/shnu_bpe.json")
    for f in glob.glob(drive + "/datasets/*"):
        shutil.copy2(f, data_root + "/datasets/" + os.path.basename(f))

    work = tempfile.mkdtemp()
    KL.stage_static_assets(work, data_root, dataset="scratch", vocab_size=V, verbose=False)
    assert os.path.exists(work + "/tokenizer/shnu_bpe.json")
    rep = KL.import_bundle(bundle, work, cfg=S.ShnuConfig(**json.load(open(cfg6))), verbose=False)
    assert any("checkpoint_latest.pt" in a["arcname"] for a in rep["adopted"])
    got = S.inspect_checkpoint(work + "/checkpoints/checkpoint_latest.pt")
    assert got["valid"] and got["step"] == 6


# ---- safe benchmark -------------------------------------------------------

def test_benchmark_runs_and_writes_nothing():
    cfg = S.ShnuConfig(vocab_size=200, n_layers=2, n_heads=2, d_model=64, block_size=32,
                       batch_size=8, grad_accum_steps=2, max_steps=20000, seed=1337, dtype="float32")
    work = tempfile.mkdtemp()
    before = set(glob.glob(work + "/**", recursive=True))
    cwd_before = set(os.listdir("."))
    b = S.benchmark_throughput(cfg, device="cpu", steps=12, warmup=3, verbose=False)
    assert b["nan_or_inf_detected"] is False
    assert b["tokens_per_step"] == 8 * 2 * 32
    assert b["steps_measured"] == 9
    assert b["tokens_per_second"] and b["tokens_per_second"] > 0
    # wrote nothing to the scratch dir or cwd
    assert set(glob.glob(work + "/**", recursive=True)) == before
    assert set(os.listdir(".")) == cwd_before

    est = S.estimate_completion(b["tokens_per_second"], remaining_steps=17000, tokens_per_step=32768)
    assert est["is_estimate"] and est["remaining_tokens"] == 17000 * 32768
    assert any(m["is_full_remaining"] and m["steps"] == 17000 for m in est["milestones"])


def test_benchmark_never_opens_real_checkpoint(monkeypatch=None):
    """Guard: benchmarking must not call torch.load at all (it reads no checkpoint)."""
    import torch
    cfg = S.ShnuConfig(vocab_size=200, n_layers=2, n_heads=2, d_model=64, block_size=32,
                       batch_size=8, grad_accum_steps=1, max_steps=20000, seed=1337, dtype="float32")
    calls = {"n": 0}
    real_load = torch.load

    def spy_load(*a, **k):
        calls["n"] += 1
        return real_load(*a, **k)

    torch.load = spy_load
    try:
        S.benchmark_throughput(cfg, device="cpu", steps=8, warmup=2, verbose=False)
    finally:
        torch.load = real_load
    assert calls["n"] == 0, "benchmark must never load a checkpoint"


if __name__ == "__main__":
    for fn in list(globals()):
        if fn.startswith("test_"):
            globals()[fn]()
    print("PASS")
