"""Tests for the one-call Colab launcher/resumer (CPU-only, tiny scratch run).

Exercises the launcher end to end without a GPU: config/param verification, the
safety gates (with the GPU gate skipped for CPU), status reconciliation, a real
fresh scratch run, an automatic resume from the checkpoint (never from step 0),
the dry-run guard, and fail-closed behavior on a bad tokenizer hash.
"""
from __future__ import annotations

import os
import sys
import json
import glob
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


def _load_launcher():
    spec = importlib.util.spec_from_file_location("colab_launch", os.path.join(_SCRIPTS, "colab_launch.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


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
    p_params = S.ShnuLM(S.ShnuConfig(**c)).num_params()
    return p, p_params


def test_launcher_dry_run_then_fresh_then_resume():
    CL = _load_launcher()
    root = tempfile.mkdtemp()
    V, exp = _scratch(root)
    cfg6, params = _cfg_path(root, V, 6)
    exp["params"] = params
    common = dict(dataset="scratch", expected=exp, require_gpu_gate=False, require_t4=False,
                  drive_root=root, min_free_gb=0.0001, allow_fresh=True, verbose=False)

    # dry run never trains
    rA = CL.launch(root, config_path=cfg6, confirm_long_run=False, **common)
    assert rA["trained"] is False and rA["dry_run"] is True
    assert rA["reconciled_step"] == 0 and rA["target_steps"] == 6

    # real fresh run 0 -> 6
    rB = CL.launch(root, config_path=cfg6, confirm_long_run=True, **common)
    st = S.TrainingStatus.for_run(root); d = st.load_or_none()
    assert rB["trained"] and d["current_step"] == 6 and d["state"] == "completed"
    assert d["session_count"] == 1

    # resume 6 -> 9 (never from zero), larger target
    cfg9, _ = _cfg_path(root, V, 9)
    rC = CL.launch(root, config_path=cfg9, confirm_long_run=True, **common)
    d2 = st.load_or_none()
    assert rC["result"]["start_step"] == 6 and rC["result"]["steps"] == 9
    assert d2["current_step"] == 9 and d2["state"] == "completed"
    assert d2["session_count"] == 2 and d2["resume_count"] == 1
    # no runtime artifact leaked into the repo; status lives under the scratch root
    assert glob.glob(os.path.join(os.path.dirname(st.status_path), "*.tmp")) == []


def test_launcher_fails_closed_on_bad_tokenizer_hash():
    CL = _load_launcher()
    root = tempfile.mkdtemp()
    V, exp = _scratch(root)
    cfg6, params = _cfg_path(root, V, 6)
    exp["params"] = params
    exp["tokenizer_sha256_prefix"] = "deadbeefdeadbeef"
    try:
        CL.launch(root, config_path=cfg6, dataset="scratch", expected=exp, confirm_long_run=True,
                  require_gpu_gate=False, require_t4=False, drive_root=root, min_free_gb=0.0001, verbose=False)
        assert False, "should have raised PreflightError"
    except PreflightError:
        pass


def test_launcher_fails_closed_on_bad_params():
    CL = _load_launcher()
    root = tempfile.mkdtemp()
    V, exp = _scratch(root)
    cfg6, _ = _cfg_path(root, V, 6)
    exp["params"] = 999  # wrong
    try:
        CL.launch(root, config_path=cfg6, dataset="scratch", expected=exp, confirm_long_run=False,
                  require_gpu_gate=False, require_t4=False, drive_root=root, min_free_gb=0.0001, verbose=False)
        assert False, "should have raised PreflightError"
    except PreflightError:
        pass


if __name__ == "__main__":
    test_launcher_dry_run_then_fresh_then_resume()
    test_launcher_fails_closed_on_bad_tokenizer_hash()
    test_launcher_fails_closed_on_bad_params()
    print("PASS")
