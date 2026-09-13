"""Tests for cross-platform checkpoint safety (CPU-only, tempdir scratch).

Covers the guarantees that make multi-platform training safe:
  * a valid checkpoint is recognized; a corrupt/empty one is rejected;
  * architecture / SHA-256 verification is fail-closed;
  * a NEWER valid checkpoint is never replaced by an OLDER one;
  * a VALID checkpoint is never replaced by a CORRUPT one;
  * choose_authoritative picks the newest valid checkpoint;
  * export → import round-trips and is hash-verified;
  * a tampered bundle (hash mismatch) is refused.

Nothing here touches any real checkpoint — every path is a fresh tempdir.
"""
from __future__ import annotations

import os
import sys
import json
import tarfile
import tempfile

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import shnu_llm as S


def _tiny_cfg(V=200, **over):
    d = dict(vocab_size=V, n_layers=2, n_heads=2, d_model=64, block_size=32, batch_size=8,
             grad_accum_steps=1, max_steps=100, seed=1337, dtype="float32")
    d.update(over)
    return S.ShnuConfig(**d)


def _make_ckpt(out_dir, step, cfg=None, name="checkpoint_latest.pt"):
    """Write a real, loadable checkpoint at a given step under out_dir/checkpoints."""
    cfg = cfg or _tiny_cfg()
    cfg.out_dir = out_dir
    S.set_seed(1337)
    model = S.ShnuLM(cfg)
    opt = S.configure_optimizer(model, cfg)
    path = os.path.join(out_dir, "checkpoints", name)
    S.save_checkpoint(path, model, opt, cfg, step, best_val=1.23, log={"step": [step]})
    return path


def test_inspect_valid_and_corrupt():
    root = tempfile.mkdtemp()
    good = _make_ckpt(root, 42)
    info = S.inspect_checkpoint(good)
    assert info["valid"] and info["step"] == 42 and info["has_model"]
    assert info["sha256"] and info["size_bytes"] > 0

    # empty file
    empty = os.path.join(root, "checkpoints", "empty.pt")
    open(empty, "wb").close()
    assert S.inspect_checkpoint(empty)["valid"] is False

    # garbage file
    bad = os.path.join(root, "checkpoints", "garbage.pt")
    with open(bad, "wb") as f:
        f.write(b"not a torch checkpoint at all")
    assert S.inspect_checkpoint(bad)["valid"] is False

    # missing file
    assert S.inspect_checkpoint(os.path.join(root, "nope.pt"))["valid"] is False


def test_validate_arch_and_sha():
    root = tempfile.mkdtemp()
    cfg = _tiny_cfg()
    p = _make_ckpt(root, 10, cfg)
    ok, info = S.validate_checkpoint(p, cfg)
    assert ok and info["arch_compatible"]

    # architecture mismatch -> fail closed
    other = _tiny_cfg(d_model=128, n_heads=2)
    ok2, info2 = S.validate_checkpoint(p, other)
    assert ok2 is False and "arch" in (info2.get("reason") or "").lower()

    # sha mismatch -> fail closed
    ok3, info3 = S.validate_checkpoint(p, cfg, expected_sha256="deadbeef" * 8)
    assert ok3 is False and "sha256" in (info3.get("reason") or "").lower()

    # correct sha passes
    good_sha = S.sha256_file(p)
    ok4, _ = S.validate_checkpoint(p, cfg, expected_sha256=good_sha)
    assert ok4


def test_decide_replace_never_older_or_corrupt():
    cfg = _tiny_cfg()
    a = tempfile.mkdtemp(); b = tempfile.mkdtemp()
    existing = _make_ckpt(a, 9, cfg)      # newer
    older = _make_ckpt(b, 3, cfg)         # older

    # older must NOT replace newer
    d1 = S.decide_replace(existing, older, cfg)
    assert d1["action"] == "keep", d1

    # newer replaces older
    d2 = S.decide_replace(older, existing, cfg)
    assert d2["action"] == "adopt", d2

    # corrupt incoming is rejected regardless
    corrupt = os.path.join(b, "checkpoints", "corrupt.pt")
    with open(corrupt, "wb") as f:
        f.write(b"garbage")
    d3 = S.decide_replace(existing, corrupt, cfg)
    assert d3["action"] == "reject", d3

    # no existing -> adopt a valid incoming
    d4 = S.decide_replace(None, existing, cfg)
    assert d4["action"] == "adopt", d4


def test_choose_authoritative():
    cfg = _tiny_cfg()
    a = tempfile.mkdtemp()
    p3 = _make_ckpt(a, 3, cfg, name="checkpoint_step_000003.pt")
    p9 = _make_ckpt(a, 9, cfg, name="checkpoint_step_000009.pt")
    corrupt = os.path.join(a, "checkpoints", "checkpoint_bad.pt")
    with open(corrupt, "wb") as f:
        f.write(b"nope")
    best, report = S.choose_authoritative([p3, p9, corrupt], cfg)
    assert best == p9 and report["chosen_step"] == 9


def test_export_import_roundtrip():
    cfg = _tiny_cfg()
    src = tempfile.mkdtemp()
    # a run with tokenizer + status + checkpoint
    os.makedirs(os.path.join(src, "tokenizer"))
    with open(os.path.join(src, "tokenizer", "shnu_bpe.json"), "w") as f:
        json.dump({"fake": "tokenizer"}, f)
    _make_ckpt(src, 6, cfg)
    st = S.TrainingStatus.for_run(src)
    st.update(cfg, current_step=6, state="active")

    arch = os.path.join(tempfile.mkdtemp(), "bundle.tar.gz")
    man = S.export_run_bundle(src, arch, dataset="scratch")
    assert os.path.exists(arch) and man["files"]
    assert any(e["arcname"] == "checkpoints/checkpoint_latest.pt" for e in man["files"])

    # import into a fresh empty destination -> adopts step 6
    dst = tempfile.mkdtemp()
    rep = S.import_run_bundle(arch, dst, cfg=cfg, verbose=False)
    assert any("checkpoint_latest.pt" in a["arcname"] for a in rep["adopted"])
    got = S.inspect_checkpoint(os.path.join(dst, "checkpoints", "checkpoint_latest.pt"))
    assert got["valid"] and got["step"] == 6


def test_import_never_overwrites_newer():
    cfg = _tiny_cfg()
    # source bundle at step 3
    src = tempfile.mkdtemp()
    _make_ckpt(src, 3, cfg)
    arch = os.path.join(tempfile.mkdtemp(), "old.tar.gz")
    S.export_run_bundle(src, arch, dataset="scratch", include_tokenizer=False, include_status=False)

    # destination already at step 9
    dst = tempfile.mkdtemp()
    _make_ckpt(dst, 9, cfg)

    rep = S.import_run_bundle(arch, dst, cfg=cfg, verbose=False)
    # the step-3 checkpoint must be KEPT out (never overwrite newer step 9)
    assert any("checkpoint_latest.pt" in k["arcname"] for k in rep["kept"]), rep
    still = S.inspect_checkpoint(os.path.join(dst, "checkpoints", "checkpoint_latest.pt"))
    assert still["step"] == 9  # unchanged


def test_tampered_bundle_is_refused():
    cfg = _tiny_cfg()
    src = tempfile.mkdtemp()
    _make_ckpt(src, 5, cfg)
    arch = os.path.join(tempfile.mkdtemp(), "b.tar.gz")
    S.export_run_bundle(src, arch, dataset="scratch", include_tokenizer=False, include_status=False)

    # rebuild the archive with a modified checkpoint but the ORIGINAL manifest
    work = tempfile.mkdtemp()
    with tarfile.open(arch, "r:gz") as t:
        t.extractall(work)
    ckpt_rel = "checkpoints/checkpoint_latest.pt"
    with open(os.path.join(work, ckpt_rel), "ab") as f:
        f.write(b"\x00tamper")  # change bytes -> hash no longer matches manifest
    tampered = os.path.join(tempfile.mkdtemp(), "tampered.tar.gz")
    with tarfile.open(tampered, "w:gz") as t:
        t.add(os.path.join(work, "MANIFEST.json"), arcname="MANIFEST.json")
        t.add(os.path.join(work, ckpt_rel), arcname=ckpt_rel)

    dst = tempfile.mkdtemp()
    try:
        S.import_run_bundle(tampered, dst, cfg=cfg, verbose=False)
        assert False, "tampered bundle should have raised"
    except ValueError as e:
        assert "SHA-256" in str(e) or "size" in str(e)
    # nothing was placed
    assert not os.path.exists(os.path.join(dst, "checkpoints", "checkpoint_latest.pt"))


if __name__ == "__main__":
    for fn in list(globals()):
        if fn.startswith("test_"):
            globals()[fn]()
    print("PASS")
