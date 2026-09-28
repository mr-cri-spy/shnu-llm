#!/usr/bin/env python3
"""Guarded EXP-001 (v0.3) training launcher.

This is the only supported way to train EXP-001. It builds the ``ShnuConfig`` from
the ``shnu_config`` block of configs/v0.3-exp001.json (the full file is not a flat
ShnuConfig and cannot be given to scripts/resume.py directly). It then delegates to
the existing ``resume_training`` with run name ``shnu_v0.3_exp001``, so the v0.2
pipeline itself is unchanged.

Training starts only if every gate passes:
  1. config preflight (scripts/validate_exp001.py): frozen values, branch,
     tokenizer file hash, run directory not a v0.2.0 directory;
  2. run-directory state: a fresh start needs an empty checkpoints directory;
     --resume needs EXP-001's own checkpoints (never a v0.2.0 checkpoint);
  3. dataset manifest (shnu_llm.exp001_data.validate_data_manifest): every file
     present and hash-matched, the frozen WikiText validation stream, and every
     reviewed hash pinned in the config;
  4. an explicit --confirm-reviewed flag.

Usage:
  python scripts/train_exp001.py --out-dir /path/to/v0.3/run --preflight-only
  python scripts/train_exp001.py --out-dir /path/to/v0.3/run --confirm-reviewed          # first session
  python scripts/train_exp001.py --out-dir /path/to/v0.3/run --confirm-reviewed --resume # later sessions
"""
from __future__ import annotations

import os
import sys
import argparse
import importlib.util

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO, "src"))
import shnu_llm as S  # noqa: E402
from shnu_llm import exp001_data as X  # noqa: E402


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, os.path.join(REPO, "scripts", f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


V = _load_script("validate_exp001")


class LaunchRefused(Exception):
    """Raised when any EXP-001 launch gate fails."""


def build_shnu_config(cfg: dict, out_dir: str) -> "S.ShnuConfig":
    """ShnuConfig from the explicit shnu_config block. Only out_dir is set, to the run directory."""
    sc = S.ShnuConfig(**cfg["shnu_config"])
    sc.out_dir = out_dir
    return sc


def check_resume_state(cfg: dict, out_dir: str) -> str:
    """For --resume: the run directory must hold EXP-001's own checkpoint_latest.pt and no v0.2.0 checkpoint."""
    import torch
    errors = []
    norm = lambda p: os.path.normpath(str(p)).rstrip("/")
    if norm(out_dir) in {norm(p) for p in V.FROZEN["v02_out_dirs"]}:
        raise LaunchRefused(f"--out-dir {out_dir!r} is a v0.2.0 run directory")
    ck_dir = os.path.join(out_dir, "checkpoints")
    latest = os.path.join(ck_dir, "checkpoint_latest.pt")
    if not os.path.isfile(latest):
        raise LaunchRefused(f"--resume given but {latest} does not exist")
    frozen = {V.FROZEN["baseline_checkpoint_sha256"], V.FROZEN["baseline_best_checkpoint_sha256"]}
    for fn in sorted(os.listdir(ck_dir)):
        if fn.endswith(".pt") and X.sha256_file(os.path.join(ck_dir, fn)) in frozen:
            errors.append(f"{fn} is a v0.2.0 checkpoint; EXP-001 must never resume from v0.2.0")
    ck = torch.load(latest, map_location="cpu", weights_only=False)
    saved = ck.get("config", {}) or {}
    if saved.get("version") != cfg["shnu_config"]["version"]:
        errors.append(f"checkpoint_latest.pt config version {saved.get('version')!r} != "
                      f"{cfg['shnu_config']['version']!r}; it is not an EXP-001 checkpoint")
    for k in ("n_layers", "n_heads", "n_kv_heads", "d_model", "d_ff", "block_size", "vocab_size",
              "tie_embeddings", "batch_size", "grad_accum_steps", "max_steps", "lr", "seed"):
        if saved.get(k) != cfg["shnu_config"][k]:
            errors.append(f"checkpoint_latest.pt {k}={saved.get(k)!r} != config {cfg['shnu_config'][k]!r}")
    if errors:
        raise LaunchRefused("\n".join(errors))
    return latest


def preflight(config_path: str, out_dir: str, *, resume: bool = False, check_branch: bool = True,
              data_frozen: dict | None = None) -> dict:
    """Run every gate except --confirm-reviewed. Raises LaunchRefused listing the problems."""
    tok_path = os.path.join(out_dir, "tokenizer", "shnu_bpe.json")
    try:
        V.validate(config_path, out_dir=None if resume else out_dir, tokenizer_path=tok_path,
                   check_branch=check_branch)
    except V.ValidationError as e:
        raise LaunchRefused(f"config preflight failed:\n{e}") from e
    cfg = V.load_config(config_path)
    if resume:
        check_resume_state(cfg, out_dir)
    try:
        man = X.validate_data_manifest(os.path.join(out_dir, "datasets"), cfg, frozen=data_frozen)
    except X.DataManifestError as e:
        raise LaunchRefused(f"dataset manifest check failed:\n{e}") from e
    return {"cfg": cfg, "manifest": man, "shnu_config": build_shnu_config(cfg, out_dir)}


def main(argv=None, *, _train_fn=None) -> int:
    ap = argparse.ArgumentParser(description="Guarded EXP-001 training launcher.")
    ap.add_argument("--config", default=V.DEFAULT_CONFIG)
    ap.add_argument("--out-dir", required=True, help="the EXP-001 run directory (never a v0.2.0 directory)")
    ap.add_argument("--resume", action="store_true", help="continue EXP-001 from its own checkpoints")
    ap.add_argument("--preflight-only", action="store_true", help="run every gate, then exit without training")
    ap.add_argument("--confirm-reviewed", action="store_true",
                    help="required to train: EXP-001 registration and data manifest have been reviewed")
    ap.add_argument("--no-branch-check", action="store_true")
    args = ap.parse_args(argv)

    try:
        ctx = preflight(args.config, args.out_dir, resume=args.resume, check_branch=not args.no_branch_check)
    except LaunchRefused as e:
        print("EXP-001 LAUNCH REFUSED", file=sys.stderr)
        for line in str(e).splitlines():
            print(f"  - {line}", file=sys.stderr)
        return 1
    print("EXP-001 preflight: all gates passed")
    if args.preflight_only:
        print("--preflight-only: not starting training.")
        return 0
    if not args.confirm_reviewed:
        print("Refusing to train without --confirm-reviewed.", file=sys.stderr)
        return 2

    cfg = ctx["cfg"]
    train_fn = _train_fn or _load_script("resume").resume_training
    train_fn(ctx["shnu_config"], args.out_dir, cfg["dataset"]["dataset_key"],
             allow_fresh=not args.resume, run_name=cfg["output"]["run_name"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
