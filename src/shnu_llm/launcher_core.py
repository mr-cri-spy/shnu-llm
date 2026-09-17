"""Shared, platform-independent launch/resume core for SHNU-LLM v0.2.

This is the single implementation of the "verify → gate → reconcile →
(optionally) resume-train" flow. Both the Colab launcher (`scripts/colab_launch.py`)
and the Kaggle launcher (`scripts/kaggle_launch.py`) delegate here, so there is
exactly **one** code path, **one** checkpoint format, and **one** resume system
(`scripts/resume.py`) across every platform. The only per-platform differences —
whether a T4 is required, where the durable store lives, how much free disk to
demand — are carried by a :class:`~shnu_llm.platform.Platform` value.

Nothing trains unless ``confirm_long_run=True``. With the default (``False``) it
performs the full dry run (verify → gates → reconcile → print the plan) and
STOPS before training. The checkpoint is always authoritative; a valid run is
never reset to step 0.
"""
from __future__ import annotations

import os
import sys
import json

from .config import ShnuConfig
from .preflight import (
    PreflightError, require_gpu, require_drive_mounted, require_free_disk,
    verify_dataset, verify_tokenizer, verify_checkpoint,
)
from . import platform as _platform_mod

# Verified v0.2 baseline integrity values (single source of truth for launchers).
DEFAULT_EXPECTED = {
    "dataset_sha256_prefix": "1f9f77ec26e7477b7bf3c80c72a2e365749731f",
    "tokenizer_sha256_prefix": "1bf9397a6213d1a303c9d149d028fa2c456dd1a9",
    "vocab_size": 16000,
    "train_tokens": 116995555,
    "val_tokens": 1181773,
    "params": 33890816,
    "arch": {"vocab_size": 16000, "n_layers": 8, "n_heads": 8,
             "d_model": 512, "block_size": 512, "max_steps": 20000},
}


def verify_code(cfg: ShnuConfig, expected: dict, verbose: bool = True) -> int:
    """Confirm the config matches the expected architecture and the model builds
    to the expected parameter count. Raises PreflightError on any mismatch."""
    import shnu_llm as S
    arch = expected.get("arch") or {}
    bad = {k: (getattr(cfg, k), v) for k, v in arch.items() if getattr(cfg, k) != v}
    if bad:
        raise PreflightError(f"config does not match expected v0.2 architecture: {bad}")
    S.set_seed(cfg.seed)
    model = S.ShnuLM(cfg)
    params = model.num_params()
    exp_params = expected.get("params")
    del model
    if exp_params is not None and params != exp_params:
        raise PreflightError(f"model params {params} != expected {exp_params}")
    if verbose:
        print(f"[launch] code/config verified: {params:,} params, arch OK")
    return params


def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def run_launch(out_dir, *, platform, dataset="wikitext103", config_path=None, expected=None,
               confirm_long_run=False, require_gpu_gate=True, require_t4=None,
               min_free_gb=None, drive_root=None, allow_fresh=True, repo_dir=None,
               verbose=True):
    """Verify → gate → reconcile → (optionally) resume-train on any platform.

    ``platform`` is a :class:`~shnu_llm.platform.Platform`; its ``require_t4``,
    ``min_free_gb`` and ``default_drive_root`` supply defaults for the same-named
    arguments when those are left ``None``. Returns a report dict with the same
    keys on every platform.
    """
    import shnu_llm as S

    if isinstance(platform, str):
        platform = _platform_mod.get_platform(platform)
    require_t4 = platform.require_t4 if require_t4 is None else require_t4
    min_free_gb = platform.min_free_gb if min_free_gb is None else min_free_gb
    drive_root = platform.default_drive_root if drive_root is None else drive_root

    repo_dir = repo_dir or os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    config_path = config_path or os.path.join(repo_dir, "configs", "v0.2.json")
    expected = expected if expected is not None else DEFAULT_EXPECTED

    head = S.current_git_commit(repo_dir)
    cfg = ShnuConfig(**json.load(open(config_path)))
    cfg.out_dir = out_dir
    if verbose:
        print(f"== SHNU-LLM v0.2 launcher [{platform.name}] ==\n[launch] repo HEAD: {head}")
        print(f"[launch] out_dir (durable): {out_dir}")

    # 1. code / config integrity
    verify_code(cfg, expected, verbose=verbose)

    ds_meta = os.path.join(out_dir, "datasets", f"{dataset}_metadata.json")
    tok_path = os.path.join(out_dir, "tokenizer", "shnu_bpe.json")
    ckpt = S.find_latest_checkpoint(os.path.join(out_dir, "checkpoints"))

    # 2. safety gates (fail-closed; never CPU fallback)
    report = {"head": head, "config": config_path, "platform": platform.name}
    if require_gpu_gate:
        report["gpu"] = require_gpu(require_t4=require_t4)
    require_drive_mounted(drive_root)
    report["free_disk_gb"] = require_free_disk(out_dir if os.path.isdir(out_dir) else drive_root,
                                               min_gb=min_free_gb)
    report["dataset"] = verify_dataset(ds_meta, expected.get("dataset_sha256_prefix"),
                                       expected.get("train_tokens"), expected.get("val_tokens"))
    report["tokenizer"] = verify_tokenizer(tok_path, expected.get("tokenizer_sha256_prefix"),
                                           expected.get("vocab_size"))
    if ckpt:
        report["checkpoint"] = verify_checkpoint(ckpt, cfg)
    if verbose:
        print("[launch] safety gates PASSED")

    # 3. reconcile status from the newest checkpoint (checkpoint authoritative)
    status = S.TrainingStatus.for_run(out_dir)
    gpu_name = report.get("gpu", {}).get("gpu") if isinstance(report.get("gpu"), dict) else None
    gpu_mem = report.get("gpu", {}).get("vram_gb") if isinstance(report.get("gpu"), dict) else None
    device = "cuda" if (require_gpu_gate or _cuda_available()) else "cpu"
    rec = status.reconcile(cfg, git_commit=head, device=device,
                           gpu_name=gpu_name, gpu_memory_gb=gpu_mem)
    start_step = int(rec.get("current_step", 0))
    target = int(cfg.max_steps)
    report["reconciled_step"] = start_step
    report["target_steps"] = target
    report["reconciliation"] = rec.get("reconciliation")
    if verbose:
        print(f"[launch] status reconciled: resume at step {start_step} / target {target} "
              f"(reconciliation occurred={rec.get('reconciliation', {}).get('occurred')})")
        print(f"[launch] latest checkpoint: {ckpt or '(none — fresh run)'}")

    # 4. gate on explicit confirmation
    if not confirm_long_run:
        if verbose:
            print("\n[launch] DRY RUN — confirm_long_run is False, so training will NOT start.")
            print(f"[launch] Ready to resume from step {start_step} to {target} when confirmed.")
        report["trained"] = False
        report["dry_run"] = True
        return report

    # 5. resume + train (resume_training owns resume/RNG/atomic ckpt/status/interrupt/fail)
    scripts_dir = os.path.join(repo_dir, "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    from resume import resume_training
    if verbose:
        print("\n[launch] == starting long training (confirmed) ==")
    result = resume_training(cfg, out_dir, dataset, allow_fresh=allow_fresh,
                             verbose=verbose, status_enabled=True)
    report["trained"] = True
    report["dry_run"] = False
    report["result"] = {k: result.get(k) for k in ("start_step", "steps", "final_val_loss")}
    return report
