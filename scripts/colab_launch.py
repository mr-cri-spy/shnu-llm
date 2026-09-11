#!/usr/bin/env python3
"""One-call SHNU-LLM v0.2 launcher / resumer for a fresh Colab runtime.

This is the single entry point the Colab launcher cell calls in-process (so the
Google Drive mount stays alive). It ties together the pieces that already exist —
it does NOT add a second checkpoint or resume system:

  1. verify the checked-out code / config integrity (params, architecture);
  2. run the safety gates (GPU is a T4, Drive mounted, free disk, dataset hash,
     tokenizer hash, checkpoint architecture) — fail-closed, never CPU fallback;
  3. reconcile ``training_status.json`` from the newest valid checkpoint
     (the checkpoint is authoritative — a valid run is never reset to step 0);
  4. resume from that exact checkpoint step and train to the configured target,
     writing atomic checkpoints and status updates, and marking interrupted /
     failed / completed safely for the next session.

Nothing trains unless ``confirm_long_run=True`` is passed. With the default
(``False``) it performs the full dry run — pull is done by the cell, then verify
→ gates → reconcile → print the plan — and STOPS before training. That dry run is
also how the launcher is validated against the real Drive.

GitHub holds code/config/docs; Google Drive holds datasets/checkpoints/status/
logs. No runtime artifact is ever written to the repo.
"""
from __future__ import annotations

import os
import sys
import json

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.join(_HERE, "..")
if os.path.join(_REPO, "src") not in sys.path:
    sys.path.insert(0, os.path.join(_REPO, "src"))

import shnu_llm as S
from shnu_llm.preflight import (
    PreflightError, require_gpu, require_drive_mounted, require_free_disk,
    verify_dataset, verify_tokenizer, verify_checkpoint,
)

# Verified v0.2 baseline integrity values (mirror scripts/train_v0_2.py).
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


def _verify_code(cfg, expected, verbose=True):
    """Confirm the config matches the expected architecture and the model builds
    to the expected parameter count. Raises PreflightError on mismatch."""
    arch = expected.get("arch") or {}
    bad = {k: (getattr(cfg, k), v) for k, v in arch.items() if getattr(cfg, k) != v}
    if bad:
        raise PreflightError(f"config does not match expected v0.2 architecture: {bad}")
    import torch
    S.set_seed(cfg.seed)
    model = S.ShnuLM(cfg)
    params = model.num_params()
    exp_params = expected.get("params")
    if exp_params is not None and params != exp_params:
        raise PreflightError(f"model params {params} != expected {exp_params}")
    del model
    if verbose:
        print(f"[launch] code/config verified: {params:,} params, arch OK")
    return params


def launch(out_dir, *, dataset="wikitext103", config_path=None, expected=None,
           confirm_long_run=False, require_gpu_gate=True, require_t4=True,
           min_free_gb=3.0, drive_root="/content/drive/MyDrive", allow_fresh=True,
           repo_dir=None, verbose=True):
    """Verify → gate → reconcile → (optionally) resume-train. Returns a dict.

    With ``confirm_long_run=False`` (default) this is a safe dry run: it never
    trains. Set it True only to actually run to the configured target step count.
    ``require_gpu_gate=False`` skips only the GPU gate (for CPU smoke tests);
    real Colab launches keep it True so a run refuses to start without a T4.
    """
    repo_dir = repo_dir or _REPO
    config_path = config_path or os.path.join(repo_dir, "configs", "v0.2.json")
    expected = expected if expected is not None else DEFAULT_EXPECTED

    head = S.current_git_commit(repo_dir)
    cfg = S.ShnuConfig(**json.load(open(config_path)))
    cfg.out_dir = out_dir
    if verbose:
        print(f"== SHNU-LLM v0.2 launcher ==\n[launch] repo HEAD: {head}")
        print(f"[launch] out_dir (Drive): {out_dir}")

    # 1. code / config integrity
    _verify_code(cfg, expected, verbose=verbose)

    ds_meta = os.path.join(out_dir, "datasets", f"{dataset}_metadata.json")
    tok_path = os.path.join(out_dir, "tokenizer", "shnu_bpe.json")
    ckpt = S.find_latest_checkpoint(os.path.join(out_dir, "checkpoints"))

    # 2. safety gates (fail-closed; never CPU fallback)
    report = {"head": head, "config": config_path}
    if require_gpu_gate:
        report["gpu"] = require_gpu(require_t4=require_t4)
    require_drive_mounted(drive_root)
    report["free_disk_gb"] = require_free_disk(out_dir if os.path.isdir(out_dir) else "/content",
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
    device = "cuda" if require_gpu_gate else ("cuda" if _cuda() else "cpu")
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

    # 5. resume + train (resume_training handles resume/RNG/atomic ckpt/status/interrupt/fail)
    from resume import resume_training  # scripts/ is on sys.path in the launcher cell
    if verbose:
        print("\n[launch] == starting long training (confirmed) ==")
    result = resume_training(cfg, out_dir, dataset, allow_fresh=allow_fresh,
                             verbose=verbose, status_enabled=True)
    report["trained"] = True
    report["dry_run"] = False
    report["result"] = {k: result.get(k) for k in ("start_step", "steps", "final_val_loss")}
    return report


def _cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def main():
    import argparse
    ap = argparse.ArgumentParser(description="SHNU-LLM v0.2 one-call launcher / resumer.")
    ap.add_argument("--out-dir", required=True, help="durable run root (a Google Drive path)")
    ap.add_argument("--dataset", default="wikitext103")
    ap.add_argument("--confirm-long-run", action="store_true",
                    help="REQUIRED to actually start the long training run")
    ap.add_argument("--allow-non-t4", dest="require_t4", action="store_false", default=True)
    ap.add_argument("--min-free-gb", type=float, default=3.0)
    args = ap.parse_args()
    try:
        launch(args.out_dir, dataset=args.dataset, confirm_long_run=args.confirm_long_run,
               require_t4=args.require_t4, min_free_gb=args.min_free_gb)
    except PreflightError as e:
        print(f"PREFLIGHT FAILED: {e}\nLauncher will NOT train. Fix the issue above and retry.")
        sys.exit(2)


if __name__ == "__main__":
    main()
