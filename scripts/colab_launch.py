#!/usr/bin/env python3
"""One-call SHNU-LLM v0.2 launcher / resumer for a fresh Colab runtime.

This is the single entry point the Colab launcher cell calls in-process (so the
Google Drive mount stays alive). It is a thin wrapper over the shared,
platform-independent launch core (``shnu_llm.launcher_core``): it does NOT add a
second checkpoint or resume system — it ties together the pieces that already
exist:

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

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.join(_HERE, "..")
if os.path.join(_REPO, "src") not in sys.path:
    sys.path.insert(0, os.path.join(_REPO, "src"))

from shnu_llm.preflight import PreflightError
from shnu_llm.launcher_core import run_launch, DEFAULT_EXPECTED, verify_code  # re-exported for callers/tests
from shnu_llm.platform import COLAB


def launch(out_dir, *, dataset="wikitext103", config_path=None, expected=None,
           confirm_long_run=False, require_gpu_gate=True, require_t4=True,
           min_free_gb=3.0, drive_root="/content/drive/MyDrive", allow_fresh=True,
           repo_dir=None, verbose=True):
    """Verify → gate → reconcile → (optionally) resume-train on Colab. Returns a dict.

    With ``confirm_long_run=False`` (default) this is a safe dry run: it never
    trains. Set it True only to actually run to the configured target step count.
    ``require_gpu_gate=False`` skips only the GPU gate (for CPU smoke tests);
    real Colab launches keep it True so a run refuses to start without a T4.
    """
    return run_launch(out_dir, platform=COLAB, dataset=dataset, config_path=config_path,
                      expected=expected, confirm_long_run=confirm_long_run,
                      require_gpu_gate=require_gpu_gate, require_t4=require_t4,
                      min_free_gb=min_free_gb, drive_root=drive_root,
                      allow_fresh=allow_fresh, repo_dir=repo_dir or _REPO, verbose=verbose)


def main():
    import argparse
    ap = argparse.ArgumentParser(description="SHNU-LLM v0.2 one-call launcher / resumer (Colab).")
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
