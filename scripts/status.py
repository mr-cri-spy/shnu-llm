#!/usr/bin/env python3
"""Human-readable SHNU-LLM v0.2 training-status inspector (CPU-safe).

Reads the durable training-status JSON from Google Drive (or any run root) and
prints a readable summary. It never touches the GPU and never starts training, so
it is safe to run from any machine that can see the Drive folder — including after
a Colab runtime has disconnected, as long as the Drive state still exists.

By default this is read-only. Pass --reconcile to repair the status file from the
newest checkpoint (the checkpoint is authoritative); this still does not train.

Usage:
  python scripts/status.py --out-dir /content/drive/MyDrive/SHNU_LLM
  python scripts/status.py --out-dir /content/drive/MyDrive/SHNU_LLM --reconcile
  python scripts/status.py --out-dir /content/drive/MyDrive/SHNU_LLM --json
"""
from __future__ import annotations

import os
import sys
import json
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import shnu_llm as S

REPO = os.path.join(os.path.dirname(__file__), "..")


def main():
    ap = argparse.ArgumentParser(description="Inspect the SHNU-LLM training status.")
    ap.add_argument("--out-dir", required=True, help="durable run root (e.g. a Google Drive path)")
    ap.add_argument("--run-name", default=S.DEFAULT_RUN_NAME)
    ap.add_argument("--config", default=os.path.join(REPO, "configs", "v0.2.json"))
    ap.add_argument("--reconcile", action="store_true",
                    help="repair the status file from the newest checkpoint (checkpoint authoritative)")
    ap.add_argument("--json", action="store_true", help="print the raw status JSON")
    args = ap.parse_args()

    status = S.TrainingStatus.for_run(args.out_dir, args.run_name)

    if args.reconcile:
        cfg = S.ShnuConfig(**json.load(open(args.config)))
        data = status.reconcile(cfg, git_commit=S.current_git_commit(REPO))
    else:
        data = status.load_or_none()

    if args.json:
        print(json.dumps(data, indent=2, default=str) if data is not None
              else json.dumps({"status": "missing", "path": status.status_path}, indent=2))
        return

    print(status.render(data))
    print(f"\n(status file: {status.status_path})")


if __name__ == "__main__":
    main()
