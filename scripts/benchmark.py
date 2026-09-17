#!/usr/bin/env python3
"""SAFE benchmark + remaining-time estimate for SHNU-LLM (any platform).

Measures training throughput on the CURRENT machine using a fresh random model
and synthetic batches, then estimates how long the remaining steps will take.

SAFETY: this never reads or writes the real checkpoint, dataset, or status. It
builds a throwaway model in memory, trains on random tensors for a few dozen
steps, and prints numbers. Run it to decide whether a platform is worth using —
it cannot advance, corrupt, or reset the real run.

    python scripts/benchmark.py                       # v0.2 arch, 50 steps
    python scripts/benchmark.py --steps 30 --current-step 3000
    python scripts/benchmark.py --config configs/v0.2.json --json

``--current-step`` is used ONLY to compute remaining = max_steps - current_step
for the estimate; it does not read or change the real checkpoint.
"""
from __future__ import annotations

import os
import sys
import json
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import shnu_llm as S

_REPO = os.path.join(os.path.dirname(__file__), "..")


def main():
    ap = argparse.ArgumentParser(description="SAFE throughput benchmark + time estimate.")
    ap.add_argument("--config", default=os.path.join(_REPO, "configs", "v0.2.json"))
    ap.add_argument("--steps", type=int, default=50, help="total steps (20-100 recommended)")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--current-step", type=int, default=None,
                    help="known durable step, to compute remaining (default: read status read-only, else 0)")
    ap.add_argument("--out-dir", default=None,
                    help="optional durable root to READ current step from status (never written)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--json", action="store_true", help="print machine-readable JSON")
    args = ap.parse_args()

    cfg = S.ShnuConfig(**json.load(open(args.config)))

    # determine current step (read-only) for the remaining-work estimate
    current_step = args.current_step
    if current_step is None and args.out_dir:
        try:
            st = S.TrainingStatus.for_run(args.out_dir).load_or_none()
            current_step = int(st.get("current_step", 0)) if st else 0
        except Exception:
            current_step = 0
    if current_step is None:
        current_step = 0
    remaining = max(0, int(cfg.max_steps) - int(current_step))

    gpu = S.describe_gpu()
    bench = S.benchmark_throughput(cfg, device=args.device, steps=args.steps,
                                   warmup=args.warmup, verbose=not args.json)
    est = S.estimate_completion(bench["tokens_per_second"], remaining_steps=remaining,
                                tokens_per_step=bench["tokens_per_step"])

    payload = {"gpu": gpu, "benchmark": bench, "current_step": current_step,
               "remaining_steps": remaining, "estimate": est}

    if args.json:
        print(json.dumps(payload, indent=2))
        return

    print("\n===== SHNU-LLM SAFE BENCHMARK =====")
    print(f"GPU available:   {gpu['available']}  ({gpu.get('name')}, "
          f"{gpu.get('vram_gb')} GB, is_T4={gpu.get('is_t4')})")
    print(f"Device used:     {bench['device']}")
    print(f"Throughput:      {bench['tokens_per_second']} tok/s "
          f"({bench['steps_measured']} steps measured)")
    print(f"Seconds/step:    {bench['seconds_per_step']}")
    print(f"Peak VRAM:       {bench['peak_vram_gb']} GB")
    print(f"NaN/Inf:         {bench['nan_or_inf_detected']}")
    print(f"\nRemaining work:  {remaining} steps "
          f"({est['remaining_tokens']:,} tokens)  [from step {current_step} -> {cfg.max_steps}]")
    print("Time estimate (ESTIMATE — pure compute, excludes eval/IO/reconnect ~15-25%):")
    if bench["tokens_per_second"]:
        for m in est["milestones"]:
            tag = "  <-- all remaining" if m["is_full_remaining"] else ""
            print(f"   {m['steps']:>6} steps: {m['hours']:>6} h "
                  f"({m['minutes']} min){tag}")
    else:
        print("   (no throughput measured — GPU unavailable or run too short)")
    if not gpu["available"]:
        print("\nNOTE: no CUDA GPU here — these are CPU numbers, NOT representative of a T4.")


if __name__ == "__main__":
    main()
