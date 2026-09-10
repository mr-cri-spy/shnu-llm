#!/usr/bin/env python3
"""GPU pilot for SHNU-LLM v0.2 — validate before any long training run.

Loads the tokenizer and pre-tokenized streams produced by prepare_data.py, builds
the model from random initialization, benchmarks throughput and memory, runs a
short training pilot (loss must decrease, no NaN/Inf), saves a checkpoint,
reloads into a fresh model, resumes from the exact step, and generates text.

Nothing here starts the long run; it only proves the configuration is safe.

Usage:
  python scripts/pilot.py --config configs/v0.2.json --out-dir /content/drive/MyDrive/SHNU_LLM \
      --dataset wikitext103 --pilot-steps 40
"""
from __future__ import annotations

import os
import sys
import json
import math
import time
import argparse

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import shnu_llm as S


def main():
    ap = argparse.ArgumentParser(description="Run a SHNU-LLM v0.2 GPU pilot.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--dataset", default="wikitext103")
    ap.add_argument("--pilot-steps", type=int, default=40)
    ap.add_argument("--bench-iters", type=int, default=10)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = S.ShnuConfig(**json.load(open(args.config)))
    cfg.out_dir = args.out_dir
    report = {"device": S.env_report(device)}

    # tokenizer + pre-tokenized streams from durable storage
    tok = S.load_tokenizer(os.path.join(args.out_dir, "tokenizer", "shnu_bpe.json"))
    cfg.vocab_size = tok.get_vocab_size()
    eos_id = tok.token_to_id("<eos>")
    ds_dir = os.path.join(args.out_dir, "datasets")
    tr = np.load(os.path.join(ds_dir, f"{args.dataset}_train_v{cfg.vocab_size}.npy"))
    va = np.load(os.path.join(ds_dir, f"{args.dataset}_val_v{cfg.vocab_size}.npy"))
    datasets = {"train": S.PackedDataset(tr, cfg.block_size, device),
                "val": S.PackedDataset(va, cfg.block_size, device)}
    report["tokens"] = {"train": int(len(tr)), "val": int(len(va))}

    # model: random init, verified via initial loss ~ ln(vocab)
    S.set_seed(cfg.seed)
    model = S.ShnuLM(cfg).to(device)
    report["parameters"] = model.num_params()
    xb, yb = datasets["train"].get_batch(min(4, cfg.batch_size), torch.Generator().manual_seed(0))
    with torch.no_grad():
        _, l0 = model(xb, yb)
    report["initial_loss"] = round(l0.item(), 4)
    report["ln_vocab"] = round(math.log(cfg.vocab_size), 4)
    report["random_init_ok"] = abs(l0.item() - math.log(cfg.vocab_size)) < 1.0

    # throughput + memory benchmark
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    opt = S.configure_optimizer(model, cfg)
    ptdtype = S.resolve_dtype(cfg, device)
    use_amp = device == "cuda" and ptdtype in (torch.float16, torch.bfloat16)
    ctx = torch.autocast(device_type="cuda", dtype=ptdtype) if use_amp else _null()
    scaler = torch.amp.GradScaler(enabled=(use_amp and ptdtype == torch.float16))
    gen = torch.Generator().manual_seed(0)
    t0 = time.time()
    for _ in range(args.bench_iters):
        x, y = datasets["train"].get_batch(cfg.batch_size, gen)
        with ctx:
            _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
    if device == "cuda":
        torch.cuda.synchronize()
    dt = time.time() - t0
    tok_per_step = cfg.batch_size * cfg.block_size
    report["throughput_tok_per_s"] = round(tok_per_step * args.bench_iters / dt, 1)
    report["peak_gpu_mem_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2) if device == "cuda" else 0.0
    report["amp_dtype"] = str(ptdtype)

    # short pilot training (fresh model so loss trajectory is clean)
    S.set_seed(cfg.seed)
    model = S.ShnuLM(cfg).to(device)
    cfg.max_steps = args.pilot_steps
    cfg.warmup_steps = max(2, args.pilot_steps // 8)
    cfg.checkpoint_interval = args.pilot_steps
    cfg.eval_interval = max(2, args.pilot_steps // 2)
    cfg.log_interval = max(1, args.pilot_steps // 4)
    r = S.train(model, datasets, cfg, device, tokenizer=tok, eos_id=eos_id, verbose=True)
    report["pilot_final_train_loss"] = round(r["final_train_loss"], 4)
    report["pilot_final_val_loss"] = round(r["final_val_loss"], 4)
    report["loss_decreased"] = r["final_train_loss"] < report["initial_loss"]
    report["no_nan_inf"] = bool(np.isfinite(r["final_train_loss"]) and np.isfinite(r["final_val_loss"]))

    # save -> reload -> resume proof
    ckdir = os.path.join(cfg.out_dir, "checkpoints")
    latest = S.find_latest_checkpoint(ckdir)
    saved = S.checkpoint_step(latest)
    cfg.max_steps = saved + max(4, args.pilot_steps // 4)
    S.set_seed(cfg.seed)
    m2 = S.ShnuLM(cfg).to(device)
    r2 = S.train(m2, datasets, cfg, device, resume_from=latest, verbose=False)
    report["resume"] = {"saved_step": saved, "resumed_step": r2["start_step"], "new_step": r2["steps"],
                        "did_not_restart": r2["start_step"] == saved and r2["start_step"] > 0}
    report["checkpoint_path"] = S.find_latest_checkpoint(ckdir)

    # generation
    report["sample"] = S.sample_text(m2, tok, cfg, device, "The ", 60, temperature=0.8, top_k=50, eos_id=eos_id)

    passed = (report["random_init_ok"] and report["loss_decreased"] and report["no_nan_inf"]
              and report["resume"]["did_not_restart"])
    report["PILOT"] = "PASS" if passed else "FAIL"
    os.makedirs(os.path.join(cfg.out_dir, "logs"), exist_ok=True)
    json.dump(report, open(os.path.join(cfg.out_dir, "logs", "pilot_report.json"), "w"), indent=2, default=str)
    print(json.dumps(report, indent=2, default=str))


class _null:
    def __enter__(self): return None
    def __exit__(self, *a): return False


if __name__ == "__main__":
    main()
