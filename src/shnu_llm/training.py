"""Training loop, learning-rate schedule, checkpointing and validation."""
from __future__ import annotations

import os
import json
import time
import math
from dataclasses import asdict
from typing import Optional

import numpy as np
import torch

from .config import ShnuConfig
from .utils import resolve_dtype
from .generation import sample_text


def get_lr(step: int, cfg: ShnuConfig) -> float:
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / max(1, cfg.warmup_steps)
    if step >= cfg.max_steps:
        return cfg.lr * cfg.min_lr_ratio
    ratio = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return cfg.lr * cfg.min_lr_ratio + coeff * (cfg.lr - cfg.lr * cfg.min_lr_ratio)


def configure_optimizer(model, cfg: ShnuConfig):
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.dim() >= 2:
            decay.append(p)
        else:
            no_decay.append(p)
    groups = [
        {"params": decay, "weight_decay": cfg.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=cfg.lr, betas=(cfg.beta1, cfg.beta2))


@torch.no_grad()
def estimate_loss(model, datasets: dict, cfg: ShnuConfig, gen: torch.Generator, ctx):
    model.eval()
    out = {}
    for split, ds in datasets.items():
        losses = torch.zeros(cfg.eval_iters)
        for k in range(cfg.eval_iters):
            x, y = ds.get_batch(cfg.batch_size, gen)
            with ctx:
                _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def save_checkpoint(path, model, optimizer, cfg, step, best_val, log, rng_state=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ckpt = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": asdict(cfg),
        "step": step,
        "best_val": best_val,
        "log": log,
        "torch_rng_state": torch.get_rng_state(),
        "numpy_rng_state": np.random.get_state(),
    }
    if rng_state is not None:
        ckpt["cuda_rng_state"] = rng_state
    elif torch.cuda.is_available():
        ckpt["cuda_rng_state"] = torch.cuda.get_rng_state_all()
    # write atomically so an interrupted save can never corrupt the checkpoint
    tmp = path + ".tmp"
    torch.save(ckpt, tmp)
    os.replace(tmp, path)


def load_checkpoint(path, model, optimizer=None, map_location="cpu"):
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt


def prune_checkpoints(ckpt_dir: str, keep_last: int = 2):
    """Keep only the newest `keep_last` numbered step checkpoints (checkpoint_latest.pt
    and checkpoint_best.pt are never pruned), bounding durable storage."""
    import glob
    if keep_last is None or keep_last < 0:
        return
    steps = sorted(glob.glob(os.path.join(ckpt_dir, "checkpoint_step_*.pt")),
                   key=lambda p: int("".join(filter(str.isdigit, os.path.basename(p))) or 0))
    for old in steps[:-keep_last] if keep_last else steps:
        try:
            os.remove(old)
        except OSError:
            pass


def train(model, datasets, cfg: ShnuConfig, device: str,
          resume_from: Optional[str] = None, verbose: bool = True,
          tokenizer=None, eos_id=None, sample_prompt="\n", keep_last_checkpoints: int = 2):
    os.makedirs(os.path.join(cfg.out_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(cfg.out_dir, "samples"), exist_ok=True)
    os.makedirs(os.path.join(cfg.out_dir, "logs"), exist_ok=True)

    model.to(device)
    optimizer = configure_optimizer(model, cfg)
    ptdtype = resolve_dtype(cfg, device)
    use_amp = device == "cuda" and ptdtype in (torch.float16, torch.bfloat16)
    ctx = (torch.autocast(device_type="cuda", dtype=ptdtype) if use_amp
           else torch.autocast(device_type="cpu", dtype=torch.bfloat16)
           if device == "cpu" and ptdtype == torch.bfloat16
           else _nullcontext())
    scaler = torch.amp.GradScaler(enabled=(use_amp and ptdtype == torch.float16))

    start_step, best_val = 0, float("inf")
    log = {"step": [], "train_loss": [], "val_loss": [], "lr": []}
    if resume_from and os.path.exists(resume_from):
        ckpt = load_checkpoint(resume_from, model, optimizer, map_location=device)
        start_step = ckpt.get("step", 0)
        best_val = ckpt.get("best_val", float("inf"))
        log = ckpt.get("log", log)
        # restore RNG state where available so a resumed run continues the same stream
        try:
            if ckpt.get("torch_rng_state") is not None:
                torch.set_rng_state(ckpt["torch_rng_state"].to("cpu") if hasattr(ckpt["torch_rng_state"], "to")
                                    else ckpt["torch_rng_state"])
            if ckpt.get("numpy_rng_state") is not None:
                np.random.set_state(ckpt["numpy_rng_state"])
            if device == "cuda" and ckpt.get("cuda_rng_state") is not None:
                torch.cuda.set_rng_state_all(ckpt["cuda_rng_state"])
        except Exception as e:
            if verbose:
                print(f"[resume] RNG restore skipped ({e!r})")
        if verbose:
            print(f"[resume] from step {start_step} (best_val={best_val:.4f}); RNG restored")

    gen = torch.Generator().manual_seed(cfg.seed + start_step)
    model.train()
    t0 = time.time()
    tokens_per_step = cfg.batch_size * cfg.grad_accum_steps * cfg.block_size
    running = None

    for step in range(start_step, cfg.max_steps):
        lr = get_lr(step, cfg)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        loss_accum = 0.0
        for _ in range(cfg.grad_accum_steps):
            x, y = datasets["train"].get_batch(cfg.batch_size, gen)
            with ctx:
                _, loss = model(x, y)
                loss = loss / cfg.grad_accum_steps
            scaler.scale(loss).backward()
            loss_accum += loss.item()
        if cfg.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        running = loss_accum if running is None else 0.9 * running + 0.1 * loss_accum
        if verbose and step % cfg.log_interval == 0:
            dt = time.time() - t0
            tps = tokens_per_step * (step - start_step + 1) / max(dt, 1e-9)
            mem = (torch.cuda.max_memory_allocated() / 1e9) if device == "cuda" else 0.0
            print(f"step {step:5d} | loss {loss_accum:.4f} | ema {running:.4f} "
                  f"| lr {lr:.2e} | {tps:8.0f} tok/s | {mem:.2f}GB")

        # periodic validation
        if step > 0 and step % cfg.eval_interval == 0:
            metrics = estimate_loss(model, datasets, cfg, gen, ctx)
            log["step"].append(step)
            log["train_loss"].append(metrics["train"])
            log["val_loss"].append(metrics["val"])
            log["lr"].append(lr)
            if verbose:
                print(f"  >> eval step {step}: train {metrics['train']:.4f} "
                      f"val {metrics['val']:.4f} ppl {math.exp(metrics['val']):.2f}")
            if metrics["val"] < best_val:
                best_val = metrics["val"]
                save_checkpoint(os.path.join(cfg.out_dir, "checkpoints", "checkpoint_best.pt"),
                                model, optimizer, cfg, step, best_val, log)

        # periodic sampling
        if tokenizer is not None and step > 0 and step % cfg.sample_interval == 0:
            sample = sample_text(model, tokenizer, cfg, device, sample_prompt,
                                 max_new_tokens=120, temperature=0.8, top_k=50, eos_id=eos_id)
            with open(os.path.join(cfg.out_dir, "samples", f"step_{step}.txt"), "w") as f:
                f.write(sample)
            if verbose:
                print(f"  >> sample @ {step}: {sample[:120]!r}")

        # periodic checkpoint (bounded storage: keep only the newest `keep_last_checkpoints`)
        if step > 0 and step % cfg.checkpoint_interval == 0:
            save_checkpoint(os.path.join(cfg.out_dir, "checkpoints", f"checkpoint_step_{step:06d}.pt"),
                            model, optimizer, cfg, step, best_val, log)
            save_checkpoint(os.path.join(cfg.out_dir, "checkpoints", "checkpoint_latest.pt"),
                            model, optimizer, cfg, step, best_val, log)
            prune_checkpoints(os.path.join(cfg.out_dir, "checkpoints"), keep_last_checkpoints)

    # final eval + checkpoint
    metrics = estimate_loss(model, datasets, cfg, gen, ctx)
    save_checkpoint(os.path.join(cfg.out_dir, "checkpoints", "checkpoint_latest.pt"),
                    model, optimizer, cfg, cfg.max_steps, best_val, log)
    with open(os.path.join(cfg.out_dir, "logs", "train_log.json"), "w") as f:
        json.dump(log, f, indent=2)
    total_time = time.time() - t0
    return {
        "final_train_loss": metrics["train"],
        "final_val_loss": metrics["val"],
        "final_val_ppl": math.exp(metrics["val"]),
        "best_val_loss": best_val,
        "steps": cfg.max_steps,
        "start_step": start_step,
        "train_time_s": total_time,
        "tokens_seen": tokens_per_step * (cfg.max_steps - start_step),
        "log": log,
    }


class _nullcontext:
    def __enter__(self): return None
    def __exit__(self, *a): return False
