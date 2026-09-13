"""SAFE benchmark + time-estimate for SHNU-LLM multi-platform planning.

This measures how fast a platform can train **without touching the real run**:

  * it builds a **fresh, random** model in memory (no checkpoint is read);
  * it trains on **synthetic random batches** (no dataset file is read);
  * it writes **nothing** to disk — no checkpoint, no status, no scratch file;
  * it runs only 20–100 optimizer steps.

It reports measured tokens/sec, peak VRAM, and whether any NaN/Inf appeared, then
projects the fixed remaining-token budget onto that measured throughput. The
projection is explicitly an **estimate** (pure compute; excludes eval, I/O, and
reconnect overhead). Because it never opens the real checkpoint, dataset, or
status, it is impossible for a benchmark to corrupt or advance the real run.
"""
from __future__ import annotations

import time
from typing import Optional

from .config import ShnuConfig
from .status import tokens_per_step


def benchmark_throughput(cfg: ShnuConfig, device: Optional[str] = None, *,
                         steps: int = 50, warmup: int = 5, verbose: bool = True) -> dict:
    """Measure optimizer-step throughput on synthetic data. Never touches disk.

    Uses the same architecture, optimizer, autocast dtype, grad-accum, and
    grad-clip as real training, but with random token batches and a fresh model,
    so the measured tokens/sec transfers to the real run while nothing real is
    read or written. ``steps`` should be 20–100; the first ``warmup`` steps are
    excluded from timing.
    """
    import torch
    import numpy as np
    from .model import ShnuLM
    from .training import configure_optimizer
    from .utils import resolve_dtype, set_seed

    if not (5 <= steps <= 500):
        raise ValueError("steps should be in [5, 500] for a SAFE short benchmark")
    warmup = max(0, min(warmup, steps - 1))
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    set_seed(cfg.seed)
    model = ShnuLM(cfg).to(device)
    model.train()
    optimizer = configure_optimizer(model, cfg)
    ptdtype = resolve_dtype(cfg, device)
    use_amp = device == "cuda" and ptdtype in (torch.float16, torch.bfloat16)
    if use_amp:
        ctx = torch.autocast(device_type="cuda", dtype=ptdtype)
    elif device == "cpu" and ptdtype == torch.bfloat16:
        ctx = torch.autocast(device_type="cpu", dtype=torch.bfloat16)
    else:
        from contextlib import nullcontext
        ctx = nullcontext()
    scaler = torch.amp.GradScaler(enabled=(use_amp and ptdtype == torch.float16))

    tps = tokens_per_step(cfg)
    B, T, V = cfg.batch_size, cfg.block_size, cfg.vocab_size
    gen = torch.Generator(device="cpu").manual_seed(cfg.seed)

    def _rand_batch():
        ids = torch.randint(0, V, (B, T + 1), generator=gen)
        x = ids[:, :-1].contiguous().to(device)
        y = ids[:, 1:].contiguous().to(device)
        return x, y

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    nan_inf = False
    losses = []
    t_start = None
    measured_steps = 0

    for step in range(steps):
        if step == warmup:
            if device == "cuda":
                torch.cuda.synchronize()
            t_start = time.time()
        lr = cfg.lr
        for pg in optimizer.param_groups:
            pg["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        loss_accum = 0.0
        for _ in range(cfg.grad_accum_steps):
            x, y = _rand_batch()
            with ctx:
                _, loss = model(x, y)
                loss = loss / cfg.grad_accum_steps
            scaler.scale(loss).backward()
            lv = loss.item()
            loss_accum += lv
            if not np.isfinite(lv):
                nan_inf = True
        if cfg.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        if step >= warmup:
            losses.append(loss_accum)
            measured_steps += 1

    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - t_start if t_start is not None else 0.0
    tok_per_s = (measured_steps * tps / elapsed) if elapsed > 0 else None
    sec_per_step = (elapsed / measured_steps) if measured_steps else None
    peak_vram_gb = round(torch.cuda.max_memory_allocated() / 1e9, 3) if device == "cuda" else None

    gpu_name = torch.cuda.get_device_name(0) if device == "cuda" else None
    result = {
        "device": device,
        "gpu_name": gpu_name,
        "steps_total": steps,
        "warmup": warmup,
        "steps_measured": measured_steps,
        "tokens_per_step": tps,
        "elapsed_seconds": round(elapsed, 4),
        "tokens_per_second": round(tok_per_s, 1) if tok_per_s else None,
        "seconds_per_step": round(sec_per_step, 4) if sec_per_step else None,
        "peak_vram_gb": peak_vram_gb,
        "nan_or_inf_detected": nan_inf,
        "first_loss": round(losses[0], 4) if losses else None,
        "last_loss": round(losses[-1], 4) if losses else None,
        "note": "synthetic random data; loss values are not meaningful — only "
                "throughput / VRAM / NaN-Inf are.",
    }
    del model, optimizer
    if device == "cuda":
        torch.cuda.empty_cache()
    if verbose:
        print(f"[benchmark] {gpu_name or device}: "
              f"{result['tokens_per_second']} tok/s, "
              f"peak {peak_vram_gb} GB, NaN/Inf={nan_inf} "
              f"({measured_steps} steps measured)")
    return result


def estimate_completion(tokens_per_second: Optional[float], *, remaining_steps: int,
                        tokens_per_step: int, milestones=(500, 1000, 5000)) -> dict:
    """Project time to train ``remaining_steps`` at a measured throughput.

    ESTIMATE ONLY — pure GPU compute, excluding eval passes, checkpoint I/O, Drive
    latency, and reconnect overhead (which added ~15–25% in real sessions). Add
    that overhead before planning sessions.
    """
    remaining_tokens = int(remaining_steps) * int(tokens_per_step)
    out = {
        "is_estimate": True,
        "basis": "pure GPU compute; excludes eval/IO/reconnect overhead (~15-25% more)",
        "tokens_per_second": tokens_per_second,
        "remaining_steps": remaining_steps,
        "tokens_per_step": tokens_per_step,
        "remaining_tokens": remaining_tokens,
        "milestones": [],
    }
    pts = sorted(set(list(milestones) + [remaining_steps]))
    for m in pts:
        toks = int(m) * int(tokens_per_step)
        secs = (toks / tokens_per_second) if tokens_per_second else None
        out["milestones"].append({
            "steps": m,
            "tokens": toks,
            "seconds": round(secs, 1) if secs is not None else None,
            "minutes": round(secs / 60, 1) if secs is not None else None,
            "hours": round(secs / 3600, 2) if secs is not None else None,
            "is_full_remaining": (m == remaining_steps),
        })
    return out
