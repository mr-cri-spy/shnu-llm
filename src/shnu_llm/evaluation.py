"""Deterministic evaluation metrics for comparing SHNU-LLM versions.

Everything here is reproducible: fixed seeds, greedy decoding by default, and a
fixed evaluation set supplied by the caller. Metrics never depend on wall-clock or
sampling unless a seed is given explicitly.
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Optional

import torch

from .training import estimate_loss
from .generation import sample_text


@torch.no_grad()
def validation_perplexity(model, datasets, cfg, device, eval_iters: Optional[int] = None) -> dict:
    """Mean validation loss and perplexity over a fixed number of held-out batches."""
    from contextlib import nullcontext
    if eval_iters is not None:
        cfg.eval_iters = eval_iters
    gen = torch.Generator().manual_seed(cfg.seed)          # fixed -> deterministic batch draw
    metrics = estimate_loss(model, datasets, cfg, gen, nullcontext())
    return {"val_loss": round(metrics["val"], 6),
            "perplexity": round(math.exp(metrics["val"]), 4),
            "train_loss": round(metrics["train"], 6),
            "eval_iters": cfg.eval_iters}


def repetition_metrics(token_ids: list[int], n: int = 3) -> dict:
    """Degeneration signals on a generated token sequence.

    - distinct_1 / distinct_2: unique unigram/bigram ratio (higher = more diverse)
    - repeated_ngram_frac: fraction of n-grams that are repeats of an earlier n-gram
    - max_consecutive_repeat: longest run of the same token
    """
    ids = list(token_ids)
    if len(ids) < 2:
        return {"distinct_1": 1.0, "distinct_2": 1.0, "repeated_ngram_frac": 0.0,
                "max_consecutive_repeat": 1, "length": len(ids)}
    distinct_1 = len(set(ids)) / len(ids)
    bigrams = list(zip(ids, ids[1:]))
    distinct_2 = len(set(bigrams)) / max(1, len(bigrams))
    ngrams = list(zip(*[ids[i:] for i in range(n)])) if len(ids) >= n else []
    seen, repeats = set(), 0
    for g in ngrams:
        if g in seen:
            repeats += 1
        seen.add(g)
    repeated_ngram_frac = repeats / max(1, len(ngrams))
    max_run = run = 1
    for a, b in zip(ids, ids[1:]):
        run = run + 1 if a == b else 1
        max_run = max(max_run, run)
    return {"distinct_1": round(distinct_1, 4), "distinct_2": round(distinct_2, 4),
            "repeated_ngram_frac": round(repeated_ngram_frac, 4),
            "max_consecutive_repeat": max_run, "length": len(ids)}


@torch.no_grad()
def context_length_test(model, cfg, device, lengths: Optional[list[int]] = None) -> list[dict]:
    """Forward random inputs at increasing sequence lengths (<= block_size) and confirm
    the model produces finite loss at each — a cheap structural check, not a quality one."""
    model.eval()
    if lengths is None:
        lengths = [8, 64, 256, cfg.block_size]
    lengths = [L for L in lengths if L <= cfg.block_size]
    out = []
    g = torch.Generator().manual_seed(cfg.seed)
    for L in lengths:
        x = torch.randint(0, cfg.vocab_size, (1, L), generator=g).to(device)
        y = torch.randint(0, cfg.vocab_size, (1, L), generator=g).to(device)
        _, loss = model(x, y)
        out.append({"length": L, "loss_finite": bool(torch.isfinite(loss).item()),
                    "loss": round(loss.item(), 4)})
    return out


@torch.no_grad()
def evaluate_generation(model, tokenizer, cfg, device, prompts: list[str],
                        max_new_tokens: int = 80, eos_id: Optional[int] = None) -> list[dict]:
    """Deterministic greedy generation on a fixed prompt set, with repetition metrics."""
    results = []
    for p in prompts:
        text = sample_text(model, tokenizer, cfg, device, p, max_new_tokens,
                           temperature=0.0, eos_id=eos_id)          # greedy = deterministic
        ids = tokenizer.encode(text).ids
        results.append({"prompt": p, "text": text, "repetition": repetition_metrics(ids)})
    return results
