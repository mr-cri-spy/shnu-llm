"""Optional decoding controls layered on top of ShnuLM's own sampler.

Each function here operates on a single step's logits (shape (B, vocab_size))
and/or the token ids generated so far, and is designed to be a pure,
independently testable building block. `ShnuLM.generate` (see model.py) wires
these in as opt-in keyword arguments; every one of them defaults to "off" so
existing callers that only pass temperature/top_k/top_p/eos_id see identical
behavior to before this module existed.

Implemented controls:
  - repetition_penalty: classic logit-scaling penalty for tokens already seen
    in the sequence (Keskar et al., CTRL-style: positive logits are divided
    by the penalty, negative logits are multiplied by it).
  - no_repeat_ngram_size: hard-blocks any token whose selection would
    reproduce an n-gram that has already appeared in the sequence.
  - top_p (nucleus filtering): kept here as a standalone, independently
    tested function; `ShnuLM.generate` may use either its inline
    implementation or this one, but the behavior is identical (see
    `top_p_filter`'s docstring for the equivalence note).
  - min_new_tokens: suppresses the EOS token's logit until at least this
    many new tokens have been generated.
"""
from __future__ import annotations

from typing import Optional, Sequence

import torch


def apply_repetition_penalty(logits: torch.Tensor, generated_ids: torch.Tensor,
                              penalty: float) -> torch.Tensor:
    """Scale logits for tokens already present in `generated_ids`.

    Standard token-logit repetition penalty: for every token id that already
    appears anywhere in the sequence so far, its logit is divided by
    `penalty` if positive, or multiplied by `penalty` if negative -- in both
    cases pushing the value down (toward less-likely), so a `penalty > 1`
    discourages repetition and `penalty == 1` is a no-op.

    Args:
        logits: (B, vocab_size) logits for the next token.
        generated_ids: (B, T) token ids generated so far (prompt + generation).
        penalty: >= 1.0 strength of the penalty; 1.0 leaves logits unchanged.

    Returns:
        A new (B, vocab_size) tensor; `logits` is not modified in place.
    """
    if penalty is None or penalty == 1.0:
        return logits
    logits = logits.clone()
    B = logits.size(0)
    for b in range(B):
        seen = torch.unique(generated_ids[b])
        seen_logits = logits[b, seen]
        penalized = torch.where(seen_logits > 0, seen_logits / penalty, seen_logits * penalty)
        logits[b, seen] = penalized
    return logits


def block_repeated_ngrams(logits: torch.Tensor, generated_ids: torch.Tensor,
                           ngram_size: int) -> torch.Tensor:
    """Mask out next-token choices that would complete an already-seen n-gram.

    For each batch row, looks at the (ngram_size - 1) most recent tokens as
    the current n-gram prefix, finds every earlier occurrence of that same
    prefix in `generated_ids`, and sets the logit of the token that followed
    each such occurrence to -inf -- so sampling can never reproduce an
    n-gram that has already appeared.

    Args:
        logits: (B, vocab_size) logits for the next token.
        generated_ids: (B, T) token ids generated so far.
        ngram_size: size of the n-gram window to prevent repeating (e.g. 3).
            Values <= 1 are a no-op (there is no meaningful prefix to match).

    Returns:
        A new (B, vocab_size) tensor; `logits` is not modified in place.
    """
    if not ngram_size or ngram_size <= 1:
        return logits
    logits = logits.clone()
    B, T = generated_ids.shape
    prefix_len = ngram_size - 1
    if T < prefix_len:
        return logits
    for b in range(B):
        seq = generated_ids[b].tolist()
        if len(seq) < prefix_len:
            continue
        prefix = tuple(seq[-prefix_len:]) if prefix_len > 0 else ()
        banned = set()
        for i in range(len(seq) - ngram_size + 1):
            if tuple(seq[i:i + prefix_len]) == prefix:
                banned.add(seq[i + prefix_len])
        for tok_id in banned:
            logits[b, tok_id] = -float("inf")
    return logits


def top_p_filter(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    """Nucleus (top-p) filtering: keep the smallest set of highest-probability
    tokens whose cumulative probability reaches `top_p`, masking the rest.

    Equivalent to the inline implementation in `ShnuLM.generate`: sorts
    logits descending, computes the cumulative softmax probability, masks
    every token whose *exclusive* cumulative probability (i.e. the sum of
    strictly higher-probability tokens) already exceeds `top_p` -- which
    guarantees the highest-probability token is always kept, so at least one
    token survives regardless of `top_p`.

    Args:
        logits: (B, vocab_size) logits for the next token.
        top_p: float in (0, 1]; 1.0 (or None) is a no-op.

    Returns:
        A new (B, vocab_size) tensor with masked entries set to -inf.
    """
    if top_p is None or top_p >= 1.0:
        return logits
    sorted_logits, sorted_idx = torch.sort(logits, descending=True)
    probs = torch.softmax(sorted_logits, dim=-1)
    cum = torch.cumsum(probs, dim=-1)
    mask = cum - probs > top_p
    sorted_logits = sorted_logits.masked_fill(mask, -float("inf"))
    out = torch.full_like(logits, -float("inf")).scatter(1, sorted_idx, sorted_logits)
    return out


def enforce_min_new_tokens(logits: torch.Tensor, num_generated: int,
                            min_new_tokens: int, eos_id: Optional[int]) -> torch.Tensor:
    """Suppress the EOS logit until `min_new_tokens` tokens have been generated.

    Args:
        logits: (B, vocab_size) logits for the next token.
        num_generated: how many new tokens have been produced so far (not
            counting the prompt).
        min_new_tokens: minimum number of new tokens required before EOS is
            allowed to be selected. <= 0 is a no-op.
        eos_id: the EOS token id, or None (no-op if there is no EOS token).

    Returns:
        A new (B, vocab_size) tensor; `logits` is not modified in place.
    """
    if not min_new_tokens or min_new_tokens <= 0 or eos_id is None:
        return logits
    if num_generated >= min_new_tokens:
        return logits
    logits = logits.clone()
    logits[:, eos_id] = -float("inf")
    return logits
