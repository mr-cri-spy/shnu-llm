"""CPU-only, deterministic tests for the optional decoding controls.

No CUDA and no trained checkpoint are needed: these use either raw logit
tensors (for the pure decoding.py helpers) or a tiny randomly-initialized
ShnuLM (for the ShnuLM.generate integration / backward-compatibility checks).
"""
from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import shnu_llm as S
from shnu_llm.decoding import (
    apply_repetition_penalty, block_repeated_ngrams, top_p_filter, enforce_min_new_tokens,
)


# ---------------------------------------------------------------------------
# repetition_penalty
# ---------------------------------------------------------------------------

def test_repetition_penalty_scales_seen_tokens_only():
    logits = torch.tensor([[2.0, -2.0, 5.0, 0.5]])
    generated = torch.tensor([[0, 1]])  # tokens 0 and 1 have appeared
    out = apply_repetition_penalty(logits, generated, penalty=2.0)
    # token 0: positive logit -> divided by penalty
    assert torch.isclose(out[0, 0], torch.tensor(1.0))
    # token 1: negative logit -> multiplied by penalty
    assert torch.isclose(out[0, 1], torch.tensor(-4.0))
    # tokens 2, 3: unseen -> unchanged
    assert torch.isclose(out[0, 2], torch.tensor(5.0))
    assert torch.isclose(out[0, 3], torch.tensor(0.5))


def test_repetition_penalty_noop_when_one_or_none():
    logits = torch.tensor([[2.0, -2.0, 5.0]])
    generated = torch.tensor([[0, 1]])
    assert torch.equal(apply_repetition_penalty(logits, generated, 1.0), logits)
    assert torch.equal(apply_repetition_penalty(logits, generated, None), logits)


# ---------------------------------------------------------------------------
# no_repeat_ngram_size
# ---------------------------------------------------------------------------

def test_no_repeat_ngram_blocks_the_repeated_trigram():
    # Sequence so far: 1, 2, 3, 1, 2  -- the trigram (1, 2, 3) already occurred
    # once (positions 0-2). With ngram_size=3, the 2-token prefix (1, 2) at
    # the end must not be followed by 3 again (that would reproduce (1,2,3)).
    generated = torch.tensor([[1, 2, 3, 1, 2]])
    vocab_size = 5
    logits = torch.zeros(1, vocab_size)  # uniform: token 3 would otherwise be a valid pick
    out = block_repeated_ngrams(logits, generated, ngram_size=3)
    assert out[0, 3].item() == -float("inf")
    # every other token must remain unblocked
    for tok_id in [0, 1, 2, 4]:
        assert out[0, tok_id].item() == 0.0


def test_no_repeat_ngram_noop_below_size_two():
    logits = torch.zeros(1, 5)
    generated = torch.tensor([[1, 2, 3, 1, 2]])
    assert torch.equal(block_repeated_ngrams(logits, generated, ngram_size=1), logits)
    assert torch.equal(block_repeated_ngrams(logits, generated, ngram_size=0), logits)
    assert torch.equal(block_repeated_ngrams(logits, generated, ngram_size=None), logits)


# ---------------------------------------------------------------------------
# top_p (nucleus) filtering
# ---------------------------------------------------------------------------

def test_top_p_filter_keeps_minimal_nucleus():
    # Probabilities (after softmax) will be dominated by index 0; construct
    # logits whose softmax gives an easy-to-reason-about distribution.
    logits = torch.log(torch.tensor([[0.7, 0.2, 0.05, 0.05]]))
    out = top_p_filter(logits, top_p=0.75)
    finite = torch.isfinite(out[0])
    # cumulative prob after token 0 alone = 0.7 < 0.75, so token 1 must also
    # survive (0.7+0.2=0.9 >= 0.75); tokens 2 and 3 must be masked.
    assert finite[0] and finite[1]
    assert not finite[2] and not finite[3]


def test_top_p_filter_always_keeps_at_least_one_token():
    logits = torch.log(torch.tensor([[0.7, 0.2, 0.05, 0.05]]))
    out = top_p_filter(logits, top_p=1e-6)
    assert torch.isfinite(out[0]).sum().item() == 1
    assert torch.isfinite(out[0, 0])  # the highest-probability token survives


def test_top_p_filter_noop_at_one_or_none():
    logits = torch.tensor([[1.0, 2.0, 3.0]])
    assert torch.equal(top_p_filter(logits, 1.0), logits)
    assert torch.equal(top_p_filter(logits, None), logits)


# ---------------------------------------------------------------------------
# min_new_tokens
# ---------------------------------------------------------------------------

def test_min_new_tokens_suppresses_eos_until_reached():
    eos_id = 2
    logits = torch.tensor([[0.0, 0.0, 10.0]])  # EOS is the clear argmax
    blocked = enforce_min_new_tokens(logits, num_generated=1, min_new_tokens=3, eos_id=eos_id)
    assert blocked[0, eos_id].item() == -float("inf")

    allowed = enforce_min_new_tokens(logits, num_generated=3, min_new_tokens=3, eos_id=eos_id)
    assert allowed[0, eos_id].item() == 10.0  # unaffected once threshold reached


def test_min_new_tokens_noop_without_eos_or_threshold():
    logits = torch.tensor([[0.0, 0.0, 10.0]])
    assert torch.equal(enforce_min_new_tokens(logits, 0, min_new_tokens=3, eos_id=None), logits)
    assert torch.equal(enforce_min_new_tokens(logits, 0, min_new_tokens=0, eos_id=2), logits)


# ---------------------------------------------------------------------------
# ShnuLM.generate integration + backward compatibility
# ---------------------------------------------------------------------------

def _tiny_cfg():
    return S.ShnuConfig(vocab_size=32, n_layers=2, n_heads=2, d_model=16,
                        block_size=32, dtype="float32", seed=1337)


def test_generate_backward_compatible_when_new_kwargs_omitted():
    """Calling generate() the old way (no repetition_penalty/no_repeat_ngram_size/
    min_new_tokens) must produce byte-identical output to explicitly passing
    those new kwargs as their no-op defaults."""
    cfg = _tiny_cfg()
    S.set_seed(cfg.seed)
    model = S.ShnuLM(cfg)
    model.eval()
    idx = torch.tensor([[1, 2, 3]])

    S.set_seed(123)
    old_style = model.generate(idx.clone(), max_new_tokens=10, temperature=0.8, top_k=10)

    S.set_seed(123)
    new_style = model.generate(idx.clone(), max_new_tokens=10, temperature=0.8, top_k=10,
                               repetition_penalty=None, no_repeat_ngram_size=None,
                               min_new_tokens=None)

    assert torch.equal(old_style, new_style)


def test_generate_greedy_deterministic_with_and_without_new_params():
    """Greedy (temperature<=0) generation must remain deterministic and
    unaffected by the new params when they are left at no-op defaults."""
    cfg = _tiny_cfg()
    S.set_seed(cfg.seed)
    model = S.ShnuLM(cfg)
    model.eval()
    idx = torch.tensor([[1, 2, 3]])

    out1 = model.generate(idx.clone(), max_new_tokens=8, temperature=0.0)
    out2 = model.generate(idx.clone(), max_new_tokens=8, temperature=0.0,
                          repetition_penalty=1.0, no_repeat_ngram_size=0, min_new_tokens=0)
    assert torch.equal(out1, out2)


def test_generate_repetition_penalty_changes_greedy_trajectory():
    """A strong repetition penalty must be able to steer greedy decoding away
    from the unpenalized trajectory once a token repeats."""
    cfg = _tiny_cfg()
    S.set_seed(cfg.seed)
    model = S.ShnuLM(cfg)
    model.eval()
    idx = torch.tensor([[1, 2, 3]])

    baseline = model.generate(idx.clone(), max_new_tokens=15, temperature=0.0)
    penalized = model.generate(idx.clone(), max_new_tokens=15, temperature=0.0,
                               repetition_penalty=5.0)
    # Not asserting a specific output (depends on random init), just that the
    # penalty is actually wired through generate() and can change behavior,
    # or -- if the unpenalized trajectory never repeats a token -- that both
    # runs stay valid same-shape tensors.
    assert baseline.shape == penalized.shape
    if len(set(baseline[0, 3:].tolist())) < baseline[0, 3:].numel():
        # baseline does repeat somewhere in the generated region -> the two
        # trajectories must diverge once the penalty kicks in
        assert not torch.equal(baseline, penalized)


def test_generate_min_new_tokens_prevents_early_eos():
    cfg = _tiny_cfg()
    S.set_seed(cfg.seed)
    model = S.ShnuLM(cfg)
    model.eval()
    eos_id = 0
    idx = torch.tensor([[1, 2, 3]])
    out = model.generate(idx.clone(), max_new_tokens=20, temperature=0.0,
                         eos_id=eos_id, min_new_tokens=10)
    generated = out[0, idx.size(1):]
    if eos_id in generated.tolist():
        first_eos = generated.tolist().index(eos_id)
        assert first_eos + 1 >= 10


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
