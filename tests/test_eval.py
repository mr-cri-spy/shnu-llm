"""Tests for the deterministic evaluation suite (CPU, tiny model, no training needed)."""
from __future__ import annotations

import os
import sys
import random
import tempfile

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import shnu_llm as S


def _tok(out):
    rng = random.Random(0)
    words = [f"w{i:02d}" for i in range(60)]
    docs = [" ".join(rng.choice(words) for _ in range(30)) for _ in range(200)]
    return S.train_tokenizer(iter(docs), 400, os.path.join(out, "tok.json")), docs


def test_repetition_metrics():
    assert S.repetition_metrics([1, 2, 3, 4, 5])["repeated_ngram_frac"] == 0.0
    deg = S.repetition_metrics([7, 7, 7, 7, 7, 7])
    assert deg["max_consecutive_repeat"] == 6 and deg["repeated_ngram_frac"] > 0.5


def test_eval_suite_runs_and_is_deterministic():
    out = tempfile.mkdtemp()
    tok, docs = _tok(out)
    cfg = S.ShnuConfig(vocab_size=tok.get_vocab_size(), n_layers=2, n_heads=2, d_model=64,
                       block_size=64, batch_size=8, dtype="float32", seed=1337, eval_iters=5)
    eos = tok.token_to_id("<eos>")
    stream = S.build_token_stream(tok, docs, eos)
    tr, va = S.train_val_split(stream, 0.1)
    datasets = {"train": S.PackedDataset(tr, cfg.block_size, "cpu"),
                "val": S.PackedDataset(va, cfg.block_size, "cpu")}
    S.set_seed(cfg.seed)
    model = S.ShnuLM(cfg)

    ppl = S.validation_perplexity(model, datasets, cfg, "cpu", eval_iters=5)
    assert ppl["perplexity"] > 0 and np.isfinite(ppl["val_loss"])

    clt = S.context_length_test(model, cfg, "cpu", [8, 32, 64])
    assert all(r["loss_finite"] for r in clt) and [r["length"] for r in clt] == [8, 32, 64]

    prompts = ["w01 w02", "w10 w11"]
    g1 = S.evaluate_generation(model, tok, cfg, "cpu", prompts, max_new_tokens=16, eos_id=eos)
    g2 = S.evaluate_generation(model, tok, cfg, "cpu", prompts, max_new_tokens=16, eos_id=eos)
    assert [x["text"] for x in g1] == [x["text"] for x in g2]        # greedy => deterministic
    assert all("repetition" in x for x in g1)
    print("eval suite OK:", ppl)


def test_bits_per_byte_matches_hand_computed_value():
    out = tempfile.mkdtemp()
    tok, docs = _tok(out)
    ids = tok.encode(docs[0]).ids
    text = tok.decode(ids)
    nats = 1.2345
    result = S.bits_per_byte(nats, tok, ids)
    expected_bytes = len(text.encode("utf-8"))
    expected_bpb = nats * (len(ids) / expected_bytes) / np.log(2)
    assert result["total_tokens"] == len(ids)
    assert result["total_bytes"] == expected_bytes
    assert abs(result["bits_per_byte"] - expected_bpb) < 1e-6


def test_bits_per_byte_zero_loss_is_zero_bpb():
    out = tempfile.mkdtemp()
    tok, docs = _tok(out)
    ids = tok.encode(docs[0]).ids
    result = S.bits_per_byte(0.0, tok, ids)
    assert result["bits_per_byte"] == 0.0


if __name__ == "__main__":
    test_repetition_metrics()
    test_eval_suite_runs_and_is_deterministic()
    test_bits_per_byte_matches_hand_computed_value()
    test_bits_per_byte_zero_loss_is_zero_bpb()
    print("PASS")
