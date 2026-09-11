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


if __name__ == "__main__":
    test_repetition_metrics()
    test_eval_suite_runs_and_is_deterministic()
    print("PASS")
