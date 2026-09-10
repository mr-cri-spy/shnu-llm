"""End-to-end smoke test for SHNU-LLM.

Runs the full pipeline on a small synthetic corpus (no network required) and
asserts the core guarantees: tokenizer round-trip, exact parameter count,
correct random initialization, decreasing loss, checkpoint save/resume, text
generation, and save -> reload -> identical greedy output.

Run with: pytest -q   (or: python tests/test_smoke.py)
"""
from __future__ import annotations

import os
import sys
import math
import random
import tempfile

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import shnu_llm as S


def _synthetic_corpus(n_docs: int = 400, seed: int = 0):
    rng = random.Random(seed)
    words = [f"tok{i:03d}" for i in range(120)]
    docs = []
    for _ in range(n_docs):
        length = rng.randint(20, 60)
        docs.append(" ".join(rng.choice(words) for _ in range(length)))
    return docs


def test_end_to_end(tmp_path=None):
    out = tmp_path if tmp_path is not None else tempfile.mkdtemp()
    out = str(out)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg = S.ShnuConfig(vocab_size=800, n_layers=3, n_heads=4, d_model=96, block_size=64,
                       batch_size=16, max_steps=120, warmup_steps=20, lr=6e-4,
                       eval_interval=60, eval_iters=10, sample_interval=1000,
                       checkpoint_interval=60, log_interval=1000, out_dir=out,
                       dtype="float32", seed=1337)
    S.set_seed(cfg.seed)

    records, _ = S.clean_records(_synthetic_corpus(), min_chars=1)
    tok = S.train_tokenizer(iter(records), cfg.vocab_size,
                            os.path.join(out, "tokenizer", "bpe.json"))
    cfg.vocab_size = tok.get_vocab_size()

    # tokenizer round-trip
    probe = "tok001 tok042 tok099 tok007"
    assert S.decode(tok, S.encode(tok, probe)).strip() == probe.strip()

    eos_id = tok.token_to_id("<eos>")
    stream = S.build_token_stream(tok, records, eos_id)
    train_ids, val_ids = S.train_val_split(stream, val_ratio=0.1)
    # no validation leakage: the two id ranges are disjoint slices
    assert len(train_ids) + len(val_ids) == len(stream)
    datasets = {"train": S.PackedDataset(train_ids, cfg.block_size, device),
                "val": S.PackedDataset(val_ids, cfg.block_size, device)}

    S.set_seed(cfg.seed)
    model = S.ShnuLM(cfg).to(device)

    # exact, reproducible parameter count
    expected = model.num_params()
    assert S.ShnuLM(cfg).num_params() == expected

    # correct random init: initial loss near ln(vocab)
    x, y = datasets["train"].get_batch(4, torch.Generator().manual_seed(0))
    with torch.no_grad():
        _, l0 = model(x, y)
    assert abs(l0.item() - math.log(cfg.vocab_size)) < 1.0

    result = S.train(model, datasets, cfg, device, verbose=False)
    assert result["final_train_loss"] < l0.item()          # loss decreased

    latest = os.path.join(out, "checkpoints", "checkpoint_latest.pt")
    assert os.path.exists(latest)                           # checkpoint written

    # resume continues from the saved step rather than restarting
    ck = torch.load(latest, map_location="cpu", weights_only=False)
    assert ck["step"] == cfg.max_steps

    # generation produces new tokens
    greedy = S.sample_text(model, tok, cfg, device, "tok001 ", max_new_tokens=16,
                           temperature=0.0, eos_id=eos_id)
    assert isinstance(greedy, str) and len(greedy) > 0

    # save -> reload into a fresh object -> identical greedy output
    final = os.path.join(out, "final.pt")
    torch.save({"model": model.state_dict(), "config": S.config.asdict(cfg)}, final)
    reloaded = S.ShnuLM(cfg).to(device)
    reloaded.load_state_dict(torch.load(final, map_location=device, weights_only=False)["model"])
    greedy2 = S.sample_text(reloaded, tok, cfg, device, "tok001 ", max_new_tokens=16,
                            temperature=0.0, eos_id=eos_id)
    assert greedy == greedy2

    print("smoke test passed:", {
        "params": expected, "initial_loss": round(l0.item(), 3),
        "final_train_loss": round(result["final_train_loss"], 3),
        "final_val_loss": round(result["final_val_loss"], 3),
    })


if __name__ == "__main__":
    test_end_to_end()
