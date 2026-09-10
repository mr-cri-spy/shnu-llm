"""Recovery test: prove training resumes from the exact saved step, not from zero.

Simulates a Colab interruption: train a few steps and checkpoint, then discard
all in-memory objects, discover the checkpoint on disk, verify compatibility,
and resume. Asserts the resumed step equals the previously saved step and that
training continues past it.

Run with: pytest -q tests/test_recovery.py   (or: python tests/test_recovery.py)
"""
from __future__ import annotations

import os
import sys
import random
import tempfile

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import shnu_llm as S


def _corpus(seed=0):
    rng = random.Random(seed)
    words = [f"w{i:02d}" for i in range(80)]
    return [" ".join(rng.choice(words) for _ in range(rng.randint(20, 50))) for _ in range(300)]


def _make_cfg(out, max_steps):
    return S.ShnuConfig(vocab_size=600, n_layers=2, n_heads=2, d_model=64, block_size=48,
                        batch_size=16, max_steps=max_steps, warmup_steps=5, lr=6e-4,
                        eval_interval=1000, eval_iters=5, sample_interval=10000,
                        checkpoint_interval=1000, log_interval=10000, out_dir=out,
                        dtype="float32", seed=1337)


def _build_data(cfg, device):
    S.set_seed(cfg.seed)
    records, _ = S.clean_records(_corpus(), min_chars=1)
    tok_path = os.path.join(cfg.out_dir, "tokenizer", "bpe.json")
    if os.path.exists(tok_path):
        tok = S.load_tokenizer(tok_path)
    else:
        tok = S.train_tokenizer(iter(records), cfg.vocab_size, tok_path)
    cfg.vocab_size = tok.get_vocab_size()
    eos = tok.token_to_id("<eos>")
    stream = S.build_token_stream(tok, records, eos)
    tr, va = S.train_val_split(stream, val_ratio=0.1)
    return {"train": S.PackedDataset(tr, cfg.block_size, device),
            "val": S.PackedDataset(va, cfg.block_size, device)}, tok


def test_recovery(tmp_path=None):
    out = str(tmp_path if tmp_path is not None else tempfile.mkdtemp())
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Phase A: train to step 30 and checkpoint, then drop everything ---
    cfgA = _make_cfg(out, max_steps=30)
    dsA, _ = _build_data(cfgA, device)
    S.set_seed(cfgA.seed)
    modelA = S.ShnuLM(cfgA).to(device)
    S.train(modelA, dsA, cfgA, device, verbose=False)
    del modelA, dsA

    ckpt_dir = os.path.join(out, "checkpoints")
    latest = S.find_latest_checkpoint(ckpt_dir)
    assert latest is not None, "no checkpoint discovered"
    previous_step = S.checkpoint_step(latest)

    # --- Phase B: fresh process simulation — discover, verify, resume ---
    cfgB = _make_cfg(out, max_steps=50)
    dsB, _ = _build_data(cfgB, device)
    ckpt = torch.load(latest, map_location=device, weights_only=False)
    ok, mism = S.verify_checkpoint_compatible(ckpt, cfgB)
    assert ok, f"checkpoint incompatible: {mism}"
    S.set_seed(cfgB.seed)
    modelB = S.ShnuLM(cfgB).to(device)
    result = S.train(modelB, dsB, cfgB, device, resume_from=latest, verbose=False)
    resumed_step = result["start_step"]
    new_step = result["steps"]

    print(f"previous step: {previous_step}")
    print(f"resumed step:  {resumed_step}")
    print(f"new step:      {new_step}")

    assert previous_step == 30
    assert resumed_step == 30, "training restarted instead of resuming!"
    assert new_step == 50 and new_step > resumed_step
    print("RECOVERY OK — resumed from exact step, did not restart from zero")


if __name__ == "__main__":
    test_recovery()
