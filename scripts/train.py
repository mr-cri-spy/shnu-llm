#!/usr/bin/env python3
"""Pretrain SHNU-LLM from scratch.

Loads an openly-licensed corpus, trains a byte-level BPE tokenizer, builds a
decoder-only Transformer from random initialization, and pretrains it with
checkpointing/resume. All weights are trained from scratch; no pretrained LLM
weights are loaded.

Usage:
    python scripts/train.py --config configs/t4.json
    python scripts/train.py --config configs/smoke.json --max-steps 300
"""
from __future__ import annotations

import os
import sys
import json
import time
import math
import argparse

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import shnu_llm as S


def load_corpus(max_chars: int):
    """Return (records, info). Prefers WikiText-103 (CC BY-SA 3.0); falls back
    to public-domain Tiny Shakespeare if the dataset hub is unreachable."""
    try:
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train", streaming=True)
        buf, total = [], 0
        for ex in ds:
            t = ex["text"]
            if t and t.strip():
                buf.append(t)
                total += len(t)
            if total >= max_chars:
                break
        return buf, {"source": "wikitext-103-raw-v1", "license": "CC BY-SA 3.0",
                     "chars": total, "records": len(buf)}
    except Exception as e:  # network/hub/version issues -> public-domain fallback
        print(f"[dataset] WikiText unavailable ({e!r}); using Tiny Shakespeare fallback")
        import urllib.request
        url = ("https://raw.githubusercontent.com/karpathy/char-rnn/"
               "master/data/tinyshakespeare/input.txt")
        txt = urllib.request.urlopen(url, timeout=60).read().decode("utf-8")
        recs = txt.split("\n\n")
        return recs, {"source": "tiny-shakespeare", "license": "public domain",
                      "chars": len(txt), "records": len(recs)}


def main():
    ap = argparse.ArgumentParser(description="Pretrain SHNU-LLM from scratch.")
    ap.add_argument("--config", required=True, help="path to a JSON config file")
    ap.add_argument("--max-steps", type=int, default=None, help="override training steps")
    ap.add_argument("--data-max-chars", type=int, default=80_000_000,
                    help="max characters of corpus to stream")
    ap.add_argument("--out-dir", default=None, help="override output directory")
    args = ap.parse_args()

    cfg_dict = json.load(open(args.config))
    if args.max_steps is not None:
        cfg_dict["max_steps"] = args.max_steps
    if args.out_dir is not None:
        cfg_dict["out_dir"] = args.out_dir
    cfg = S.ShnuConfig(**cfg_dict)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("environment:", json.dumps(S.env_report(device)))
    S.set_seed(cfg.seed)

    records, info = load_corpus(args.data_max_chars)
    print("dataset:", json.dumps(info))
    records, clean_stats = S.clean_records(records, min_chars=32)
    print("cleaning:", json.dumps(clean_stats))

    tok_path = os.path.join(cfg.out_dir, "tokenizer", "shnu_bpe.json")
    tok = S.train_tokenizer(iter(records[:200_000]), cfg.vocab_size, tok_path)
    cfg.vocab_size = tok.get_vocab_size()
    eos_id = tok.token_to_id("<eos>")
    probe = "SHNU-LLM is a small language model trained from scratch."
    assert S.decode(tok, S.encode(tok, probe)).strip() == probe.strip(), "tokenizer roundtrip failed"
    print(f"tokenizer: vocab={cfg.vocab_size}, roundtrip OK")

    stream = S.build_token_stream(tok, records, eos_id)
    train_ids, val_ids = S.train_val_split(stream, val_ratio=0.02)
    datasets = {
        "train": S.PackedDataset(train_ids, cfg.block_size, device),
        "val": S.PackedDataset(val_ids, cfg.block_size, device),
    }
    print(f"tokens: total={len(stream):,} train={len(train_ids):,} val={len(val_ids):,}")

    S.set_seed(cfg.seed)
    model = S.ShnuLM(cfg).to(device)
    print(f"parameters: {model.num_params():,} (non-embedding {model.num_params(True):,})")
    # Correct random initialization is confirmed by an initial loss near ln(vocab_size):
    xb, yb = datasets["train"].get_batch(4, torch.Generator().manual_seed(0))
    with torch.no_grad():
        _, l0 = model(xb, yb)
    print(f"initialization: random (checkpoint=NONE); initial loss {l0.item():.4f} "
          f"vs ln(vocab)={math.log(cfg.vocab_size):.4f}")

    latest = os.path.join(cfg.out_dir, "checkpoints", "checkpoint_latest.pt")
    resume = latest if os.path.exists(latest) else None
    result = S.train(model, datasets, cfg, device, resume_from=resume,
                     tokenizer=tok, eos_id=eos_id, sample_prompt="The ", verbose=True)

    # Export a self-contained, reloadable model.
    final_dir = os.path.join(cfg.out_dir, "final")
    os.makedirs(os.path.join(final_dir, "tokenizer"), exist_ok=True)
    torch.save({"model": model.state_dict(), "config": S.config.asdict(cfg),
                "version": cfg.version}, os.path.join(final_dir, "shnu_llm_v0.1.pt"))
    open(os.path.join(final_dir, "config.json"), "w").write(cfg.to_json())
    import shutil
    shutil.copy(tok_path, os.path.join(final_dir, "tokenizer", "shnu_bpe.json"))

    report = {
        "project": f"{cfg.name} {cfg.version}", "status": "Completed",
        "parameters": model.num_params(), "vocabulary": cfg.vocab_size,
        "context_length": cfg.block_size, "dataset": info,
        "training_tokens": result["tokens_seen"], "training_steps": result["steps"],
        "final_train_loss": round(result["final_train_loss"], 4),
        "final_val_loss": round(result["final_val_loss"], 4),
        "perplexity": round(result["final_val_ppl"], 2),
        "gpu": S.env_report(device).get("gpu_name", "CPU"),
        "training_time_s": round(result["train_time_s"], 1),
        "model_export": final_dir,
    }
    os.makedirs(os.path.join(cfg.out_dir, "evaluation"), exist_ok=True)
    json.dump(report, open(os.path.join(cfg.out_dir, "evaluation", "final_report.json"), "w"),
              indent=2, default=str)
    print("final report:", json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
