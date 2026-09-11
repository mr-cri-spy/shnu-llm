#!/usr/bin/env python3
"""Deterministic v0.2 evaluation suite for SHNU-LLM.

Loads a trained checkpoint (or a random-init model, for a structural dry run),
the tokenizer, and the fixed validation stream, then reports validation loss,
perplexity, a fixed-prompt greedy generation set with repetition metrics, and a
context-length structural test. Reproducible: fixed seed, greedy decoding.

Usage:
  python evaluation/evaluate.py --config configs/v0.2.json \
      --out-dir /content/drive/MyDrive/SHNU_LLM --dataset wikitext103 \
      --checkpoint /content/drive/MyDrive/SHNU_LLM/checkpoints/checkpoint_latest.pt
"""
from __future__ import annotations

import os
import sys
import json
import argparse

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import shnu_llm as S

HERE = os.path.dirname(__file__)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/v0.2.json")
    ap.add_argument("--out-dir", required=True, help="dir with tokenizer/ and datasets/ (e.g. Drive)")
    ap.add_argument("--dataset", default="wikitext103")
    ap.add_argument("--checkpoint", default=None, help="checkpoint to load; omit for a random-init dry run")
    ap.add_argument("--prompts", default=os.path.join(HERE, "eval_prompts.json"))
    args = ap.parse_args()

    spec = json.load(open(args.prompts))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = S.ShnuConfig(**json.load(open(args.config)))
    cfg.seed = spec.get("seed", cfg.seed)

    tok = S.load_tokenizer(os.path.join(args.out_dir, "tokenizer", "shnu_bpe.json"))
    cfg.vocab_size = tok.get_vocab_size()
    eos_id = tok.token_to_id("<eos>")

    va = np.load(os.path.join(args.out_dir, "datasets", f"{args.dataset}_val_v{cfg.vocab_size}.npy"))
    tr_path = os.path.join(args.out_dir, "datasets", f"{args.dataset}_train_v{cfg.vocab_size}.npy")
    tr = np.load(tr_path) if os.path.exists(tr_path) else va
    datasets = {"train": S.PackedDataset(tr, cfg.block_size, device),
                "val": S.PackedDataset(va, cfg.block_size, device)}

    S.set_seed(cfg.seed)
    model = S.ShnuLM(cfg).to(device)
    loaded = "random-init (dry run)"
    if args.checkpoint and os.path.exists(args.checkpoint):
        ck = torch.load(args.checkpoint, map_location=device, weights_only=False)
        ok, mism = S.verify_checkpoint_compatible(ck, cfg)
        if not ok:
            raise SystemExit(f"[eval] checkpoint incompatible: {mism}")
        model.load_state_dict(ck["model"])
        loaded = f"{os.path.basename(args.checkpoint)} @ step {ck.get('step')}"

    report = {
        "checkpoint": loaded,
        "model_parameters": model.num_params(),
        "vocab_size": cfg.vocab_size,
        "device": S.env_report(device),
        "perplexity": S.validation_perplexity(model, datasets, cfg, device, spec.get("eval_iters", 100)),
        "context_length_test": S.context_length_test(model, cfg, device, spec.get("context_lengths")),
        "generation": S.evaluate_generation(model, tok, cfg, device, spec["prompts"],
                                            spec.get("max_new_tokens", 80), eos_id=eos_id),
    }
    # aggregate degeneration signal across the fixed prompt set
    reps = [g["repetition"]["repeated_ngram_frac"] for g in report["generation"]]
    report["mean_repeated_ngram_frac"] = round(sum(reps) / max(1, len(reps)), 4)

    os.makedirs(os.path.join(args.out_dir, "evaluation"), exist_ok=True)
    out = os.path.join(args.out_dir, "evaluation", "eval_report.json")
    json.dump(report, open(out, "w"), indent=2, default=str)
    # console summary (compact)
    print(json.dumps({k: report[k] for k in
                      ["checkpoint", "model_parameters", "vocab_size",
                       "perplexity", "mean_repeated_ngram_frac"]}, indent=2, default=str))
    print("context_length_test:", report["context_length_test"])
    for g in report["generation"][:3]:
        print(f"  [{g['prompt']!r}] -> {g['text'][:80]!r} rep={g['repetition']['repeated_ngram_frac']}")
    print("full report ->", out)


if __name__ == "__main__":
    main()
