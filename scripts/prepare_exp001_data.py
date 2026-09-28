#!/usr/bin/env python3
"""Deterministically build the EXP-001 (v0.3) training corpus. Never trains.

The output is the frozen v0.2.0 WikiText-103 train stream plus a FineWeb-Edu slice
of the same size, document-shuffled with seed 1337. It is written to
``<out-dir>/datasets/`` with ``EXP001_DATA_MANIFEST.json`` (SHA-256 of every file).
The training-loop validation stream is a byte copy of the v0.2.0 WikiText
validation stream.

FineWeb-Edu input, one of:
  --fineweb-jsonl PATH   local JSON-lines file with "id" and "text" per line, in
                         dataset order (for offline preparation or testing)
  --allow-download       stream HuggingFaceFW/fineweb-edu from the Hugging Face hub.
                         Refused unless configs/v0.3-exp001.json pins
                         dataset.components[fineweb_edu].revision, so that the
                         document order is reproducible.

Usage (after review; not run as part of registration):
  python scripts/prepare_exp001_data.py --v02-datasets-dir /path/to/v0.2/datasets \
      --tokenizer /path/to/v0.3/run/tokenizer/shnu_bpe.json \
      --out-dir /path/to/v0.3/run --fineweb-jsonl fineweb_sample10bt_ordered.jsonl

After it finishes, review the manifest and pin its hashes in configs/v0.3-exp001.json.
Until they are pinned, scripts/train_exp001.py refuses to start.
"""
from __future__ import annotations

import os
import sys
import json
import argparse
import importlib.util

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO, "src"))
from shnu_llm import exp001_data as X  # noqa: E402

_spec = importlib.util.spec_from_file_location("validate_exp001", os.path.join(REPO, "scripts", "validate_exp001.py"))
V = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(V)


def iter_jsonl(path: str, id_field: str, text_field: str):
    with open(path, "r", encoding="utf-8") as f:
        for n, line in enumerate(f):
            if not line.strip():
                continue
            rec = json.loads(line)
            if id_field not in rec or text_field not in rec:
                raise X.DataManifestError(f"{path}:{n + 1} lacks {id_field!r}/{text_field!r}")
            yield str(rec[id_field]), rec[text_field]


def iter_hub(fw: dict):  # pragma: no cover - network path, never exercised in tests
    from datasets import load_dataset
    ds = load_dataset(fw["hf_dataset"], name=fw["hf_config"], split="train",
                      streaming=True, revision=fw["revision"])
    for rec in ds:
        yield str(rec[fw["id_field"]]), rec[fw["text_field"]]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Build the EXP-001 mixed corpus (no training).")
    ap.add_argument("--config", default=V.DEFAULT_CONFIG)
    ap.add_argument("--v02-datasets-dir", required=True,
                    help="directory holding the v0.2.0 wikitext103_{train,val}_v16000.npy")
    ap.add_argument("--tokenizer", required=True, help="the v0.2.0 tokenizer file")
    ap.add_argument("--out-dir", required=True, help="the new EXP-001 run directory")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--fineweb-jsonl")
    src.add_argument("--allow-download", action="store_true")
    ap.add_argument("--no-branch-check", action="store_true")
    args = ap.parse_args(argv)

    try:
        V.validate(args.config, out_dir=args.out_dir, tokenizer_path=args.tokenizer,
                   check_branch=not args.no_branch_check)
    except V.ValidationError as e:
        print("EXP-001 data preparation refused: config preflight failed", file=sys.stderr)
        for line in str(e).splitlines():
            print(f"  - {line}", file=sys.stderr)
        return 1

    cfg = V.load_config(args.config)
    ds = cfg["dataset"]
    fw = next(c for c in ds["components"] if c["name"] == "fineweb_edu")
    source = {k: fw.get(k) for k in ("hf_dataset", "hf_config", "revision", "id_field", "text_field", "license")}
    if args.allow_download:
        if not fw.get("revision"):
            print("refusing to download: dataset.components[fineweb_edu].revision is not pinned in the config",
                  file=sys.stderr)
            return 1
        docs = iter_hub(fw)
        source["mode"] = "hub-streaming"
    else:
        docs = iter_jsonl(args.fineweb_jsonl, fw["id_field"], fw["text_field"])
        source.update(mode="local-jsonl", local_file=os.path.basename(args.fineweb_jsonl),
                      local_file_sha256=X.sha256_file(args.fineweb_jsonl))

    names = {"train": f"wikitext103_train_v{cfg['tokenizer']['vocab_size']}.npy",
             "val": f"wikitext103_val_v{cfg['tokenizer']['vocab_size']}.npy"}
    try:
        man = X.prepare_exp001_data(
            wikitext_train_path=os.path.join(args.v02_datasets_dir, names["train"]),
            wikitext_val_path=os.path.join(args.v02_datasets_dir, names["val"]),
            tokenizer_path=args.tokenizer, fineweb_docs=docs,
            out_datasets_dir=os.path.join(args.out_dir, "datasets"),
            fineweb_source=source, script_sha256=X.sha256_file(os.path.abspath(__file__)))
    except (X.DataManifestError, FileNotFoundError) as e:
        print(f"EXP-001 data preparation FAILED:\n{e}", file=sys.stderr)
        return 1
    print(json.dumps({k: {"file": v["file"], "sha256": v["sha256"], "tokens": v.get("tokens")}
                      for k, v in {**man["inputs"], **man["files"]}.items()}, indent=2))
    print("Review the manifest, then pin these hashes in configs/v0.3-exp001.json. No training was started.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
