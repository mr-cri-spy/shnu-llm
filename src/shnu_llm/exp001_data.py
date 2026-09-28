"""EXP-001 (v0.3) data preparation and dataset-manifest validation.

EXP-001 changes exactly one variable: the training corpus. It becomes the frozen
v0.2.0 WikiText-103 training stream plus a FineWeb-Edu slice of the same size. This
module builds that mixed corpus deterministically and verifies it before training.

Rules enforced here:

* The v0.2.0 WikiText-103 ``.npy`` streams are reused byte-for-byte, never
  re-prepared. The validation stream must hash to ``WIKITEXT_VAL_SHA256``.
* The v0.2.0 tokenizer is reused unchanged and must hash to ``TOKENIZER_SHA256``.
* The training-loop validation stream for the mix is a byte copy of the v0.2.0
  WikiText validation stream, so in-loop evaluation and ``checkpoint_best``
  selection use the same data as v0.2.0.
* Every output file is SHA-256-hashed into a manifest. ``validate_data_manifest``
  refuses anything that is missing, mismatched, or not yet pinned in the config.

Nothing here downloads data. FineWeb-Edu documents come from an iterator the
caller supplies (a local JSONL file, or a streaming dataset opened by the
preparation script only when explicitly allowed).
"""
from __future__ import annotations

import os
import json
import shutil
import hashlib
import datetime as _dt
from typing import Iterable, Iterator, Optional, Tuple

import numpy as np

from .data import clean_text

MANIFEST_SCHEMA_VERSION = 1
MANIFEST_FILENAME = "EXP001_DATA_MANIFEST.json"

# Two kinds of fixed values, kept apart on purpose.
#
# FROZEN FROM v0.2.0: measured on the actual v0.2.0 artifacts. The two WikiText
# streams were hashed on 2026-09-28 in two independent copies (Kaggle datasets
# shnu-llm-v02 and shnu-ckpt-17k); both copies are identical. Their
# wikitext103_metadata.json has SHA-256 35cf6ec9df48500f1204d830d730f37c8fb4d6a1db4267e38ba9804440db5470.
#
# NEW IN EXP-001: choices made for this experiment; not historical v0.2 facts.
FROZEN_DATA = {
    # --- frozen from v0.2.0 ---
    "tokenizer_sha256": "1bf9397a6213d1a303c9d149d028fa2c456dd1a94da57c233b5b21588d1ac017",
    "vocab_size": 16000,
    "wikitext_dtype": "uint16",
    "wikitext_train_tokens": 116995555,
    "wikitext_train_sha256": "8b5ec7cf9c41f482fe425403dcebd6508921feb1e04c244f3242b23e4e923124",
    "wikitext_train_size_bytes": 233991238,
    "wikitext_val_tokens": 1181773,
    "wikitext_val_sha256": "058b710795d4a8dd6b7655a635eda645d7009b2112cccd53e7712783e5eddcb8",
    "wikitext_val_size_bytes": 2363674,
    "seed": 1337,
    # --- new in EXP-001 ---
    "fineweb_target_train_tokens": 116995555,
    "fineweb_heldout_target_tokens": 1181773,
    "fineweb_min_chars": 64,
    "dataset_key": "wikitext103_fineweb_edu_mix",
}


class DataManifestError(Exception):
    """Raised when EXP-001 data or its manifest is missing, inconsistent, or unpinned."""


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def file_names(dataset_key: str, vocab_size: int) -> dict:
    """File names inside ``<out_dir>/datasets/`` (the layout ``scripts/resume.py`` reads)."""
    return {
        "mix_train": f"{dataset_key}_train_v{vocab_size}.npy",
        "mix_val": f"{dataset_key}_val_v{vocab_size}.npy",
        "fineweb_train": f"fineweb_edu_train_v{vocab_size}.npy",
        "fineweb_heldout": f"fineweb_edu_heldout_v{vocab_size}.npy",
        "fineweb_train_doc_ids": "fineweb_edu_train_doc_ids.txt",
        "fineweb_heldout_doc_ids": "fineweb_edu_heldout_doc_ids.txt",
    }


# ---------------------------------------------------------------------------
# document handling
# ---------------------------------------------------------------------------

def iter_clean_docs(docs: Iterable[Tuple[str, str]], min_chars: int, stats: dict) -> Iterator[Tuple[str, str]]:
    """Streaming equivalent of ``data.clean_records`` for (doc_id, text) pairs.

    Same rules in the same order: drop empty or whitespace-only text, apply
    ``clean_text``, drop text shorter than ``min_chars`` after stripping, and drop
    exact duplicates (by SHA-1 of the cleaned text, so the result is the same in
    every process).
    """
    seen = set()
    for doc_id, text in docs:
        stats["records_in"] = stats.get("records_in", 0) + 1
        if not text or not text.strip():
            stats["removed_empty"] = stats.get("removed_empty", 0) + 1
            continue
        text = clean_text(text)
        if len(text.strip()) < min_chars:
            stats["removed_short"] = stats.get("removed_short", 0) + 1
            continue
        h = hashlib.sha1(text.encode("utf-8")).digest()
        if h in seen:
            stats["removed_dup"] = stats.get("removed_dup", 0) + 1
            continue
        seen.add(h)
        stats["kept"] = stats.get("kept", 0) + 1
        yield doc_id, text


def build_fineweb_slices(docs: Iterable[Tuple[str, str]], tokenizer, eos_id: int, *,
                         train_target: int, heldout_target: int, min_chars: int):
    """Encode cleaned documents in the given order into a training slice and a
    disjoint held-out slice.

    The training slice stops at the first document boundary at or after
    ``train_target`` tokens. The held-out slice takes the following documents and
    stops at the first boundary at or after ``heldout_target``. Every document is
    followed by ``eos_id``, as in ``data.build_token_stream``. Raises if the
    source runs out before both targets are reached.
    """
    stats: dict = {}
    train, heldout = [], []
    train_ids, heldout_ids = [], []
    n_train = n_held = 0
    for doc_id, text in iter_clean_docs(docs, min_chars, stats):
        ids = tokenizer.encode(text).ids + [eos_id]
        if n_train < train_target:
            train.append(np.asarray(ids, dtype=np.uint16))
            train_ids.append(doc_id)
            n_train += len(ids)
        elif n_held < heldout_target:
            heldout.append(np.asarray(ids, dtype=np.uint16))
            heldout_ids.append(doc_id)
            n_held += len(ids)
        else:
            break
    if n_train < train_target or n_held < heldout_target:
        raise DataManifestError(
            f"FineWeb-Edu source exhausted: train {n_train:,}/{train_target:,}, "
            f"held-out {n_held:,}/{heldout_target:,} tokens")
    overlap = set(train_ids) & set(heldout_ids)
    if overlap:
        raise DataManifestError(f"{len(overlap)} document id(s) appear in both train and held-out slices")
    return (np.concatenate(train), np.concatenate(heldout), train_ids, heldout_ids, stats)


def split_at_eos(ids: np.ndarray, eos_id: int):
    """(start, end) half-open ranges, one per document, each ending with its EOS.
    A trailing segment without EOS (the train/val split boundary) is kept as its own segment."""
    ends = np.flatnonzero(ids == eos_id) + 1
    starts = np.concatenate([[0], ends[:-1]]) if len(ends) else np.array([0])
    segs = list(zip(starts.tolist(), ends.tolist())) if len(ends) else []
    last = int(ends[-1]) if len(ends) else 0
    if last < len(ids):
        segs.append((last, len(ids)))
    return segs


def mix_streams(wikitext_train: np.ndarray, fineweb_train: np.ndarray, eos_id: int, seed: int) -> np.ndarray:
    """Document-level shuffle of both streams with a fixed seed, then concatenation.
    Every token of both inputs appears exactly once in the output."""
    parts = [(0, s, e) for s, e in split_at_eos(wikitext_train, eos_id)]
    parts += [(1, s, e) for s, e in split_at_eos(fineweb_train, eos_id)]
    order = np.random.default_rng(seed).permutation(len(parts))
    srcs = (wikitext_train, fineweb_train)
    out = np.concatenate([srcs[parts[i][0]][parts[i][1]:parts[i][2]] for i in order])
    assert len(out) == len(wikitext_train) + len(fineweb_train)
    return out.astype(np.uint16, copy=False)


# ---------------------------------------------------------------------------
# preparation
# ---------------------------------------------------------------------------

def _entry(path: str, tokens: Optional[int] = None) -> dict:
    e = {"file": os.path.basename(path), "sha256": sha256_file(path), "size_bytes": os.path.getsize(path)}
    if tokens is not None:
        e["tokens"] = int(tokens)
    return e


def prepare_exp001_data(*, wikitext_train_path: str, wikitext_val_path: str, tokenizer_path: str,
                        fineweb_docs: Iterable[Tuple[str, str]], out_datasets_dir: str,
                        fineweb_source: dict, frozen: Optional[dict] = None,
                        script_sha256: Optional[str] = None) -> dict:
    """Build the EXP-001 corpus into ``out_datasets_dir`` and write the manifest.

    Verifies the frozen inputs first (tokenizer hash; both WikiText streams'
    SHA-256, size, dtype and token count) and stops before writing anything if one
    differs.
    Refuses to overwrite an existing manifest. Returns the manifest dict.
    """
    fz = dict(FROZEN_DATA, **(frozen or {}))
    problems = []
    tok_sha = sha256_file(tokenizer_path)
    if tok_sha != fz["tokenizer_sha256"]:
        problems.append(f"tokenizer SHA-256 {tok_sha} != frozen {fz['tokenizer_sha256']}")
    val_sha = sha256_file(wikitext_val_path)
    if val_sha != fz["wikitext_val_sha256"]:
        problems.append(f"WikiText validation SHA-256 {val_sha} != frozen {fz['wikitext_val_sha256']}")
    train_sha = sha256_file(wikitext_train_path)
    if train_sha != fz["wikitext_train_sha256"]:
        problems.append(f"WikiText train SHA-256 {train_sha} != frozen {fz['wikitext_train_sha256']}")
    for label, path, key in (("train", wikitext_train_path, "wikitext_train_size_bytes"),
                             ("val", wikitext_val_path, "wikitext_val_size_bytes")):
        if os.path.getsize(path) != fz[key]:
            problems.append(f"WikiText {label} size {os.path.getsize(path)} bytes != frozen {fz[key]}")
    wt_train = np.load(wikitext_train_path)
    wt_val = np.load(wikitext_val_path)
    for label, arr in (("train", wt_train), ("val", wt_val)):
        if str(arr.dtype) != fz["wikitext_dtype"] or arr.ndim != 1:
            problems.append(f"WikiText {label} dtype/shape {arr.dtype}/{arr.shape} != frozen {fz['wikitext_dtype']}/1-D")
    if len(wt_train) != fz["wikitext_train_tokens"]:
        problems.append(f"WikiText train tokens {len(wt_train):,} != frozen {fz['wikitext_train_tokens']:,}")
    if len(wt_val) != fz["wikitext_val_tokens"]:
        problems.append(f"WikiText val tokens {len(wt_val):,} != frozen {fz['wikitext_val_tokens']:,}")
    if problems:
        raise DataManifestError("refusing to prepare EXP-001 data:\n" + "\n".join(problems))

    from .tokenizer import load_tokenizer
    tok = load_tokenizer(tokenizer_path)
    if tok.get_vocab_size() != fz["vocab_size"]:
        raise DataManifestError(f"tokenizer vocab {tok.get_vocab_size()} != frozen {fz['vocab_size']}")
    eos_id = tok.token_to_id("<eos>")

    os.makedirs(out_datasets_dir, exist_ok=True)
    manifest_path = os.path.join(out_datasets_dir, MANIFEST_FILENAME)
    if os.path.exists(manifest_path):
        raise DataManifestError(f"manifest already exists, refusing to overwrite: {manifest_path}")
    names = file_names(fz["dataset_key"], fz["vocab_size"])
    paths = {k: os.path.join(out_datasets_dir, v) for k, v in names.items()}
    for k, p in paths.items():
        if os.path.exists(p):
            raise DataManifestError(f"output already exists, refusing to overwrite: {p}")

    fw_train, fw_held, train_doc_ids, held_doc_ids, clean_stats = build_fineweb_slices(
        fineweb_docs, tok, eos_id, train_target=fz["fineweb_target_train_tokens"],
        heldout_target=fz["fineweb_heldout_target_tokens"], min_chars=fz["fineweb_min_chars"])
    mixed = mix_streams(wt_train, fw_train, eos_id, fz["seed"])

    np.save(paths["fineweb_train"], fw_train)
    np.save(paths["fineweb_heldout"], fw_held)
    np.save(paths["mix_train"], mixed)
    shutil.copyfile(wikitext_val_path, paths["mix_val"])   # byte copy: same SHA as the v0.2.0 stream
    for key, ids in (("fineweb_train_doc_ids", train_doc_ids), ("fineweb_heldout_doc_ids", held_doc_ids)):
        with open(paths[key], "w", encoding="utf-8") as f:
            f.write("".join(f"{i}\n" for i in ids))

    manifest = {
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "experiment": "EXP-001",
        "dataset_key": fz["dataset_key"],
        "created_utc": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "numpy_version": np.__version__,
        "script_sha256": script_sha256,
        "tokenizer": {"file": os.path.basename(tokenizer_path), "sha256": tok_sha, "vocab_size": tok.get_vocab_size(),
                      "eos_id": eos_id},
        "params": {"fineweb_min_chars": fz["fineweb_min_chars"], "seed": fz["seed"],
                   "fineweb_target_train_tokens": fz["fineweb_target_train_tokens"],
                   "fineweb_heldout_target_tokens": fz["fineweb_heldout_target_tokens"],
                   "mixing": "document-level shuffle (numpy default_rng(seed).permutation over EOS-split "
                             "documents of both streams), then concatenation"},
        "fineweb_source": fineweb_source,
        "fineweb_clean_stats": clean_stats,
        "inputs": {
            "wikitext103_train": _entry(wikitext_train_path, len(wt_train)),
            "wikitext103_val": _entry(wikitext_val_path, len(wt_val)),
        },
        "files": {
            "mix_train": _entry(paths["mix_train"], len(mixed)),
            "mix_val": _entry(paths["mix_val"], len(wt_val)),
            "fineweb_train": dict(_entry(paths["fineweb_train"], len(fw_train)), documents=len(train_doc_ids)),
            "fineweb_heldout": dict(_entry(paths["fineweb_heldout"], len(fw_held)), documents=len(held_doc_ids)),
            "fineweb_train_doc_ids": _entry(paths["fineweb_train_doc_ids"]),
            "fineweb_heldout_doc_ids": _entry(paths["fineweb_heldout_doc_ids"]),
        },
    }
    tmp = manifest_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    os.replace(tmp, manifest_path)
    return manifest


# ---------------------------------------------------------------------------
# validation (used by the launcher before any training)
# ---------------------------------------------------------------------------

def pinned_hashes(cfg: dict) -> dict:
    """The manifest hashes that must be pinned in configs/v0.3-exp001.json before training."""
    ds = cfg.get("dataset", {}) or {}
    comps = {c.get("name"): c for c in ds.get("components", []) or []}
    vs = ds.get("validation_sets", {}) or {}
    return {
        "inputs.wikitext103_train": (comps.get("wikitext103") or {}).get("train_npy_sha256"),
        "files.fineweb_train": (comps.get("fineweb_edu") or {}).get("train_npy_sha256"),
        "files.fineweb_train_doc_ids": (comps.get("fineweb_edu") or {}).get("document_id_list_sha256"),
        "files.mix_train": ds.get("mixed_train_npy_sha256"),
        "files.fineweb_heldout": (vs.get("fineweb_edu_heldout") or {}).get("sha256"),
        "files.fineweb_heldout_doc_ids": (vs.get("fineweb_edu_heldout") or {}).get("document_id_list_sha256"),
    }


def validate_data_manifest(datasets_dir: str, cfg: dict, *, frozen: Optional[dict] = None,
                           rehash: bool = True) -> dict:
    """Fail-closed check of ``<out_dir>/datasets`` against its manifest and the config.

    Collects every problem and raises ``DataManifestError`` listing all of them.
    Returns the manifest when everything is consistent.
    """
    fz = dict(FROZEN_DATA, **(frozen or {}))
    errors = []
    mpath = os.path.join(datasets_dir, MANIFEST_FILENAME)
    if not os.path.isfile(mpath):
        raise DataManifestError(f"dataset manifest missing: {mpath}")
    try:
        man = json.load(open(mpath, encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise DataManifestError(f"dataset manifest is not valid JSON: {e}") from e

    if man.get("experiment") != "EXP-001":
        errors.append(f"manifest experiment {man.get('experiment')!r} != 'EXP-001'")
    ds_cfg = cfg.get("dataset", {}) or {}
    if man.get("dataset_key") != ds_cfg.get("dataset_key") or man.get("dataset_key") != fz["dataset_key"]:
        errors.append(f"manifest dataset_key {man.get('dataset_key')!r} != config {ds_cfg.get('dataset_key')!r}")
    tok = man.get("tokenizer", {}) or {}
    if tok.get("sha256") != fz["tokenizer_sha256"]:
        errors.append(f"manifest tokenizer SHA-256 {tok.get('sha256')} != frozen {fz['tokenizer_sha256']}")

    ins, files = man.get("inputs", {}) or {}, man.get("files", {}) or {}
    wt_tr, wt_va = ins.get("wikitext103_train") or {}, ins.get("wikitext103_val") or {}
    if wt_tr.get("tokens") != fz["wikitext_train_tokens"]:
        errors.append(f"manifest WikiText train tokens {wt_tr.get('tokens')} != frozen {fz['wikitext_train_tokens']}")
    if wt_tr.get("sha256") != fz["wikitext_train_sha256"]:
        errors.append(f"manifest WikiText train SHA-256 {wt_tr.get('sha256')} != frozen {fz['wikitext_train_sha256']}")
    if wt_tr.get("size_bytes") != fz["wikitext_train_size_bytes"]:
        errors.append(f"manifest WikiText train size {wt_tr.get('size_bytes')} != frozen {fz['wikitext_train_size_bytes']}")
    if wt_va.get("size_bytes") != fz["wikitext_val_size_bytes"]:
        errors.append(f"manifest WikiText val size {wt_va.get('size_bytes')} != frozen {fz['wikitext_val_size_bytes']}")
    if (man.get("params") or {}).get("fineweb_min_chars") != fz["fineweb_min_chars"]:
        errors.append(f"manifest fineweb_min_chars {(man.get('params') or {}).get('fineweb_min_chars')} "
                      f"!= EXP-001 value {fz['fineweb_min_chars']}")

    # FineWeb-Edu source identity: a pinned hub revision, or the SHA-256 of a local export
    comps = {c.get("name"): c for c in (ds_cfg.get("components") or [])}
    fw_cfg = comps.get("fineweb_edu") or {}
    src = man.get("fineweb_source") or {}
    mode = src.get("mode")
    if not (fw_cfg.get("revision") or fw_cfg.get("source_export_sha256")):
        errors.append("config pins no FineWeb-Edu source identity: set dataset.components[fineweb_edu]."
                      "revision (hub) or source_export_sha256 (local export) after review")
    if mode == "hub-streaming":
        if not src.get("revision"):
            errors.append("manifest FineWeb-Edu source has no revision")
        elif src.get("revision") != fw_cfg.get("revision"):
            errors.append(f"manifest FineWeb-Edu revision {src.get('revision')!r} != config {fw_cfg.get('revision')!r}")
    elif mode == "local-jsonl":
        if not src.get("local_file_sha256"):
            errors.append("manifest FineWeb-Edu local export has no SHA-256")
        elif src.get("local_file_sha256") != fw_cfg.get("source_export_sha256"):
            errors.append(f"manifest FineWeb-Edu export SHA-256 {src.get('local_file_sha256')} != config "
                          f"{fw_cfg.get('source_export_sha256')}")
    else:
        errors.append(f"manifest FineWeb-Edu source mode {mode!r} is not 'hub-streaming' or 'local-jsonl'")
    if wt_va.get("sha256") != fz["wikitext_val_sha256"]:
        errors.append(f"manifest WikiText val SHA-256 {wt_va.get('sha256')} != frozen {fz['wikitext_val_sha256']}")
    if wt_va.get("tokens") != fz["wikitext_val_tokens"]:
        errors.append(f"manifest WikiText val tokens {wt_va.get('tokens')} != frozen {fz['wikitext_val_tokens']}")

    required = ("mix_train", "mix_val", "fineweb_train", "fineweb_heldout",
                "fineweb_train_doc_ids", "fineweb_heldout_doc_ids")
    for k in required:
        if k not in files:
            errors.append(f"manifest lacks files.{k}")
    mv = files.get("mix_val") or {}
    if mv and mv.get("sha256") != fz["wikitext_val_sha256"]:
        errors.append(f"mix validation stream SHA-256 {mv.get('sha256')} is not the frozen v0.2.0 "
                      f"WikiText validation stream {fz['wikitext_val_sha256']}")
    fw, mt, fh = files.get("fineweb_train") or {}, files.get("mix_train") or {}, files.get("fineweb_heldout") or {}
    if fw and fw.get("tokens", 0) < fz["fineweb_target_train_tokens"]:
        errors.append(f"FineWeb-Edu train tokens {fw.get('tokens')} < target {fz['fineweb_target_train_tokens']}")
    if fh and fh.get("tokens", 0) < fz["fineweb_heldout_target_tokens"]:
        errors.append(f"FineWeb-Edu held-out tokens {fh.get('tokens')} < target {fz['fineweb_heldout_target_tokens']}")
    if mt and fw and wt_tr and mt.get("tokens") != wt_tr.get("tokens", 0) + fw.get("tokens", 0):
        errors.append(f"mix train tokens {mt.get('tokens')} != WikiText {wt_tr.get('tokens')} + "
                      f"FineWeb-Edu {fw.get('tokens')}")
    expected_names = file_names(fz["dataset_key"], fz["vocab_size"])
    for k in required:
        if k in files and files[k].get("file") != expected_names[k]:
            errors.append(f"files.{k}.file {files[k].get('file')!r} != expected {expected_names[k]!r}")

    # every listed file must exist and match its recorded size and hash
    if rehash:
        for k, e in files.items():
            p = os.path.join(datasets_dir, e.get("file", ""))
            if not os.path.isfile(p):
                errors.append(f"{k}: file missing: {p}")
                continue
            if os.path.getsize(p) != e.get("size_bytes"):
                errors.append(f"{k}: size {os.path.getsize(p)} != manifest {e.get('size_bytes')}")
                continue
            if sha256_file(p) != e.get("sha256"):
                errors.append(f"{k}: SHA-256 does not match the manifest (file changed or corrupt)")

    # the config must pin the reviewed hashes; null means "not reviewed yet"
    for label, pinned in pinned_hashes(cfg).items():
        sec, key = label.split(".", 1)
        actual = ((man.get(sec) or {}).get(key) or {}).get("sha256")
        if pinned is None:
            errors.append(f"config does not pin {label} (manifest value {actual}); review the manifest "
                          "and record it in configs/v0.3-exp001.json before training")
        elif pinned != actual:
            errors.append(f"{label}: config pins {pinned} but manifest has {actual}")

    if errors:
        raise DataManifestError("\n".join(errors))
    return man
