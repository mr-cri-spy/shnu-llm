"""Pre-flight safety checks for SHNU-LLM long training on free Colab.

These checks exist to make a long training run **refuse to start** in an unsafe
environment rather than silently doing the wrong thing (e.g. burning hours on CPU,
or resuming from an incompatible checkpoint). Every check raises `PreflightError`
on failure; nothing here falls back to CPU.

Guards:
  * a GPU is present (and, by default, is a T4)
  * Google Drive is mounted
  * enough free disk for checkpoints
  * the dataset matches its recorded SHA-256
  * the tokenizer matches its recorded SHA-256
  * a discovered checkpoint is architecture-compatible
"""
from __future__ import annotations

import os
import json
import shutil
import hashlib

import torch

from .config import ShnuConfig
from .persistence import verify_checkpoint_compatible


class PreflightError(RuntimeError):
    """Raised when an environment/integrity check fails; long training must stop."""


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def require_gpu(require_t4: bool = True) -> dict:
    if not torch.cuda.is_available():
        raise PreflightError("no CUDA GPU available — refusing to run long training on CPU")
    name = torch.cuda.get_device_name(0)
    if require_t4 and "T4" not in name:
        raise PreflightError(f"GPU is {name!r}, not a T4 — refusing (set require_t4=False to override)")
    props = torch.cuda.get_device_properties(0)
    return {"gpu": name, "vram_gb": round(props.total_memory / 1e9, 2),
            "cuda": torch.version.cuda}


def require_drive_mounted(drive_root: str = "/content/drive/MyDrive") -> None:
    if not os.path.isdir(drive_root):
        raise PreflightError(f"Google Drive not mounted at {drive_root} — mount it first")


def require_free_disk(path: str, min_gb: float = 2.0) -> float:
    free_gb = shutil.disk_usage(path).free / 1e9
    if free_gb < min_gb:
        raise PreflightError(f"insufficient free disk at {path}: {free_gb:.1f} GB < {min_gb} GB required")
    return round(free_gb, 2)


def verify_dataset(metadata_path: str, expected_sha256_prefix: str | None = None,
                   expected_train_tokens: int | None = None,
                   expected_val_tokens: int | None = None) -> dict:
    if not os.path.exists(metadata_path):
        raise PreflightError(f"dataset metadata missing: {metadata_path}")
    meta = json.load(open(metadata_path))
    sha = meta.get("raw_sha256", "")
    if expected_sha256_prefix and not sha.startswith(expected_sha256_prefix):
        raise PreflightError(f"dataset SHA-256 mismatch: {sha[:16]}… != expected {expected_sha256_prefix}…")
    tok = meta.get("tokens", {})
    if expected_train_tokens is not None and tok.get("train") != expected_train_tokens:
        raise PreflightError(f"train token count {tok.get('train')} != expected {expected_train_tokens}")
    if expected_val_tokens is not None and tok.get("val") != expected_val_tokens:
        raise PreflightError(f"val token count {tok.get('val')} != expected {expected_val_tokens}")
    return {"dataset_sha256": sha, "tokens": tok, "license": meta.get("license")}


def verify_tokenizer(tokenizer_path: str, expected_sha256_prefix: str | None = None,
                     expected_vocab: int | None = None) -> dict:
    if not os.path.exists(tokenizer_path):
        raise PreflightError(f"tokenizer missing: {tokenizer_path}")
    sha = _sha256(tokenizer_path)
    if expected_sha256_prefix and not sha.startswith(expected_sha256_prefix):
        raise PreflightError(f"tokenizer SHA-256 mismatch: {sha[:16]}… != expected {expected_sha256_prefix}…")
    result = {"tokenizer_sha256": sha}
    if expected_vocab is not None:
        from .tokenizer import load_tokenizer
        vocab = load_tokenizer(tokenizer_path).get_vocab_size()
        if vocab != expected_vocab:
            raise PreflightError(f"tokenizer vocab {vocab} != expected {expected_vocab}")
        result["vocab_size"] = vocab
    return result


def verify_checkpoint(checkpoint_path: str, cfg: ShnuConfig, device: str = "cpu") -> dict:
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        raise PreflightError("no checkpoint to verify")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    ok, mism = verify_checkpoint_compatible(ckpt, cfg)
    if not ok:
        raise PreflightError(f"checkpoint architecture mismatch: {mism}")
    return {"checkpoint_step": int(ckpt.get("step", -1)), "compatible": True}


def run_preflight(cfg: ShnuConfig, *, require_t4: bool = True, min_free_gb: float = 2.0,
                  dataset_metadata: str | None = None, tokenizer_path: str | None = None,
                  expected: dict | None = None, checkpoint_path: str | None = None,
                  drive_root: str = "/content/drive/MyDrive") -> dict:
    """Run all applicable checks. Returns a report dict; raises PreflightError on any failure."""
    expected = expected or {}
    report = {"passed": False}
    report["gpu"] = require_gpu(require_t4=require_t4)
    require_drive_mounted(drive_root)
    report["free_disk_gb"] = require_free_disk(cfg.out_dir if os.path.isdir(cfg.out_dir) else "/content",
                                               min_gb=min_free_gb)
    if dataset_metadata:
        report["dataset"] = verify_dataset(dataset_metadata,
                                           expected.get("dataset_sha256_prefix"),
                                           expected.get("train_tokens"), expected.get("val_tokens"))
    if tokenizer_path:
        report["tokenizer"] = verify_tokenizer(tokenizer_path,
                                               expected.get("tokenizer_sha256_prefix"),
                                               expected.get("vocab_size"))
    if checkpoint_path:
        report["checkpoint"] = verify_checkpoint(checkpoint_path, cfg)
    report["passed"] = True
    return report
