"""Cross-platform checkpoint safety for SHNU-LLM multi-platform training.

The v0.2 run continues across several free-GPU platforms (Colab, Kaggle, ...).
Colab mounts Google Drive natively; **Kaggle cannot**, so a checkpoint sometimes
has to travel between a platform and the canonical Drive store as a file bundle.
That movement is the dangerous moment — it is where an older or corrupt
checkpoint could silently overwrite a newer, valid one. This module makes that
impossible.

Guarantees enforced here, on **every** platform:

  * A checkpoint is trusted only after content verification — it exists, is
    non-empty, its SHA-256 matches the manifest, it loads, it carries a model
    state dict and a training ``step``, and its architecture matches the config.
  * A **newer** valid checkpoint is **never** replaced by an older one.
  * A **valid** checkpoint is **never** replaced by a corrupt/unverifiable one.
  * The **checkpoint is authoritative** over any status/metadata file.
  * Exactly **one** checkpoint is authoritative at a time; the bundle carries a
    manifest so the receiving side re-verifies before adopting anything.

Nothing here trains, mutates optimizer state, or edits step numbers. It only
inspects, verifies, and safely moves checkpoint *files*. It never deletes the
real checkpoint: adoption is an atomic ``os.replace`` of a verified file, and a
rejected import leaves the destination untouched.
"""
from __future__ import annotations

import os
import json
import time
import shutil
import hashlib
import tarfile
import tempfile
import datetime as _dt
from typing import Optional, Tuple, Iterable

from .config import ShnuConfig
from .persistence import (
    find_latest_checkpoint, verify_checkpoint_compatible, checkpoint_step,
)

BUNDLE_MANIFEST_NAME = "MANIFEST.json"
BUNDLE_SCHEMA_VERSION = 1

# Architecture fields recorded/compared for a checkpoint (mirror persistence._ARCH_KEYS).
_ARCH_KEYS = ("n_layers", "n_heads", "n_kv_heads", "d_model", "d_ff",
              "block_size", "vocab_size", "tie_embeddings")


def sha256_file(path: str, chunk: int = 1 << 20) -> str:
    """Streaming SHA-256 of a file (never loads the whole file into memory)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _utc_from_mtime(path: str) -> Optional[str]:
    try:
        return _dt.datetime.fromtimestamp(
            os.path.getmtime(path), _dt.timezone.utc
        ).strftime("%Y-%m-%dT%H:%M:%SZ")
    except OSError:
        return None


def expected_param_count(cfg: ShnuConfig) -> int:
    """Deterministic parameter count for a config (builds the model once on CPU)."""
    from .model import ShnuLM  # local import: avoid importing torch at module load
    from .utils import set_seed
    set_seed(getattr(cfg, "seed", 1337))
    model = ShnuLM(cfg)
    n = model.num_params()
    del model
    return n


def inspect_checkpoint(path: str, *, compute_sha: bool = True) -> dict:
    """Read a checkpoint file and report what it contains — no config needed.

    Returns a dict; ``valid`` is True only if the file loads and carries the
    minimum training payload (a model state dict and an integer ``step``).
    Never raises: an unreadable/corrupt file comes back as ``valid=False`` with
    a ``reason``, so callers can fail closed instead of crashing.
    """
    info = {
        "path": path,
        "filename": os.path.basename(path) if path else None,
        "exists": bool(path) and os.path.exists(path),
        "size_bytes": None,
        "sha256": None,
        "step": None,
        "best_val": None,
        "arch": None,
        "has_model": False,
        "has_optimizer": False,
        "mtime_utc": None,
        "valid": False,
        "reason": None,
    }
    if not info["exists"]:
        info["reason"] = "file does not exist"
        return info
    try:
        info["size_bytes"] = os.path.getsize(path)
        info["mtime_utc"] = _utc_from_mtime(path)
    except OSError as e:
        info["reason"] = f"cannot stat file: {e!r}"
        return info
    if info["size_bytes"] == 0:
        info["reason"] = "file is empty (0 bytes)"
        return info
    if compute_sha:
        try:
            info["sha256"] = sha256_file(path)
        except OSError as e:
            info["reason"] = f"cannot read file for hashing: {e!r}"
            return info
    try:
        import torch
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as e:  # torch raises many things on a truncated/corrupt file
        info["reason"] = f"checkpoint does not load (corrupt?): {e!r}"
        return info
    if not isinstance(ckpt, dict):
        info["reason"] = "checkpoint payload is not a dict"
        return info
    info["has_model"] = isinstance(ckpt.get("model"), dict) and len(ckpt["model"]) > 0
    info["has_optimizer"] = "optimizer" in ckpt
    step = ckpt.get("step")
    info["step"] = int(step) if isinstance(step, int) else None
    bv = ckpt.get("best_val")
    info["best_val"] = float(bv) if isinstance(bv, (int, float)) else None
    saved_cfg = ckpt.get("config", {}) or {}
    info["arch"] = {k: saved_cfg.get(k) for k in _ARCH_KEYS if k in saved_cfg}
    if not info["has_model"]:
        info["reason"] = "no non-empty model state dict"
        return info
    if info["step"] is None:
        info["reason"] = "no integer training step"
        return info
    info["valid"] = True
    info["reason"] = "ok"
    return info


def validate_checkpoint(path: str, cfg: Optional[ShnuConfig] = None,
                        expected_sha256: Optional[str] = None) -> Tuple[bool, dict]:
    """Full verification of a single checkpoint file. Fail-closed.

    Checks (each must pass): the payload is valid (see ``inspect_checkpoint``);
    the SHA-256 matches ``expected_sha256`` if given; the architecture matches
    ``cfg`` if given. Returns (ok, info) and never raises.
    """
    info = inspect_checkpoint(path, compute_sha=True)
    if not info["valid"]:
        return False, info
    if expected_sha256 is not None and info["sha256"] != expected_sha256:
        info["valid"] = False
        info["reason"] = (f"sha256 mismatch: {str(info['sha256'])[:16]}… "
                          f"!= expected {expected_sha256[:16]}…")
        return False, info
    if cfg is not None:
        try:
            import torch
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            ok, mism = verify_checkpoint_compatible(ckpt, cfg)
        except Exception as e:
            info["valid"] = False
            info["reason"] = f"re-load for arch check failed: {e!r}"
            return False, info
        if not ok:
            info["valid"] = False
            info["reason"] = f"architecture incompatible with config: {mism}"
            info["arch_mismatch"] = mism
            return False, info
        info["arch_compatible"] = True
    return True, info


def decide_replace(existing_path: Optional[str], incoming_path: str,
                   cfg: Optional[ShnuConfig] = None,
                   incoming_expected_sha256: Optional[str] = None) -> dict:
    """Decide whether ``incoming`` may take the place of ``existing``. Fail-closed.

    Rules (in order):
      1. If the incoming checkpoint does not verify, REJECT it — a valid or
         even a missing destination is never overwritten with something corrupt.
      2. If there is no existing (valid) destination, ADOPT the incoming one.
      3. If both are valid: ADOPT only when incoming.step > existing.step;
         otherwise KEEP the existing (never replace newer/equal with older).

    Returns {"action": "adopt"|"keep"|"reject", "reason", "existing_step",
    "incoming_step", ...}. This function decides only; it moves nothing.
    """
    inc_ok, inc = validate_checkpoint(incoming_path, cfg, incoming_expected_sha256)
    result = {
        "action": None, "reason": None,
        "incoming_step": inc.get("step"), "incoming_valid": inc_ok,
        "existing_step": None, "existing_valid": None,
        "incoming_info": inc,
    }
    if not inc_ok:
        result["action"] = "reject"
        result["reason"] = f"incoming checkpoint failed verification: {inc.get('reason')}"
        return result

    ex_ok, ex = (False, {"reason": "no existing checkpoint"})
    if existing_path and os.path.exists(existing_path):
        ex_ok, ex = validate_checkpoint(existing_path, cfg)
    result["existing_step"] = ex.get("step")
    result["existing_valid"] = ex_ok

    if not ex_ok:
        result["action"] = "adopt"
        result["reason"] = ("no valid existing checkpoint at destination "
                            f"({ex.get('reason')}); adopting verified incoming "
                            f"(step {inc.get('step')})")
        return result

    if (inc.get("step") or -1) > (ex.get("step") or -1):
        result["action"] = "adopt"
        result["reason"] = (f"incoming step {inc.get('step')} > existing step "
                            f"{ex.get('step')}; adopting newer")
    else:
        result["action"] = "keep"
        result["reason"] = (f"incoming step {inc.get('step')} <= existing step "
                            f"{ex.get('step')}; keeping existing (never older-over-newer)")
    return result


def choose_authoritative(paths: Iterable[str], cfg: Optional[ShnuConfig] = None) -> Tuple[Optional[str], dict]:
    """Given candidate checkpoint paths (possibly across locations), return the
    single authoritative one: the VALID checkpoint with the highest step.

    Invalid/corrupt candidates are ignored (never chosen). Returns
    (best_path_or_None, report). Answers the multi-session question
    "what is the newest valid checkpoint?" deterministically.
    """
    report = {"candidates": [], "chosen": None, "chosen_step": None}
    best_path, best_step = None, -1
    for p in paths:
        if not p:
            continue
        ok, info = validate_checkpoint(p, cfg)
        report["candidates"].append(
            {"path": p, "valid": ok, "step": info.get("step"), "reason": info.get("reason")}
        )
        if ok and (info.get("step") or -1) > best_step:
            best_path, best_step = p, int(info.get("step"))
    report["chosen"] = best_path
    report["chosen_step"] = best_step if best_path else None
    return best_path, report


# --------------------------------------------------------------------------
# Export / import bridge (for platforms that cannot mount the canonical store,
# e.g. Kaggle <-> Google Drive). Moves the checkpoint(s) + tokenizer + status as
# a single tar.gz carrying a manifest of SHA-256 + size for every file.
# --------------------------------------------------------------------------

_DEFAULT_CKPTS = ("checkpoint_latest.pt", "checkpoint_best.pt")


def build_bundle_manifest(out_dir: str, *, dataset: str = "wikitext103",
                          checkpoints: Iterable[str] = _DEFAULT_CKPTS,
                          include_tokenizer: bool = True,
                          include_status: bool = True) -> dict:
    """Describe the files that a run bundle would contain (no archive written).

    Records, per file: relative path inside the bundle, SHA-256, and byte size.
    For checkpoints it also records step + architecture. The manifest is what the
    receiving side verifies against, so nothing is adopted unhashed.
    """
    ck_dir = os.path.join(out_dir, "checkpoints")
    files = []

    def _add(abs_path, arcname):
        if os.path.exists(abs_path) and os.path.getsize(abs_path) > 0:
            files.append((abs_path, arcname))

    for name in checkpoints:
        _add(os.path.join(ck_dir, name), f"checkpoints/{name}")
    if include_tokenizer:
        _add(os.path.join(out_dir, "tokenizer", "shnu_bpe.json"), "tokenizer/shnu_bpe.json")
    if include_status:
        # status file lives at runs/<run>/training_status.json — record all runs' status
        runs_dir = os.path.join(out_dir, "runs")
        if os.path.isdir(runs_dir):
            for root, _dirs, fnames in os.walk(runs_dir):
                for fn in fnames:
                    if fn == "training_status.json":
                        ap = os.path.join(root, fn)
                        rel = os.path.relpath(ap, out_dir)
                        _add(ap, rel)

    manifest = {
        "bundle_schema_version": BUNDLE_SCHEMA_VERSION,
        "created_utc": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "dataset": dataset,
        "files": [],
    }
    for abs_path, arcname in files:
        entry = {
            "arcname": arcname,
            "sha256": sha256_file(abs_path),
            "size_bytes": os.path.getsize(abs_path),
        }
        if arcname.startswith("checkpoints/"):
            ci = inspect_checkpoint(abs_path, compute_sha=False)
            entry["step"] = ci.get("step")
            entry["arch"] = ci.get("arch")
            entry["checkpoint_valid"] = ci.get("valid")
        manifest["files"].append(entry)
    return manifest


def export_run_bundle(out_dir: str, archive_path: str, *, dataset: str = "wikitext103",
                      checkpoints: Iterable[str] = _DEFAULT_CKPTS,
                      include_tokenizer: bool = True, include_status: bool = True) -> dict:
    """Create ``archive_path`` (tar.gz) containing the run's checkpoint(s),
    tokenizer, and status, plus a MANIFEST.json of SHA-256 + size for each.

    Datasets (the large ``.npy`` streams) are deliberately **not** bundled — on a
    platform like Kaggle they are attached as a read-only Kaggle dataset. Returns
    the manifest. Never mutates ``out_dir``.
    """
    manifest = build_bundle_manifest(out_dir, dataset=dataset, checkpoints=checkpoints,
                                     include_tokenizer=include_tokenizer,
                                     include_status=include_status)
    if not manifest["files"]:
        raise FileNotFoundError(f"nothing to export from {out_dir} (no checkpoints/tokenizer/status found)")
    os.makedirs(os.path.dirname(os.path.abspath(archive_path)) or ".", exist_ok=True)
    tmp = archive_path + ".tmp"
    with tarfile.open(tmp, "w:gz") as tar:
        # manifest first
        mbytes = json.dumps(manifest, indent=2).encode()
        ti = tarfile.TarInfo(BUNDLE_MANIFEST_NAME)
        ti.size = len(mbytes)
        ti.mtime = int(time.time())
        import io
        tar.addfile(ti, io.BytesIO(mbytes))
        for entry in manifest["files"]:
            tar.add(os.path.join(out_dir, entry["arcname"]), arcname=entry["arcname"])
    os.replace(tmp, archive_path)
    manifest["archive_path"] = archive_path
    manifest["archive_sha256"] = sha256_file(archive_path)
    manifest["archive_size_bytes"] = os.path.getsize(archive_path)
    return manifest


def import_run_bundle(archive_path: str, out_dir: str, *, cfg: Optional[ShnuConfig] = None,
                      force: bool = False, verbose: bool = True) -> dict:
    """Verify and import a run bundle into ``out_dir``. Fail-closed.

    Steps: extract to a temp dir; verify every file's SHA-256 + size against the
    manifest; validate the incoming checkpoint(s); for each checkpoint apply
    ``decide_replace`` against any existing local checkpoint (never older-over-
    newer, never corrupt-over-valid) — unless ``force``. Verified files are moved
    into place with atomic ``os.replace``. Rejected checkpoints leave the local
    file untouched. Returns a detailed report; raises on manifest/hash failure.
    """
    report = {"archive": archive_path, "verified": [], "adopted": [], "kept": [],
              "rejected": [], "errors": []}
    if not os.path.exists(archive_path):
        raise FileNotFoundError(f"bundle not found: {archive_path}")

    tmp = tempfile.mkdtemp(prefix="shnu_bundle_")
    try:
        with tarfile.open(archive_path, "r:gz") as tar:
            _safe_extractall(tar, tmp)
        man_path = os.path.join(tmp, BUNDLE_MANIFEST_NAME)
        if not os.path.exists(man_path):
            raise ValueError("bundle has no MANIFEST.json — refusing to import unverifiable archive")
        manifest = json.load(open(man_path))

        # 1. verify every file against the manifest (hash + size) before anything moves
        for entry in manifest.get("files", []):
            ap = os.path.join(tmp, entry["arcname"])
            if not os.path.exists(ap):
                raise ValueError(f"manifest lists {entry['arcname']} but it is missing from the archive")
            size = os.path.getsize(ap)
            if size != entry["size_bytes"]:
                raise ValueError(f"size mismatch for {entry['arcname']}: {size} != {entry['size_bytes']}")
            digest = sha256_file(ap)
            if digest != entry["sha256"]:
                raise ValueError(f"SHA-256 mismatch for {entry['arcname']} — corrupt/tampered bundle")
            report["verified"].append(entry["arcname"])
        if verbose:
            print(f"[import] manifest verified: {len(report['verified'])} file(s) hash-checked")

        # 2. apply each file with the safety rules
        for entry in manifest.get("files", []):
            arc = entry["arcname"]
            src = os.path.join(tmp, arc)
            dst = os.path.join(out_dir, arc)
            os.makedirs(os.path.dirname(dst), exist_ok=True)

            if arc.startswith("checkpoints/"):
                decision = decide_replace(dst if os.path.exists(dst) else None, src, cfg,
                                          incoming_expected_sha256=entry["sha256"])
                if force and decision["action"] == "keep":
                    decision = {"action": "adopt", "reason": "force=True override of keep",
                                **{k: decision[k] for k in decision if k not in ("action", "reason")}}
                if decision["action"] == "adopt":
                    _atomic_place(src, dst)
                    report["adopted"].append({"arcname": arc, "reason": decision["reason"]})
                    if verbose:
                        print(f"[import] adopted {arc}: {decision['reason']}")
                elif decision["action"] == "keep":
                    report["kept"].append({"arcname": arc, "reason": decision["reason"]})
                    if verbose:
                        print(f"[import] kept existing {arc}: {decision['reason']}")
                else:  # reject
                    report["rejected"].append({"arcname": arc, "reason": decision["reason"]})
                    if verbose:
                        print(f"[import] REJECTED {arc}: {decision['reason']}")
            else:
                # tokenizer / status: replace only if absent or byte-identical is not required;
                # tokenizer is integrity-checked by hash already, status is a convenience view.
                if arc.startswith("tokenizer/") and os.path.exists(dst):
                    # never silently swap a tokenizer that differs — that would change the run
                    if sha256_file(dst) != entry["sha256"]:
                        report["rejected"].append(
                            {"arcname": arc, "reason": "local tokenizer differs from bundle — not overwriting"})
                        if verbose:
                            print(f"[import] REJECTED {arc}: local tokenizer differs from bundle")
                        continue
                _atomic_place(src, dst)
                report["adopted"].append({"arcname": arc, "reason": "verified non-checkpoint file placed"})
        return report
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _atomic_place(src: str, dst: str) -> None:
    """Move ``src`` onto ``dst`` atomically (copy to dst.tmp on same fs, then replace)."""
    tmp = dst + ".tmp"
    shutil.copy2(src, tmp)
    os.replace(tmp, dst)


def _safe_extractall(tar: tarfile.TarFile, dest: str) -> None:
    """Extract a tar safely — reject any member that escapes ``dest`` (path traversal)."""
    dest_abs = os.path.abspath(dest)
    for member in tar.getmembers():
        target = os.path.abspath(os.path.join(dest, member.name))
        if not (target == dest_abs or target.startswith(dest_abs + os.sep)):
            raise ValueError(f"unsafe path in archive: {member.name}")
        if member.issym() or member.islnk():
            raise ValueError(f"refusing symlink/hardlink member in archive: {member.name}")
    tar.extractall(dest)
