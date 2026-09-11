"""Persistent, crash-safe training-status monitor for SHNU-LLM.

This is a **side-car** around the existing checkpoint/resume system (see
``persistence.py`` and ``training.py``); it does not implement a second
checkpoint or resume mechanism. It maintains a small machine-readable JSON file
on durable storage (Google Drive) so a human — or a fresh Colab runtime — can see
what a long, multi-session training run is doing without loading the model.

Design rules (mandatory):

* **The checkpoint is authoritative.** On a fresh runtime the newest valid
  checkpoint is the source of truth for ``current_step``; the status JSON is only
  a convenience view. If the two disagree, the checkpoint wins and the status is
  repaired from it. A valid checkpoint is **never** reset to step 0 because the
  status file is stale, missing, or corrupted.
* **Atomic writes.** The file is written to a ``*.tmp`` sibling and then
  ``os.replace``-d into place, so a Colab disconnect mid-write can never leave
  corrupted JSON behind.
* **Infrequent writes.** Status is updated at meaningful events (start, resume,
  eval, checkpoint, interruption, failure, completion) — never every batch — to
  avoid hammering Google Drive I/O.

A disconnected Colab runtime does **not** continue GPU training; this file only
reflects the last durable state written before the disconnect.
"""
from __future__ import annotations

import os
import json
import math
import time
import tempfile
import datetime as _dt
from typing import Optional

from .config import ShnuConfig
from .persistence import find_latest_checkpoint, verify_checkpoint_compatible, checkpoint_step

STATUS_SCHEMA_VERSION = 1
DEFAULT_RUN_NAME = "shnu_v0.2"
STATUS_FILENAME = "training_status.json"

VALID_STATES = ("active", "paused", "interrupted", "completed", "failed")


def _utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def tokens_per_step(cfg: ShnuConfig) -> int:
    """Tokens consumed per optimizer step = micro_batch × grad_accum × block_size."""
    return cfg.batch_size * cfg.grad_accum_steps * cfg.block_size


def current_git_commit(repo_dir: str) -> Optional[str]:
    """Best-effort short-and-long HEAD commit for the repo, or None if unavailable."""
    import subprocess
    try:
        out = subprocess.run(["git", "-C", repo_dir, "rev-parse", "HEAD"],
                              capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or None
    except Exception:
        return None


class TrainingStatus:
    """Reads/writes/repairs the durable training-status JSON for one run."""

    def __init__(self, status_path: str, checkpoints_dir: str, run_name: str = DEFAULT_RUN_NAME):
        self.status_path = status_path
        self.checkpoints_dir = checkpoints_dir
        self.run_name = run_name

    # ---- construction helpers ---------------------------------------------
    @classmethod
    def for_run(cls, out_dir: str, run_name: str = DEFAULT_RUN_NAME) -> "TrainingStatus":
        """Build the manager from the durable run root (e.g. Drive SHNU_LLM dir).

        Status file:   <out_dir>/runs/<run_name>/training_status.json
        Checkpoints:   <out_dir>/checkpoints    (the established checkpoint dir)
        """
        status_path = os.path.join(out_dir, "runs", run_name, STATUS_FILENAME)
        checkpoints_dir = os.path.join(out_dir, "checkpoints")
        return cls(status_path, checkpoints_dir, run_name)

    # ---- schema -----------------------------------------------------------
    @staticmethod
    def default(cfg: ShnuConfig, git_commit: Optional[str] = None) -> dict:
        tgt_steps = int(cfg.max_steps)
        tps = tokens_per_step(cfg)
        return {
            "schema_version": STATUS_SCHEMA_VERSION,
            "project": cfg.name,
            "version": cfg.version,
            "run_name": DEFAULT_RUN_NAME,
            "config": {
                "vocab_size": cfg.vocab_size, "n_layers": cfg.n_layers, "n_heads": cfg.n_heads,
                "d_model": cfg.d_model, "block_size": cfg.block_size, "batch_size": cfg.batch_size,
                "grad_accum_steps": cfg.grad_accum_steps, "lr": cfg.lr, "max_steps": cfg.max_steps,
                "tokens_per_optimizer_step": tps,
            },
            "git_commit": git_commit,
            "state": "active",
            "current_step": 0,
            "target_steps": tgt_steps,
            "tokens_processed": 0,
            "target_tokens": tgt_steps * tps,
            "completion_percent": 0.0,
            "train_loss": None,
            "validation_loss": None,
            "perplexity": None,
            "best_validation_loss": None,
            "best_validation_step": None,
            "latest_checkpoint": None,
            "best_checkpoint": None,
            "checkpoint_timestamp": None,
            "session_count": 0,
            "resume_count": 0,
            "elapsed_seconds": 0.0,
            "throughput_tokens_per_second": None,
            "device": None,
            "gpu_name": None,
            "gpu_memory_gb": None,
            "last_update_timestamp": None,
            "last_error": None,
            "reconciliation": {"occurred": False, "reason": None,
                               "checkpoint_step": None, "status_step_before": None},
        }

    # ---- io ---------------------------------------------------------------
    def load_or_none(self) -> Optional[dict]:
        """Return the parsed status, or None if it is missing or corrupt.

        A corrupt file is treated as missing — never fatal — so a bad write from a
        previous disconnect can be repaired rather than crashing recovery.
        """
        if not os.path.exists(self.status_path):
            return None
        try:
            with open(self.status_path, "r") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else None
        except (json.JSONDecodeError, ValueError, OSError):
            return None

    def write(self, data: dict) -> str:
        """Atomically write the status JSON (tmp file + os.replace)."""
        os.makedirs(os.path.dirname(self.status_path), exist_ok=True)
        data = dict(data)
        data["last_update_timestamp"] = _utc_now()
        fd, tmp = tempfile.mkstemp(dir=os.path.dirname(self.status_path), suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(data, f, indent=2, default=str)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, self.status_path)   # atomic on POSIX
        finally:
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except OSError:
                    pass
        return self.status_path

    # ---- derived fields ---------------------------------------------------
    @staticmethod
    def _recompute_derived(data: dict) -> dict:
        tps = data.get("config", {}).get("tokens_per_optimizer_step") or 0
        step = data.get("current_step") or 0
        tgt = data.get("target_steps") or 0
        data["tokens_processed"] = int(step) * int(tps)
        data["target_tokens"] = int(tgt) * int(tps)
        data["completion_percent"] = round(100.0 * step / tgt, 4) if tgt else 0.0
        vl = data.get("validation_loss")
        if vl is not None and isinstance(vl, (int, float)) and math.isfinite(vl):
            data["perplexity"] = round(math.exp(vl), 4)
        return data

    def update(self, cfg: Optional[ShnuConfig] = None, **fields) -> dict:
        """Load (or default), merge ``fields``, recompute derived values, write atomically."""
        data = self.load_or_none()
        if data is None:
            if cfg is None:
                raise ValueError("cannot create a fresh status without a cfg")
            data = self.default(cfg)
        if "state" in fields and fields["state"] not in VALID_STATES:
            raise ValueError(f"invalid state {fields['state']!r}; allowed: {VALID_STATES}")
        data.update(fields)
        data = self._recompute_derived(data)
        self.write(data)
        return data

    # ---- reconciliation (checkpoint is authoritative) ---------------------
    def reconcile(self, cfg: ShnuConfig, *, git_commit: Optional[str] = None,
                  device: Optional[str] = None, gpu_name: Optional[str] = None,
                  gpu_memory_gb: Optional[float] = None) -> dict:
        """Reconcile the status JSON against the newest valid checkpoint.

        The checkpoint always wins. Returns the reconciled status dict (also written
        to disk). Never resets a valid (step > 0) checkpointed run to step 0.
        """
        status = self.load_or_none()
        data = status if status is not None else self.default(cfg, git_commit)
        if git_commit is not None:
            data["git_commit"] = git_commit
        for k, v in (("device", device), ("gpu_name", gpu_name), ("gpu_memory_gb", gpu_memory_gb)):
            if v is not None:
                data[k] = v

        latest = find_latest_checkpoint(self.checkpoints_dir)
        recon = {"occurred": False, "reason": None, "checkpoint_step": None,
                 "status_step_before": (status or {}).get("current_step"),
                 "checkpoint_found": latest is not None,
                 "timestamp": _utc_now()}

        if latest is None:
            # No checkpoint => there is no valid run to protect; this is a fresh run.
            if status is not None and (status.get("current_step") or 0) > 0:
                recon["occurred"] = True
                recon["reason"] = ("status claims current_step > 0 but no checkpoint exists; "
                                   "cannot verify progress — starting fresh (no valid checkpoint to resume)")
                data["last_error"] = "status/checkpoint mismatch: status had progress but no checkpoint found"
            else:
                recon["reason"] = "fresh run: no checkpoint present"
            data["current_step"] = 0
            data["latest_checkpoint"] = None
            data["fresh_run"] = True
            data["reconciliation"] = recon
            data = self._recompute_derived(data)
            self.write(data)
            return data

        # A checkpoint exists -> it is authoritative.
        ck_step = checkpoint_step(latest)
        try:
            import torch
            ckpt = torch.load(latest, map_location="cpu", weights_only=False)
        except Exception as e:
            # cannot read newest checkpoint; record but do NOT wipe status progress
            recon["occurred"] = True
            recon["reason"] = f"failed to read newest checkpoint {os.path.basename(latest)}: {e!r}"
            data["last_error"] = recon["reason"]
            data["state"] = "failed"
            data["reconciliation"] = recon
            data = self._recompute_derived(data)
            self.write(data)
            return data

        ok, mism = verify_checkpoint_compatible(ckpt, cfg)
        ck_best = ckpt.get("best_val")
        status_step = (status or {}).get("current_step")
        recon["checkpoint_step"] = ck_step

        if status is None:
            recon["occurred"] = True
            recon["reason"] = "status missing or corrupt -> repaired from checkpoint"
        elif status_step != ck_step:
            recon["occurred"] = True
            recon["reason"] = (f"status step {status_step} != checkpoint step {ck_step} "
                               f"-> trusting checkpoint (authoritative)")
        else:
            recon["reason"] = "status agrees with checkpoint"

        # repair authoritative fields from the checkpoint
        data["current_step"] = ck_step
        data["latest_checkpoint"] = latest
        data["checkpoint_timestamp"] = _utc_now() if not os.path.exists(latest) else \
            _dt.datetime.fromtimestamp(os.path.getmtime(latest), _dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        data["fresh_run"] = False
        if ck_best is not None and isinstance(ck_best, (int, float)) and math.isfinite(ck_best):
            data["best_validation_loss"] = ck_best
        # best checkpoint (if present) gives best_validation_step
        best_path = os.path.join(self.checkpoints_dir, "checkpoint_best.pt")
        if os.path.exists(best_path):
            data["best_checkpoint"] = best_path
            data["best_validation_step"] = checkpoint_step(best_path)
        data["checkpoint_compatible"] = ok
        if not ok:
            data["state"] = "failed"
            data["last_error"] = f"checkpoint architecture incompatible with config: {mism}"
        elif ck_step >= (data.get("target_steps") or cfg.max_steps):
            data["state"] = "completed"
        data["reconciliation"] = recon
        data = self._recompute_derived(data)
        self.write(data)
        return data

    # ---- rendering --------------------------------------------------------
    def render(self, data: Optional[dict] = None) -> str:
        d = data if data is not None else self.load_or_none()
        if d is None:
            return (f"SHNU-LLM {DEFAULT_RUN_NAME} STATUS\n\n"
                    f"No status file found at {self.status_path}\n"
                    f"(run has not started, or Drive is not mounted).")

        def g(k, default="—"):
            v = d.get(k)
            return default if v is None else v

        recon = d.get("reconciliation") or {}
        lines = [
            f"SHNU-LLM {d.get('version', '')} STATUS",
            "",
            f"State:              {g('state')}",
            f"Git commit:         {g('git_commit')}",
            f"Step:               {g('current_step')} / {g('target_steps')}",
            f"Progress:           {g('completion_percent')}%",
            f"Tokens processed:   {g('tokens_processed'):,}" if isinstance(d.get('tokens_processed'), int)
            else f"Tokens processed:   {g('tokens_processed')}",
            f"Target tokens:      {g('target_tokens'):,}" if isinstance(d.get('target_tokens'), int)
            else f"Target tokens:      {g('target_tokens')}",
            f"Train loss:         {g('train_loss')}",
            f"Validation loss:    {g('validation_loss')}",
            f"Perplexity:         {g('perplexity')}",
            f"Best val loss:      {g('best_validation_loss')} @ step {g('best_validation_step')}",
            f"Latest checkpoint:  {g('latest_checkpoint')}",
            f"Best checkpoint:    {g('best_checkpoint')}",
            f"Session count:      {g('session_count')}",
            f"Resume count:       {g('resume_count')}",
            f"Elapsed seconds:    {g('elapsed_seconds')}",
            f"GPU:                {g('gpu_name')} ({g('device')})",
            f"Throughput:         {g('throughput_tokens_per_second')} tok/s",
            f"Last update:        {g('last_update_timestamp')}",
            f"Last error:         {g('last_error')}",
            f"Reconciliation:     occurred={recon.get('occurred')} "
            f"(ckpt_step={recon.get('checkpoint_step')}, status_before={recon.get('status_step_before')})",
        ]
        if recon.get("reason"):
            lines.append(f"  reason: {recon.get('reason')}")
        return "\n".join(lines)
