"""Platform abstraction for SHNU-LLM multi-platform training.

The v0.2 run continues on more than one free-GPU service. The *training* is
identical everywhere — same model, same checkpoint format, same
``resume_training`` path — only the **environment** differs:

  * where durable state lives and how it is reached (Colab mounts Google Drive;
    Kaggle uses a local working dir + an explicit export/import bridge);
  * whether to insist on a T4 specifically or accept any CUDA GPU;
  * how much free disk to require.

This module captures exactly those differences in one small ``Platform`` value
so the launchers stay thin and share a single code path. It adds **no** second
resume system and **no** second checkpoint format.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class Platform:
    """Environment-specific settings for one free-GPU service."""
    name: str
    require_t4: bool           # insist the GPU is a T4 (Colab), or accept any CUDA GPU
    uses_drive_mount: bool     # True if the canonical store is a mounted Google Drive
    default_out_dir: str       # default durable run root on this platform
    default_drive_root: str    # dir that must exist to prove the store is reachable
    min_free_gb: float
    notes: str = ""


# Colab: our primary platform. Drive is mounted natively; a T4 is expected.
COLAB = Platform(
    name="colab",
    require_t4=True,
    uses_drive_mount=True,
    default_out_dir="/content/drive/MyDrive/SHNU_LLM",
    default_drive_root="/content/drive/MyDrive",
    min_free_gb=3.0,
    notes="Google Drive mounted at /content/drive; checkpoints live on Drive.",
)

# Kaggle: secondary platform. No Drive mount. Durable state is a local working
# dir populated by a verified import bundle; GPU may be a T4 (2x) or a P100, so
# we accept any CUDA GPU but still FAIL CLOSED on no GPU.
KAGGLE = Platform(
    name="kaggle",
    require_t4=False,
    uses_drive_mount=False,
    default_out_dir="/kaggle/working/SHNU_LLM",
    default_drive_root="/kaggle/working",
    min_free_gb=3.0,
    notes=("No Google Drive mount. Checkpoints move via the verified "
           "export/import bundle (see checkpoint_safety). Accepts any CUDA GPU."),
)

# Generic: any other legitimate free GPU host. Accept any CUDA GPU; the caller
# provides the durable path. Still fail-closed on no GPU.
GENERIC = Platform(
    name="generic",
    require_t4=False,
    uses_drive_mount=False,
    default_out_dir=os.path.join(os.getcwd(), "SHNU_LLM"),
    default_drive_root=os.getcwd(),
    min_free_gb=3.0,
    notes="Any legitimate free CUDA GPU host; durable path supplied by caller.",
)

_PLATFORMS = {p.name: p for p in (COLAB, KAGGLE, GENERIC)}


def get_platform(name: str) -> Platform:
    """Look up a platform by name (colab|kaggle|generic)."""
    key = (name or "").strip().lower()
    if key not in _PLATFORMS:
        raise ValueError(f"unknown platform {name!r}; known: {sorted(_PLATFORMS)}")
    return _PLATFORMS[key]


def detect_platform() -> Platform:
    """Best-effort detection of the current free-GPU host.

    Kaggle sets ``KAGGLE_KERNEL_RUN_TYPE`` and has ``/kaggle/working``; Colab has
    the ``google.colab`` module and ``/content``. Falls back to GENERIC. Detection
    only picks sensible defaults — every setting can still be overridden per call.
    """
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or os.path.isdir("/kaggle/working"):
        return KAGGLE
    if os.path.isdir("/content"):
        return COLAB
    try:
        import google.colab  # noqa: F401
        return COLAB
    except Exception:
        pass
    return GENERIC


def describe_gpu() -> dict:
    """Non-raising GPU description for preflight/benchmark logging.

    Reports what is present without deciding policy — the fail-closed decision is
    made by ``preflight.require_gpu``. Returns ``available=False`` when there is
    no CUDA device (so a launcher can refuse to train on CPU).
    """
    info = {"available": False, "name": None, "vram_gb": None, "is_t4": False,
            "cuda": None, "device_count": 0}
    try:
        import torch
        info["cuda"] = getattr(torch.version, "cuda", None)
        if torch.cuda.is_available():
            info["available"] = True
            info["device_count"] = torch.cuda.device_count()
            name = torch.cuda.get_device_name(0)
            info["name"] = name
            info["is_t4"] = "T4" in name
            info["vram_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2)
    except Exception as e:
        info["error"] = repr(e)
    return info
