"""Reproducibility and environment utilities."""
from __future__ import annotations

import random
import platform

import numpy as np
import torch

from .config import ShnuConfig


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_dtype(cfg: ShnuConfig, device: str) -> torch.dtype:
    if cfg.dtype == "float32":
        return torch.float32
    if cfg.dtype == "bfloat16":
        return torch.bfloat16
    if cfg.dtype == "float16":
        return torch.float16
    # auto
    if device == "cuda":
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32


def env_report(device: str) -> dict:
    import platform
    rep = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": device,
    }
    if torch.cuda.is_available():
        rep["gpu_name"] = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        rep["gpu_memory_gb"] = round(props.total_memory / 1e9, 2)
    return rep
