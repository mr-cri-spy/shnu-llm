"""Model and training configuration for SHNU-LLM."""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from typing import Optional


@dataclass
class ShnuConfig:
    # --- identity ---
    name: str = "SHNU-LLM"
    version: str = "v0.1"

    # --- tokenizer ---
    vocab_size: int = 16000          # trained BPE vocab (incl. special tokens)

    # --- model architecture ---
    n_layers: int = 8
    n_heads: int = 8
    n_kv_heads: Optional[int] = None  # None => == n_heads (full MHA); else GQA
    d_model: int = 512
    d_ff: Optional[int] = None        # None => derived SwiGLU width
    block_size: int = 256             # context length (max sequence length)
    rope_theta: float = 10000.0
    dropout: float = 0.0
    tie_embeddings: bool = True

    # --- training ---
    batch_size: int = 32              # micro-batch (per optimizer step slice)
    grad_accum_steps: int = 1
    max_steps: int = 2000
    warmup_steps: int = 100
    lr: float = 3e-4
    min_lr_ratio: float = 0.1         # cosine floor = lr * ratio
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    eval_interval: int = 200
    eval_iters: int = 50
    sample_interval: int = 500
    checkpoint_interval: int = 500
    log_interval: int = 20
    seed: int = 1337

    # --- runtime / io ---
    out_dir: str = "SHNU_LLM"
    dtype: str = "auto"               # "auto"|"float32"|"bfloat16"|"float16"
    compile_model: bool = False

    def __post_init__(self):
        assert self.d_model % self.n_heads == 0, "d_model must divide n_heads"
        if self.n_kv_heads is None:
            self.n_kv_heads = self.n_heads
        assert self.n_heads % self.n_kv_heads == 0, "n_heads must divide n_kv_heads"
        if self.d_ff is None:
            # SwiGLU: keep param-equivalent to 4*d_model MLP => ~ (8/3)*d_model,
            # rounded to a multiple of 64.
            hidden = int(8 / 3 * self.d_model)
            self.d_ff = ((hidden + 63) // 64) * 64

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)
