"""SHNU-LLM: a decoder-only Transformer pretrained from scratch.

SHNU-LLM is initialized randomly and pretrained from scratch; pretrained LLM
weights are not used.
"""
from .config import ShnuConfig
from .utils import set_seed, resolve_dtype, env_report
from .data import (
    clean_text, clean_records, build_token_stream, train_val_split, PackedDataset,
)
from .tokenizer import (
    SPECIAL_TOKENS, train_tokenizer, load_tokenizer, encode, decode,
)
from .model import ShnuLM
from .training import (
    get_lr, configure_optimizer, estimate_loss, save_checkpoint, load_checkpoint,
    train, prune_checkpoints,
)
from .generation import sample_text
from .persistence import (
    find_latest_checkpoint, verify_checkpoint_compatible, checkpoint_step,
)
from .evaluation import (
    validation_perplexity, repetition_metrics, context_length_test, evaluate_generation,
)
from .preflight import (
    PreflightError, run_preflight, require_gpu, require_drive_mounted, require_free_disk,
    verify_dataset, verify_tokenizer, verify_checkpoint,
)
from .status import (
    TrainingStatus, tokens_per_step, current_git_commit,
    VALID_STATES, STATUS_SCHEMA_VERSION, DEFAULT_RUN_NAME,
)

__version__ = "0.1.0"
__all__ = [
    "ShnuConfig", "set_seed", "resolve_dtype", "env_report",
    "clean_text", "clean_records", "build_token_stream", "train_val_split", "PackedDataset",
    "SPECIAL_TOKENS", "train_tokenizer", "load_tokenizer", "encode", "decode",
    "ShnuLM", "get_lr", "configure_optimizer", "estimate_loss",
    "save_checkpoint", "load_checkpoint", "train", "prune_checkpoints", "sample_text",
    "find_latest_checkpoint", "verify_checkpoint_compatible", "checkpoint_step",
    "validation_perplexity", "repetition_metrics", "context_length_test", "evaluate_generation",
    "PreflightError", "run_preflight", "require_gpu", "require_drive_mounted",
    "require_free_disk", "verify_dataset", "verify_tokenizer", "verify_checkpoint",
    "TrainingStatus", "tokens_per_step", "current_git_commit",
    "VALID_STATES", "STATUS_SCHEMA_VERSION", "DEFAULT_RUN_NAME",
]
