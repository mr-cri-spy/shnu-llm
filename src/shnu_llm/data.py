"""Dataset cleaning, tokenized packing, and train/val splitting."""
from __future__ import annotations

import numpy as np
import torch


def clean_text(text: str) -> str:
    """Light normalization suitable for LM pretraining."""
    # normalize newlines, strip carriage returns and null bytes
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
    return text


def clean_records(records, min_chars: int = 1):
    """Filter/clean an iterable of raw text records.

    Removes empty / whitespace-only / too-short records and exact duplicates.
    Returns (kept_list, stats_dict).
    """
    seen = set()
    kept = []
    n_in = n_empty = n_short = n_dup = 0
    for r in records:
        n_in += 1
        if not r or not r.strip():
            n_empty += 1
            continue
        r = clean_text(r)
        if len(r.strip()) < min_chars:
            n_short += 1
            continue
        h = hash(r)
        if h in seen:
            n_dup += 1
            continue
        seen.add(h)
        kept.append(r)
    stats = dict(records_in=n_in, kept=len(kept),
                 removed_empty=n_empty, removed_short=n_short, removed_dup=n_dup)
    return kept, stats


def build_token_stream(tok, records, eos_id: int):
    """Encode records and concatenate with an EOS between documents."""
    ids = []
    for r in records:
        ids.extend(tok.encode(r).ids)
        ids.append(eos_id)
    return np.array(ids, dtype=np.uint16 if tok.get_vocab_size() < 65536 else np.uint32)


def train_val_split(token_ids: np.ndarray, val_ratio: float = 0.05):
    """Split the token stream into train/val with NO overlap (no leakage)."""
    n_val = int(len(token_ids) * val_ratio)
    train_ids = token_ids[:-n_val]
    val_ids = token_ids[-n_val:]
    return train_ids, val_ids


class PackedDataset:
    """Serves contiguous (block_size+1) windows for causal LM training."""
    def __init__(self, token_ids: np.ndarray, block_size: int, device: str):
        self.data = torch.from_numpy(token_ids.astype(np.int64))
        self.block_size = block_size
        self.device = device

    def __len__(self):
        return len(self.data) - self.block_size - 1

    def get_batch(self, batch_size: int, generator: torch.Generator):
        ix = torch.randint(len(self), (batch_size,), generator=generator)
        x = torch.stack([self.data[i:i + self.block_size] for i in ix])
        y = torch.stack([self.data[i + 1:i + 1 + self.block_size] for i in ix])
        if self.device == "cuda":
            x = x.pin_memory().to(self.device, non_blocking=True)
            y = y.pin_memory().to(self.device, non_blocking=True)
        else:
            x, y = x.to(self.device), y.to(self.device)
        return x, y
