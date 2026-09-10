"""Byte-level BPE tokenizer trained from scratch on the project corpus."""
from __future__ import annotations

import os


SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>"]


def train_tokenizer(text_iter, vocab_size: int, save_path: str):
    """Train a byte-level BPE tokenizer from scratch and save it."""
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        show_progress=False,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tok.train_from_iterator(text_iter, trainer=trainer)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    tok.save(save_path)
    return tok


def load_tokenizer(path: str):
    from tokenizers import Tokenizer
    return Tokenizer.from_file(path)


def encode(tok, text: str):
    return tok.encode(text).ids


def decode(tok, ids):
    return tok.decode(ids)
