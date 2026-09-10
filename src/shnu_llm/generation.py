"""Text generation helper built on the model's own sampler."""
from __future__ import annotations

import torch


def sample_text(model, tokenizer, cfg, device, prompt, max_new_tokens=100,
                temperature=0.8, top_k=50, top_p=None, eos_id=None):
    ids = tokenizer.encode(prompt).ids
    if len(ids) == 0:
        ids = [tokenizer.token_to_id("<bos>")]
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    out = model.generate(idx, max_new_tokens=max_new_tokens, temperature=temperature,
                         top_k=top_k, top_p=top_p, eos_id=eos_id)
    return tokenizer.decode(out[0].tolist())
