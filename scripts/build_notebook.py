#!/usr/bin/env python3
"""Generate notebooks/SHNU_LLM_From_Scratch.ipynb — the reproducible Colab notebook.

The notebook clones this repository, installs dependencies, mounts Google Drive
for durable training state, and resumes training from the latest checkpoint. It
is designed to run from a completely fresh Colab runtime with no manual edits.
"""
import json
import os

REPO_URL = "https://github.com/mr-cri-spy/shnu-llm.git"


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip("\n").splitlines(keepends=True)}


cells = []

cells.append(md(
"""# SHNU-LLM — reproducible training on Colab

Clone the repository, install dependencies, mount Google Drive for durable state, and
resume training from the latest checkpoint. Runs from a **fresh** Colab runtime with no
manual edits.

**SHNU-LLM is initialized randomly and pretrained from scratch; pretrained LLM weights are not used.**

**Storage split**
- **GitHub** — source code, notebook, tests, configs, documentation, git history.
- **Google Drive** (`MyDrive/SHNU_LLM/`) — checkpoints, optimizer & scheduler state, tokenizer,
  datasets, tokenized caches, training logs, generated samples, exported model.
- **Colab** — temporary GPU compute only. Never keep permanent state in `/content`.

To resume in one step from a fresh runtime, jump to **Section 5** and run the single
`RESUME SHNU-LLM TRAINING` cell — it performs the whole chain by itself."""))

cells.append(md("## 1. Setup — clone repository & install dependencies"))
cells.append(code(f"""
REPO_URL = "{REPO_URL}"
REPO_DIR = "/content/shnu-llm"
import os, sys, subprocess
if not os.path.isdir(os.path.join(REPO_DIR, ".git")):
    subprocess.run(["git", "clone", REPO_URL, REPO_DIR], check=True)
else:
    subprocess.run(["git", "-C", REPO_DIR, "pull", "--ff-only"], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", os.path.join(REPO_DIR, "requirements.txt")])
if os.path.join(REPO_DIR, "src") not in sys.path:
    sys.path.insert(0, os.path.join(REPO_DIR, "src"))
import shnu_llm as S
COMMIT = subprocess.run(["git", "-C", REPO_DIR, "rev-parse", "--short", "HEAD"],
                        capture_output=True, text=True).stdout.strip()
print("shnu_llm", S.__version__, "| commit", COMMIT)
"""))

cells.append(md("## 2. Google Drive — durable training state\n"
                "Mounting Drive opens a Google authorization popup that **you** approve; the notebook "
                "cannot approve it for you."))
cells.append(code("""
from google.colab import drive
drive.mount("/content/drive")
DRIVE_ROOT = "/content/drive/MyDrive/SHNU_LLM"
for sub in ["checkpoints", "tokenizer", "datasets", "logs", "samples", "final", "evaluation"]:
    os.makedirs(os.path.join(DRIVE_ROOT, sub), exist_ok=True)
print("Drive root:", DRIVE_ROOT)
"""))

cells.append(md("## 3. Configuration & GPU detection"))
cells.append(code("""
import json, torch
cfg = S.ShnuConfig(**json.load(open(os.path.join(REPO_DIR, "configs", "t4.json"))))
cfg.out_dir = DRIVE_ROOT                      # all state lives on Drive, not /content
device = "cuda" if torch.cuda.is_available() else "cpu"
print("environment:", json.dumps(S.env_report(device)))
"""))

cells.append(md("## 4. Corpus, tokenizer & tokenized dataset (cached on Drive)\n"
                "WikiText-103 (CC BY-SA 3.0) with a public-domain Tiny Shakespeare fallback. The trained "
                "tokenizer and the tokenized stream are cached on Drive so later runs skip re-tokenizing."))
cells.append(code("""
import numpy as np

def load_corpus(max_chars=80_000_000):
    try:
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train", streaming=True)
        buf, total = [], 0
        for ex in ds:
            t = ex["text"]
            if t and t.strip():
                buf.append(t); total += len(t)
            if total >= max_chars:
                break
        return buf, {"source": "wikitext-103-raw-v1", "license": "CC BY-SA 3.0", "chars": total}
    except Exception as e:
        print("[dataset] WikiText unavailable:", repr(e), "-> Tiny Shakespeare fallback")
        import urllib.request
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        txt = urllib.request.urlopen(url, timeout=60).read().decode("utf-8")
        return txt.split("\\n\\n"), {"source": "tiny-shakespeare", "license": "public domain", "chars": len(txt)}

tok_path = os.path.join(DRIVE_ROOT, "tokenizer", "shnu_bpe.json")
records, DATASET_INFO = load_corpus()
records, _ = S.clean_records(records, min_chars=32)
if os.path.exists(tok_path):
    tok = S.load_tokenizer(tok_path); print("tokenizer: loaded from Drive")
else:
    tok = S.train_tokenizer(iter(records[:200_000]), cfg.vocab_size, tok_path); print("tokenizer: trained & cached on Drive")
cfg.vocab_size = tok.get_vocab_size(); eos_id = tok.token_to_id("<eos>")
cache = os.path.join(DRIVE_ROOT, "datasets", f"stream_vocab{cfg.vocab_size}.npy")
if os.path.exists(cache):
    stream = np.load(cache); print("tokenized stream: loaded from Drive")
else:
    stream = S.build_token_stream(tok, records, eos_id); np.save(cache, stream); print("tokenized stream: built & cached on Drive")
train_ids, val_ids = S.train_val_split(stream, val_ratio=0.02)
datasets = {"train": S.PackedDataset(train_ids, cfg.block_size, device),
            "val": S.PackedDataset(val_ids, cfg.block_size, device)}
print("dataset:", json.dumps(DATASET_INFO), "| tokens:", len(stream))
"""))

cells.append(md("## 5. ▶️ RESUME SHNU-LLM TRAINING\n"
                "**One self-contained cell.** From a fresh runtime it performs the whole chain: "
                "GitHub pull → Drive mount → environment setup → checkpoint discovery → compatibility "
                "verification → model loading → resume from the exact saved step. No manual step numbers."))
cells.append(code("""
# ============================ RESUME SHNU-LLM TRAINING ============================
# Self-contained: safe to run as the ONLY cell in a fresh Colab runtime.
import os, sys, json, subprocess

REPO_URL = \"""" + REPO_URL + """\"
REPO_DIR = "/content/shnu-llm"
if not os.path.isdir(os.path.join(REPO_DIR, ".git")):
    subprocess.run(["git", "clone", REPO_URL, REPO_DIR], check=True)
else:
    subprocess.run(["git", "-C", REPO_DIR, "pull", "--ff-only"], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", os.path.join(REPO_DIR, "requirements.txt")])
if os.path.join(REPO_DIR, "src") not in sys.path:
    sys.path.insert(0, os.path.join(REPO_DIR, "src"))
import shnu_llm as S, torch, numpy as np

from google.colab import drive
drive.mount("/content/drive")
DRIVE_ROOT = "/content/drive/MyDrive/SHNU_LLM"
for sub in ["checkpoints", "tokenizer", "datasets", "logs", "samples", "final"]:
    os.makedirs(os.path.join(DRIVE_ROOT, sub), exist_ok=True)

cfg = S.ShnuConfig(**json.load(open(os.path.join(REPO_DIR, "configs", "t4.json"))))
cfg.out_dir = DRIVE_ROOT
device = "cuda" if torch.cuda.is_available() else "cpu"
print("commit:", subprocess.run(["git","-C",REPO_DIR,"rev-parse","--short","HEAD"],capture_output=True,text=True).stdout.strip(),
      "| device:", json.dumps(S.env_report(device)))

# corpus + tokenizer (reuse Drive caches)
def _load_corpus(max_chars=80_000_000):
    try:
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train", streaming=True)
        buf, total = [], 0
        for ex in ds:
            t = ex["text"]
            if t and t.strip():
                buf.append(t); total += len(t)
            if total >= max_chars: break
        return buf
    except Exception:
        import urllib.request
        u = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        return urllib.request.urlopen(u, timeout=60).read().decode("utf-8").split("\\n\\n")

tok_path = os.path.join(DRIVE_ROOT, "tokenizer", "shnu_bpe.json")
records, _ = S.clean_records(_load_corpus(), min_chars=32)
tok = S.load_tokenizer(tok_path) if os.path.exists(tok_path) else S.train_tokenizer(iter(records[:200_000]), cfg.vocab_size, tok_path)
cfg.vocab_size = tok.get_vocab_size(); eos_id = tok.token_to_id("<eos>")
cache = os.path.join(DRIVE_ROOT, "datasets", f"stream_vocab{cfg.vocab_size}.npy")
stream = np.load(cache) if os.path.exists(cache) else S.build_token_stream(tok, records, eos_id)
if not os.path.exists(cache): np.save(cache, stream)
train_ids, val_ids = S.train_val_split(stream, val_ratio=0.02)
datasets = {"train": S.PackedDataset(train_ids, cfg.block_size, device),
            "val": S.PackedDataset(val_ids, cfg.block_size, device)}

# checkpoint discovery + verification + resume from the exact step
ckpt_dir = os.path.join(DRIVE_ROOT, "checkpoints")
latest = S.find_latest_checkpoint(ckpt_dir)
if latest is None:
    print("no checkpoint found -> starting a fresh run (step 0)")
else:
    ck = torch.load(latest, map_location=device, weights_only=False)
    ok, mism = S.verify_checkpoint_compatible(ck, cfg)
    assert ok, f"checkpoint architecture mismatch: {mism}"
    print(f"resuming from {os.path.basename(latest)} at step {S.checkpoint_step(latest)} (verified compatible)")

S.set_seed(cfg.seed)
model = S.ShnuLM(cfg).to(device)
result = S.train(model, datasets, cfg, device, resume_from=latest,
                 tokenizer=tok, eos_id=eos_id, sample_prompt="The ", verbose=True)
print(f"started at step {result['start_step']} -> finished at step {result['steps']}")

# export to Drive
final = os.path.join(DRIVE_ROOT, "final")
os.makedirs(os.path.join(final, "tokenizer"), exist_ok=True)
torch.save({"model": model.state_dict(), "config": S.config.asdict(cfg), "version": cfg.version},
           os.path.join(final, "shnu_llm_v0.1.pt"))
open(os.path.join(final, "config.json"), "w").write(cfg.to_json())
import shutil; shutil.copy(tok_path, os.path.join(final, "tokenizer", "shnu_bpe.json"))
print("exported model to", final)
# =================================================================================
"""))

cells.append(md("## 6. Evaluate & generate"))
cells.append(code("""
import math
metrics_val = result["final_val_loss"]
print(f"final val loss = {metrics_val:.4f} | perplexity = {math.exp(metrics_val):.2f}")
for p in ["The ", "In the ", "History of "]:
    print(p, "->", repr(S.sample_text(model, tok, cfg, device, p, 60, temperature=0.8, top_k=50, eos_id=eos_id)))
"""))

cells.append(md("## 7. Pre-training safety checklist"))
cells.append(code("""
checks = {
    "repository_url": REPO_URL,
    "git_commit": subprocess.run(["git","-C",REPO_DIR,"rev-parse","HEAD"],capture_output=True,text=True).stdout.strip(),
    "drive_root": DRIVE_ROOT,
    "checkpoint_dir": os.path.join(DRIVE_ROOT, "checkpoints"),
    "dataset": DATASET_INFO if "DATASET_INFO" in dir() else "(see Section 4)",
    "tokenizer_vocab": cfg.vocab_size,
    "architecture": {"n_layers": cfg.n_layers, "n_heads": cfg.n_heads, "d_model": cfg.d_model,
                     "block_size": cfg.block_size, "params": model.num_params()},
    "pretrained_weights": "NONE (random initialization)",
    "paid_resources": "NONE (free Colab / open-source software only)",
}
print(json.dumps(checks, indent=2, default=str))
"""))

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "name": "python3"},
                   "language_info": {"name": "python"},
                   "accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"}},
      "nbformat": 4, "nbformat_minor": 0}

out = os.path.join(os.path.dirname(__file__), "..", "notebooks", "SHNU_LLM_From_Scratch.ipynb")
json.dump(nb, open(out, "w"), indent=1)
print("wrote", os.path.abspath(out), "with", len(cells), "cells")
