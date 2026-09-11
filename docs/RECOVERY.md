# SHNU-LLM — Fresh-Runtime Recovery Procedure

Free Colab runtimes are ephemeral: the GPU and local disk vanish when the session
times out or disconnects. SHNU-LLM treats Colab as **disposable compute** and Google
Drive as the **single source of durable state**. A new runtime can therefore pick up an
interrupted run exactly where it stopped — no manual step editing, no accidental
restart.

## What is durable vs. ephemeral

| Lives on Google Drive (durable) | Lives only in the runtime (ephemeral) |
|---|---|
| Tokenizer (`tokenizer/shnu_bpe.json`) | The GPU and CUDA context |
| Tokenized streams (`datasets/*.npy`) | Local `/content` working copy of the repo |
| Checkpoints (`checkpoints/*.pt`) | Any file not explicitly written to Drive |
| Dataset metadata (hashes, token counts) | Process memory, optimizer state in RAM |

Drive root used throughout: `/content/drive/MyDrive/SHNU_LLM/`.

## What a checkpoint contains

Every checkpoint is a complete, self-describing resume point:

- model weights
- optimizer state (AdamW moments)
- scheduler / global step number
- full config (architecture + hyperparameters)
- RNG state (Python, NumPy, CUDA) for reproducible continuation

Checkpoints are written **atomically** (`save → *.tmp → os.replace`) so a disconnect
mid-save can never corrupt the live checkpoint.

## The fresh-runtime sequence

1. **Clone the code from GitHub.** The repo is the source of truth for code; it carries
   no weights or data.
2. **Verify the commit.** Confirm the cloned HEAD is the expected commit so the resume
   runs against known code.
3. **Mount Google Drive.** OAuth is approved by the user; the session never handles
   credentials directly.
4. **Verify the dataset.** Check `wikitext103_metadata.json`'s SHA-256 prefix and exact
   train/val token counts against the recorded baseline.
5. **Verify the tokenizer.** Check `shnu_bpe.json`'s SHA-256 prefix and vocab size.
6. **Auto-discover the newest checkpoint.** `find_latest_checkpoint()` prefers
   `checkpoint_latest.pt`, else the highest `checkpoint_step_*.pt`. **No step number is
   ever typed by hand.**
7. **Verify checkpoint compatibility.** The checkpoint's architecture keys must match
   the current config; a mismatch stops the run rather than loading silently.
8. **Restore and continue.** Model, optimizer, scheduler, global step, and RNG state are
   restored; training continues from the exact saved step.
9. **Checkpoint periodically.** Every 500 steps a new checkpoint is written atomically
   and old ones are pruned (`keep_last=2`, plus `latest`/`best`), bounding Drive usage.
10. **On the next interruption, repeat from step 1.** The loop is indefinitely
    resumable across as many runtimes as a long run needs.

## Fail-safe behavior (this is the important part)

The resume path is designed to **fail safely** rather than quietly do the wrong thing:

- **No checkpoint found → STOP.** `scripts/resume.py` raises and exits rather than
  starting a brand-new run, so an interrupted long run is never silently replaced by a
  fresh one. Starting from scratch is only possible by explicitly passing
  `--allow-fresh`.
- **Incompatible checkpoint → STOP.** An architecture mismatch raises instead of
  loading partial/garbage weights.
- **Wrong dataset or tokenizer hash → STOP.** The run refuses to continue against data
  it cannot verify is the baseline.
- **No GPU / not a T4 / Drive not mounted / low disk → STOP.** The pre-flight gates
  (`src/shnu_llm/preflight.py`) raise `PreflightError`; the run **never falls back to
  CPU**. See [Safety gates](#safety-gates).

## Commands

Autonomous resume (the normal case — continues from the newest checkpoint, refuses to
start fresh):

```bash
python scripts/resume.py --config configs/v0.2.json \
    --out-dir /content/drive/MyDrive/SHNU_LLM --dataset wikitext103
```

Deliberately start a fresh run when no checkpoint exists yet:

```bash
python scripts/resume.py --config configs/v0.2.json \
    --out-dir /content/drive/MyDrive/SHNU_LLM --dataset wikitext103 --allow-fresh
```

Guarded long-run entrypoint (runs all gates, then requires explicit confirmation):

```bash
python scripts/train_v0_2.py --out-dir /content/drive/MyDrive/SHNU_LLM --confirm-long-run
```

## Safety gates

`src/shnu_llm/preflight.py` provides the gates that protect a free-tier run. Each raises
`PreflightError` on failure and the pipeline stops — none of them degrade to CPU or to
unverified data:

| Gate | Stops the run when |
|---|---|
| `require_gpu(require_t4=True)` | no CUDA GPU, or the GPU is not a T4 |
| `require_drive_mounted(...)` | Google Drive is not mounted |
| `require_free_disk(path, min_gb)` | insufficient free disk for checkpoints |
| `verify_dataset(...)` | dataset SHA-256 prefix or token counts do not match baseline |
| `verify_tokenizer(...)` | tokenizer SHA-256 prefix or vocab size do not match baseline |
| `verify_checkpoint(...)` | checkpoint architecture is incompatible with the config |

## Verification status

The resume/recovery logic is covered by the local test suite
(`tests/test_recovery.py`, `tests/test_preflight.py`) and has been exercised end-to-end
on a fresh runtime against Drive in earlier sessions. The **v0.2 long training run
itself has not been executed**, and the **T4 pilot has not been run** — this document
describes the procedure and guarantees, not a completed training result.
