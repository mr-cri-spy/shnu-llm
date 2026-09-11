# SHNU-LLM — Persistent Training Monitor

A long v0.2 run spans many free-Colab sessions. The training monitor keeps a small,
crash-safe JSON file on Google Drive so a human — or a fresh runtime — can see what the
run is doing without loading the model or the GPU. It is a **side-car** on top of the
existing checkpoint/resume system (`src/shnu_llm/persistence.py`, `training.py`), not a
second checkpoint or resume mechanism.

> **A disconnected Colab runtime does NOT continue GPU training.** The status file only
> reflects the last durable state written before the disconnect. Monitoring is
> observation, not continuation.

## Exact status-file path

```
<out_dir>/runs/shnu_v0.2/training_status.json
```

For the production run `--out-dir /content/drive/MyDrive/SHNU_LLM`, so the file is:

```
/content/drive/MyDrive/SHNU_LLM/runs/shnu_v0.2/training_status.json
```

Checkpoints stay in their established location, `<out_dir>/checkpoints/` — the monitor
reconciles against them but never moves or owns them.

## Status schema

`TrainingStatus.default(cfg)` (`src/shnu_llm/status.py`) creates every field:

| Field | Meaning |
|---|---|
| `schema_version` | status schema version (currently 1) |
| `project`, `version`, `run_name` | identity |
| `config` | vocab, layers, heads, d_model, block_size, batch, grad_accum, lr, max_steps, `tokens_per_optimizer_step` |
| `git_commit` | repo HEAD the run is using |
| `state` | one of `active`, `paused`, `interrupted`, `completed`, `failed` |
| `current_step` / `target_steps` | optimizer steps done / target (20,000) |
| `tokens_processed` / `target_tokens` | `current_step × 32,768` / `target_steps × 32,768` |
| `completion_percent` | `100 × current_step / target_steps` |
| `train_loss`, `validation_loss`, `perplexity` | latest metrics (`perplexity = exp(validation_loss)`) |
| `best_validation_loss`, `best_validation_step` | best eval so far and its step (from `checkpoint_best.pt`) |
| `latest_checkpoint`, `best_checkpoint`, `checkpoint_timestamp` | durable checkpoint pointers |
| `session_count`, `resume_count` | how many sessions / resumes have occurred |
| `elapsed_seconds`, `throughput_tokens_per_second` | timing |
| `device`, `gpu_name`, `gpu_memory_gb` | runtime environment |
| `last_update_timestamp`, `last_error` | bookkeeping |
| `reconciliation` | `{occurred, reason, checkpoint_step, status_step_before, checkpoint_found}` |

## Status update timing

Status is written only at meaningful events, never per batch, to keep Google Drive I/O
light:

- **start / resume** — once when a session begins
- **evaluation** — at each `eval_interval` (every 500 steps for v0.2)
- **checkpoint save** — at each `checkpoint_interval` (every 1,000 steps)
- **interruption** — on `KeyboardInterrupt` (Colab disconnect / manual stop) → `interrupted`
- **failure** — on an unhandled exception → `failed`
- **completion** — when the loop reaches `max_steps` → `completed`

In-loop events are delivered through an optional `status_cb` argument to `train()`
(default `None`, so existing callers and tests are unaffected). Session-level events
(start/resume/interrupt/fail) are handled by `scripts/resume.py`.

## Atomic-write behavior

Every write goes to a `*.tmp` sibling file, is `flush`+`fsync`-ed, then `os.replace`-d
onto the real path — an atomic rename on POSIX. A Colab disconnect mid-write therefore
leaves either the old valid file or the new valid file, **never** a half-written,
corrupt JSON. A file that is somehow still unreadable is treated as missing (see below),
never as fatal.

## Checkpoint authority (the mandatory rule)

**CHECKPOINT STATE > STATUS JSON.** On a fresh runtime, `resume.py` calls
`TrainingStatus.reconcile(cfg)` before training:

1. find the newest valid checkpoint (`find_latest_checkpoint`);
2. read its step and verify architecture compatibility (`verify_checkpoint_compatible`);
3. read `training_status.json` if it exists;
4. compare checkpoint step with status step;
5. if they disagree — or the status is missing or corrupt — **trust the checkpoint** and
   repair the status from it;
6. record that reconciliation occurred (reason, checkpoint step, prior status step).

A valid checkpoint at step *N > 0* is **never** reset to step 0 because the status file
is stale, missing, or corrupt. If no checkpoint exists at all, the run is explicitly
recorded as a fresh run (`fresh_run: true`, `current_step: 0`); if the status claimed
progress but no checkpoint backs it, that contradiction is recorded in `last_error`
rather than silently trusted.

## Fresh-runtime recovery (multi-session training)

Each new free-Colab session: clone the repo from GitHub, mount Drive, reconcile status
from the newest checkpoint, resume training from that exact step (via the existing
resume path — RNG, optimizer and scheduler state included), and checkpoint again every
1,000 steps. The full procedure is in [`RECOVERY.md`](RECOVERY.md); the monitor adds only
the status view on top of it. `session_count` and `resume_count` accumulate across
sessions so you can see how many times the run has been picked up.

## Manual status inspection

Read-only, CPU-only, safe from anywhere that can see the Drive folder — including after
the Colab runtime has disconnected:

```bash
python scripts/status.py --out-dir /content/drive/MyDrive/SHNU_LLM          # human-readable
python scripts/status.py --out-dir /content/drive/MyDrive/SHNU_LLM --json    # raw JSON
python scripts/status.py --out-dir /content/drive/MyDrive/SHNU_LLM --reconcile  # repair from checkpoint
```

Without `--reconcile` the command only reads and prints; it never writes, never touches
the GPU, and never starts training.

## Google Drive vs GitHub persistence

- **Google Drive** holds all runtime state: datasets, tokenizer, checkpoints, logs, and
  this `training_status.json`. None of it is committed to Git.
- **GitHub** holds only source, configs, tests and documentation — including the monitor
  code (`status.py`, `scripts/status.py`, `tests/test_status.py`) but never any produced
  artifact.

## Limitations of external monitoring

- The status file is a **view of the last durable write**, not a live process feed. If a
  session dies between two events, the file shows the last event's state until a new
  session reconciles it from the checkpoint.
- Reading the status file does not prove a run is still training — a disconnected runtime
  leaves the file `active` even though no GPU work is happening. Treat `state: active`
  with an old `last_update_timestamp` as "last seen active", and reconcile to learn the
  true committed step.
- The monitor never fabricates progress: it only reports steps that a real checkpoint on
  Drive backs.
