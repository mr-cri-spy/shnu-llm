# Multi-Platform Training for SHNU-LLM v0.2

The v0.2 run reaches 20,000 steps across many short free-GPU sessions, possibly
on more than one service. This document describes how a session on **any**
supported platform continues the **same** run from the **same** newest valid
checkpoint — without ever creating a second authoritative checkpoint, resetting
the run to step 0, or training on CPU.

See also: [`FREE_GPU_OPTIONS.md`](FREE_GPU_OPTIONS.md) (which services and why),
[`LAUNCHER.md`](LAUNCHER.md) (the Colab one-cell launcher),
[`MONITORING.md`](MONITORING.md) (the status file),
[`RECOVERY.md`](RECOVERY.md) (fresh-runtime recovery).

## One run, one checkpoint format, one resume path

Everything platform-specific is isolated in a single small value,
`shnu_llm.platform.Platform` (whether a T4 is required, where durable state
lives, how much free disk to demand). Both launchers delegate to **one** shared
core, `shnu_llm.launcher_core.run_launch`, which performs the identical flow
everywhere:

    verify code/params  →  safety gates (fail-closed)  →  reconcile status
    from newest checkpoint (authoritative)  →  DRY-RUN stop, or resume-train

Resume itself is always `scripts/resume.py` (`resume_training`) and the
checkpoint is always the same format written by `training.save_checkpoint`. There
is no second checkpoint system and no second resume system — a platform only
changes the *environment*, never the *training*.

## The multi-session invariant

> **The newest _valid_ checkpoint is the single source of truth for the current
> step.** The status JSON is a convenience view and is always repaired from the
> checkpoint. A valid run is never reset to step 0.

Concretely, at the start of every session, on every platform:

1. **What is the newest valid checkpoint?** `find_latest_checkpoint` locates it;
   `checkpoint_safety.validate_checkpoint` verifies it (loads, has a model state
   dict and a step, architecture matches the config). `choose_authoritative`
   answers this deterministically even when several checkpoints or locations are
   in play — it returns the *valid* checkpoint with the *highest* step and
   ignores any corrupt candidate.
2. **Reconcile the status** from that checkpoint (`TrainingStatus.reconcile`).
3. **Resume** from exactly that step (optimizer, scheduler, and RNG restored).

Because the checkpoint is authoritative, a stale, missing, or corrupt status file
can never move the run backwards, and two machines can never disagree about the
step: whoever holds the highest-step valid checkpoint defines the run.

## Platform A — Google Colab (primary)

Colab mounts Google Drive natively, and the canonical store already lives there,
so a session is literally the one-cell launcher in [`LAUNCHER.md`](LAUNCHER.md):

```python
CONFIRM_LONG_RUN = False   # dry run; set True on a real T4 to train
OUT_DIR = "/content/drive/MyDrive/SHNU_LLM"
# (clone/pull repo, mount Drive, then:)
import colab_launch
report = colab_launch.launch(OUT_DIR, confirm_long_run=CONFIRM_LONG_RUN)
```

No checkpoint movement is needed — Drive *is* the canonical store. Re-run the
same cell after any disconnect; it resumes from the newest checkpoint.

## Platform B — Kaggle (secondary)

Kaggle **cannot mount Google Drive**, so the checkpoint travels as a verified
bundle. Static assets (tokenizer + datasets) never change, so they are attached
once as a read-only **Kaggle Dataset**; only the checkpoint moves each session.

### One-cell Kaggle launcher

```python
# ===================== SHNU-LLM v0.2 — KAGGLE LAUNCHER / RESUMER =====================
CONFIRM_LONG_RUN = False   # leave False for a safe dry run; True to actually train
OUT_DIR   = "/kaggle/working/SHNU_LLM"
DATA_ROOT = "/kaggle/input/shnu-llm-v02-data"      # attached read-only dataset (tokenizer+datasets)
IMPORT_BUNDLE = "/kaggle/input/shnu-llm-v02-ckpt/run.tar.gz"  # newest checkpoint bundle, or None
EXPORT_BUNDLE = "/kaggle/working/run_out.tar.gz"   # written at session end for the next session

import os, sys, subprocess
REPO = "https://github.com/mr-cri-spy/shnu-llm.git"; D = "/kaggle/working/shnu-llm"
if not os.path.isdir(os.path.join(D, ".git")):
    subprocess.run(["git", "clone", "--quiet", REPO, D], check=True)
subprocess.run(["git", "-C", D, "fetch", "--quiet", "origin", "main"], check=True)
subprocess.run(["git", "-C", D, "reset", "--hard", "--quiet", "origin/main"], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", os.path.join(D, "requirements.txt")])

sys.path.insert(0, os.path.join(D, "scripts"))
import kaggle_launch
report = kaggle_launch.launch(
    OUT_DIR, data_root=DATA_ROOT, import_from=IMPORT_BUNDLE,
    confirm_long_run=CONFIRM_LONG_RUN, allow_fresh=False,
)
if CONFIRM_LONG_RUN:
    kaggle_launch.export_bundle(OUT_DIR, EXPORT_BUNDLE)   # hand off to the next session
print("KAGGLE REPORT:", {k: report.get(k) for k in
      ("platform", "head", "reconciled_step", "target_steps", "trained", "dry_run")})
# ====================================================================================
```

What it does, in order: pull the committed code → stage the static assets from
the attached dataset → **import** the newest checkpoint bundle (hash-verified;
never adopts an older or corrupt checkpoint over a newer valid one) → verify code
→ safety gates (a GPU is **required**; a T4 is *not* required since Kaggle may
give a P100) → reconcile → dry-run stop, or resume-train → **export** the updated
bundle for the next session.

### The checkpoint bundle (export/import)

A bundle is a `tar.gz` produced by `checkpoint_safety.export_run_bundle`
containing `checkpoint_latest.pt` (+ `checkpoint_best.pt` if present), the
tokenizer, the status file, and a `MANIFEST.json` recording SHA-256 + size (and
step + architecture) for every file. Importing (`import_run_bundle`) **fails
closed**:

- every file is re-hashed and size-checked against the manifest before anything
  moves — a tampered or truncated bundle is refused (`ValueError`), and nothing
  is placed;
- each checkpoint is validated (loads, has a model + step, architecture matches);
- the never-older-over-newer / never-corrupt-over-valid rules decide whether to
  adopt, keep, or reject — an older or corrupt incoming checkpoint leaves the
  local one untouched;
- a differing tokenizer is refused rather than silently swapped (that would
  change the run).

Datasets (the large `.npy` streams) are **not** in the bundle — they ride in the
attached read-only Kaggle Dataset, because they never change.

### CLI equivalent

```bash
# import newest bundle, dry-run (no training):
python scripts/kaggle_launch.py --out-dir /kaggle/working/SHNU_LLM \
    --data-root /kaggle/input/shnu-llm-v02-data \
    --import-bundle /kaggle/input/shnu-llm-v02-ckpt/run.tar.gz

# resume-train, then export the updated bundle:
python scripts/kaggle_launch.py --out-dir /kaggle/working/SHNU_LLM \
    --data-root /kaggle/input/shnu-llm-v02-data \
    --import-bundle /kaggle/input/shnu-llm-v02-ckpt/run.tar.gz \
    --export-bundle /kaggle/working/run_out.tar.gz --confirm-long-run

# export only (e.g. to seed the first Kaggle bundle FROM Drive on a Colab session):
python scripts/kaggle_launch.py --out-dir /content/drive/MyDrive/SHNU_LLM \
    --export-bundle /content/drive/MyDrive/SHNU_LLM/run.tar.gz --export-only
```

## Deciding whether a platform is worth a session — SAFE benchmark

Before committing a session, measure the machine you were given without touching
the real run:

```bash
python scripts/benchmark.py --steps 50 --current-step 3000
```

`scripts/benchmark.py` builds a **fresh random** model, trains on **synthetic**
batches for a few dozen steps, and writes **nothing** — it never opens the real
checkpoint, dataset, or status (there is a test asserting `torch.load` is never
called). It reports measured tokens/sec, peak VRAM, and NaN/Inf, then projects
the fixed remaining-token budget onto that throughput as an **estimate** (pure
compute; add ~15–25% for eval/IO/reconnect). If it reports no GPU, that runtime
is CPU — stop and wait for a GPU rather than burning the session.

## Golden rules (enforced in code, not just prose)

1. **Fail closed on no GPU** — every launcher refuses to train on CPU
   (`preflight.require_gpu`). Colab additionally requires a T4; Kaggle requires
   any CUDA GPU.
2. **Checkpoint is authoritative** — status is only a view; reconciliation always
   repairs from the newest valid checkpoint and never resets a valid run to 0.
3. **Never older-over-newer, never corrupt-over-valid** — enforced on every
   import and by `decide_replace` / `choose_authoritative`.
4. **One authoritative checkpoint at a time** — the bundle carries a manifest so
   the receiver re-verifies; two checkpoints are never both "the truth".
5. **Legitimate free tiers only** — one account per service, no quota evasion. A
   busy or CPU-only free runtime is a reason to wait, never to work around.
