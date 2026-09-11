# SHNU-LLM v0.2 — One-Cell Colab Launcher / Resumer

A long v0.2 run spans many free-Colab sessions. This launcher makes every session
identical: **paste and run one cell.** It pulls the latest code, verifies it,
mounts Drive, runs the safety gates, reconciles the status from the newest
checkpoint, resumes from the exact saved step (never from zero when a valid
checkpoint exists), trains to the configured target, and checkpoints + updates the
status atomically. When the runtime dies, you just run the same cell again — no
manual checkpoint selection, step selection, or dataset setup.

The cell is a thin, stable bootstrap; all logic lives in version control
(`scripts/colab_launch.py`, which orchestrates the existing checkpoint/resume,
status, and pre-flight code). GitHub holds code/config/docs; Google Drive holds
datasets/checkpoints/status/logs; no runtime artifact is ever committed.

## The cell

```python
# ===================== SHNU-LLM v0.2 — LAUNCHER / RESUMER =====================
# Run this one cell to resume (or, when confirmed, start) the v0.2 training run.
# Re-run it as-is after any Colab disconnect; it always continues from the newest
# checkpoint on Drive. It will NOT train unless CONFIRM_LONG_RUN is set to True.

CONFIRM_LONG_RUN = False   # <-- leave False for a safe dry run; set True to actually train
OUT_DIR = "/content/drive/MyDrive/SHNU_LLM"

import os, sys, subprocess
REPO = "https://github.com/mr-cri-spy/shnu-llm.git"
D = "/content/shnu-llm"

# 1) pull the latest main (clone on a fresh runtime, else hard-sync)
if not os.path.isdir(os.path.join(D, ".git")):
    subprocess.run(["git", "clone", "--quiet", REPO, D], check=True)
subprocess.run(["git", "-C", D, "fetch", "--quiet", "origin", "main"], check=True)
subprocess.run(["git", "-C", D, "checkout", "--quiet", "main"], check=True)
subprocess.run(["git", "-C", D, "reset", "--hard", "--quiet", "origin/main"], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", os.path.join(D, "requirements.txt")])

# 2) mount Google Drive (durable datasets / checkpoints / status / logs)
from google.colab import drive
drive.mount("/content/drive")

# 3) verify -> gate -> reconcile -> (optionally) resume-train, all in-process
sys.path.insert(0, os.path.join(D, "scripts"))
import colab_launch
report = colab_launch.launch(OUT_DIR, confirm_long_run=CONFIRM_LONG_RUN)
print("\nLAUNCH REPORT:", {k: report.get(k) for k in
      ("head", "reconciled_step", "target_steps", "trained", "dry_run")})
# =============================================================================
```

## What the cell does (in order)

1. **Pull latest main** — clone on a fresh runtime, otherwise `fetch` + `reset --hard origin/main`, then install requirements. You always run the current committed code.
2. **Mount Google Drive** at `/content/drive`.
3. `colab_launch.launch(OUT_DIR, confirm_long_run=CONFIRM_LONG_RUN)` then:
   - **Verifies code/config** — the v0.2 config matches the expected architecture and the model builds to exactly 33,890,816 parameters.
   - **Runs the safety gates** (fail-closed, never CPU fallback): GPU present and is a **T4**, Drive mounted, enough free disk, dataset SHA-256 + token counts, tokenizer SHA-256 + vocab, and — if a checkpoint exists — its architecture compatibility.
   - **Reconciles `training_status.json`** from the newest checkpoint (the checkpoint is authoritative; a valid run is never reset to step 0).
   - **If `CONFIRM_LONG_RUN` is False:** prints the plan (resume-from step → target) and **STOPS** — a safe dry run that also serves as validation.
   - **If `CONFIRM_LONG_RUN` is True:** resumes from the exact checkpoint step and trains to `max_steps` (20,000), writing atomic checkpoints and status updates, and marking `interrupted` / `failed` / `completed` for the next session.

## Multi-session behavior

- **First ever run** (no checkpoint): starts fresh at step 0 (the launcher passes `allow_fresh=True` — it is the deliberate entry point).
- **After a disconnect**: the same cell resumes from the newest checkpoint on Drive with RNG/optimizer/scheduler state restored — no manual step or checkpoint selection.
- **On interruption/failure**: the last atomic checkpoint (written every `checkpoint_interval` steps) plus the status file are enough for the next launch to continue; the status records `interrupted`/`failed` with the reason.
- **Monitor while it runs**: from any machine that can see the Drive folder, `python scripts/status.py --out-dir /content/drive/MyDrive/SHNU_LLM` prints the live-as-of-last-write status (see [`MONITORING.md`](MONITORING.md)). A disconnected runtime does not keep training — the status shows the last durable state until the next launch reconciles it.

## Starting the real 20,000-step run

The run stays gated behind an explicit flag exactly like `scripts/train_v0_2.py --confirm-long-run`. To actually start (or resume) training, change the first line of the cell to:

```python
CONFIRM_LONG_RUN = True
```

and run the cell on a runtime with a real T4. Until then every run of the cell is a safe dry run that verifies, gates, reconciles, and stops.
