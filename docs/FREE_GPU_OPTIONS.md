# Free GPU Options for SHNU-LLM v0.2 Training

This document surveys **legitimate, free** GPU services for continuing the v0.2
run toward 20,000 steps across multiple sessions. It is written to keep two
things strictly separate:

- **Confirmed** — provider policy as publicly documented, or a number this
  project has itself measured on real hardware.
- **Estimate** — a projection derived from a measured number; labelled as such
  and never presented as fact.

We use only each provider's normal free tier through one account. We do **not**
bypass quotas, create multiple accounts, evade limits, or use unauthorized
compute. When a provider's free GPU is unavailable, the correct action is to
wait for the quota window or use another legitimate free tier — never to work
around the limit.

> Provider policies change often. Every "documented" figure below is *as of the
> last review* and should be re-checked against the provider's own current
> docs before relying on it. Treat them as guidance, not guarantees.

## Our measured baseline (confirmed)

These come from this project's own runs, not from any third party:

| Quantity | Value | Source |
|---|---|---|
| Model size | 33,890,816 params | measured (`ShnuLM.num_params()`) |
| Tokens per optimizer step | 32,768 (16 × 4 × 512) | config, confirmed |
| T4 throughput (pilot) | ≈ 9,940 tok/s | measured, free-Colab T4 |
| T4 throughput (real run) | ≈ 8,833 tok/s | measured, real v0.2 sessions |
| Peak GPU memory | ≈ 5.28 GB | measured, T4 |
| Steps per free-Colab session | ≈ 1,000 (≈ 79 min) | observed across real sessions |
| Durable checkpoint (current) | step 3,000 / 20,000 | Drive `training_status.json` |
| Remaining work | 17,000 steps = 557,056,000 tokens | derived from config |

The compute budget for the remaining work is fixed: **557,056,000 tokens**.
Everything below estimates *how long* that takes on each service by dividing
that fixed budget by a measured or documented throughput.

## Service comparison

| Service | Free GPU | Session cap | Quota | Native Drive mount | Persistent storage | Best for |
|---|---|---|---|---|---|---|
| **Google Colab (Free)** | 1× T4 (16 GB), when available | ~12 h documented; **disconnects far sooner in practice** | Dynamic, undocumented daily usage limit | **Yes** (`google.colab.drive`) | None on host; use Drive | Primary — our checkpoints already live on Drive |
| **Kaggle Notebooks** | 2× T4 (16 GB) **or** 1× P100 (16 GB) | 12 h max per session | **~30 h/week** GPU (documented, has varied) | **No** | `/kaggle/working` (~20 GB, persists within a notebook's versions); attach datasets read-only | Secondary — biggest documented weekly budget |
| **Paperspace Gradient (Free)** | M4000 / sometimes free A4000-class, subject to availability | ~6 h documented | Availability-gated; free GPUs often busy | No | Persisted `/notebooks` per-project storage | Occasional overflow |
| **SageMaker Studio Lab** | 1× T4 (16 GB) | ~4 h GPU / session, ~8 h/day documented | Daily GPU-hour cap | No | 15 GB persistent project storage | Occasional overflow |

Confirmed vs estimate for the table:

- **Confirmed (provider-documented):** Colab has a native Drive mount and no
  fixed quota; Kaggle offers 2×T4 **or** P100, a 12 h session cap, and a weekly
  GPU-hour budget (commonly documented around 30 h/week, though the exact number
  has changed over time); Kaggle has **no** native Google Drive mount.
- **Confirmed (measured by us):** the throughput, memory, and per-session step
  counts in the baseline table.
- **Estimate:** everything in the "time to finish" section below, and the
  Paperspace / Studio Lab session figures (those vary widely by availability and
  are included for completeness, not as a committed plan).

## Time-to-finish estimates (all ESTIMATES)

Pure GPU compute only — excludes eval passes, checkpoint I/O, Drive latency, and
reconnect overhead, which together added roughly 15–25% in our real sessions.
Divide the fixed 557,056,000-token budget by throughput:

| Throughput assumption | 500 steps | 1,000 steps | 5,000 steps | 17,000 steps (all remaining) |
|---|---|---|---|---|
| 8,833 tok/s (measured, conservative) | ≈ 31 min | ≈ 62 min | ≈ 5.2 h | ≈ **17.5 h** |
| 9,940 tok/s (measured, pilot best) | ≈ 27 min | ≈ 55 min | ≈ 4.6 h | ≈ **15.6 h** |

With real-session overhead folded in, plan on **≈ 20–22 h of wall-clock GPU
time** to finish the remaining 17,000 steps, split across sessions.

Session-count estimates (also estimates, and provider-availability-dependent):

- **Colab Free** at ~1,000 steps/session → **≈ 17 more sessions**.
- **Kaggle** at up to 12 h/session and a single T4 → an uninterrupted session
  could cover **several thousand steps**; a ~30 h/week budget could in principle
  cover the entire remaining run inside about one week, **if** sessions run
  clean and the quota holds. This is the optimistic case and is not guaranteed.

P100 vs T4: the P100 has no efficient bf16 path, so on Kaggle the **T4 (2×T4,
using one)** is the like-for-like choice for our bf16 autocast setup; a P100
would require fp16+GradScaler and is treated as a fallback, not the plan.
Kaggle's second T4 is **not** used — the training code is single-GPU by design,
and adding data-parallelism is out of scope for this task.

## Practical constraints that shape the design

1. **Kaggle cannot mount Google Drive.** Checkpoints must move between Kaggle
   and the canonical Drive store by an explicit, checksum-verified export/import
   step (see `docs/MULTI_PLATFORM_TRAINING.md`). Only **one** location is ever
   authoritative at a time; two checkpoints are never both "the truth".
2. **All platforms are ephemeral compute.** The host disk vanishes at session
   end. Durable state (checkpoints, status, tokenizer, datasets) lives in the
   canonical store; the GPU host is scratch.
3. **Fail closed on no GPU.** Every launcher refuses to train on CPU. A free
   service handing back a CPU runtime is a reason to stop and wait, not to burn
   the run on CPU.
4. **The checkpoint is authoritative.** Across every platform, the newest
   *valid* checkpoint defines the current step. The status file is a convenience
   view and is always repaired from the checkpoint.

## Recommendation

Keep **Colab Free** as the primary platform — our checkpoints already live on
Drive, and the one-cell launcher resumes with zero manual steps. Use **Kaggle**
as the secondary platform for its larger documented weekly budget, moving
checkpoints via the verified export/import bridge. Both paths converge on the
**same single checkpoint format** and the **same Drive-canonical store**, so no
matter where a session runs, the next session — anywhere — continues from the
newest valid checkpoint.

Sources (provider policies, re-verify before relying on them):
[Kaggle efficient GPU usage docs](https://www.kaggle.com/docs/efficient-gpu-usage),
[Kaggle weekly GPU quota discussion](https://www.kaggle.com/general/108481),
[Colab alternatives overview (Thunder Compute)](https://www.thundercompute.com/blog/colab-alternatives-for-cheap-deep-learning-in-2025),
[Free cloud GPU comparison (AIMultiple)](https://aimultiple.com/free-cloud-gpu).
