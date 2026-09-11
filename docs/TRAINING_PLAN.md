# SHNU-LLM v0.2 — Training Plan

This document records the **deterministic** training plan for SHNU-LLM v0.2. Every
figure below is either fixed by the config or computed from it. Values that depend on
real GPU throughput are explicitly marked **UNKNOWN** and are only filled in after a
measured pilot — they are never invented.

> **Status:** v0.2 long training has **not** been run, and the T4 pilot has **not**
> been run. This is a plan, not a result.

Regenerate the machine-readable version of this plan at any time:

```bash
python scripts/training_plan.py
```

## Model and data fixed points

| Quantity | Value | Source |
|---|---|---|
| Parameters | 33,890,816 | measured (`model.num_params()`) |
| d_model / layers / heads | 512 / 8 / 8 | `configs/v0.2.json` |
| Context length (block size) | 512 | `configs/v0.2.json` |
| Vocabulary | 16,000 | v0.2 tokenizer |
| Micro-batch size | 16 | `configs/v0.2.json` |
| Gradient accumulation | 4 | `configs/v0.2.json` |
| Effective batch | 64 sequences | 16 × 4 |
| Train tokens available | 116,995,555 | `wikitext103_metadata.json` (Drive) |
| Validation tokens available | 1,181,773 | `wikitext103_metadata.json` (Drive) |

## Tokens per step

```
tokens per optimizer step = micro_batch × grad_accum × block_size
                          = 16 × 4 × 512
                          = 32,768 tokens
```

Each **optimizer step** consumes 32,768 tokens and is made of 4 micro-steps of
8,192 tokens each (accumulated before the weight update).

## Steps per epoch

```
steps per epoch = train_tokens / tokens_per_optimizer_step
                = 116,995,555 / 32,768
                ≈ 3,570 steps   (3,570.4)
```

## Recommended training budget

For a ~34M-parameter model on ~117M tokens, roughly **2 epochs** (≈ 2× the corpus) is a
sensible, compute-aware target — enough to move well past the v0.1 proof-of-concept
regime without overfitting a single pass or wasting scarce free-tier compute.

| Quantity | Value |
|---|---|
| Recommended epochs | 2 |
| Recommended training tokens | 233,991,110 |
| Recommended total steps | 7,141 |

`configs/v0.2.json` sets `max_steps = 20000` as an upper safety ceiling; the
**recommended** target above (7,141 steps) is the planned stopping point and is where
training should be evaluated and, if healthy, stopped or extended deliberately.

## Checkpoint and validation cadence

| Interval | Value | Rationale |
|---|---|---|
| Checkpoint every | 500 steps | ≈ every 16.4M tokens; bounds lost work on a free-tier disconnect to < ~7% of one epoch |
| Validation every | 500 steps | aligned with checkpoints so each saved state has a paired val-loss/perplexity |

At 500-step checkpoints the recommended run produces ≈ 14 checkpoint events.

## Checkpoint storage

Checkpoints store model + optimizer + scheduler step + config + RNG state (everything
needed for an exact resume).

| Quantity | Value | Source |
|---|---|---|
| Single checkpoint size | 406.8 MB | **measured** on disk for this config |
| Retention policy | keep last 2 + `latest` + `best` | `prune_checkpoints(keep_last=2)` |
| Expected peak checkpoint storage | 1,627.2 MB | 4 × 406.8 MB worst case during rotation |

Optimizer state (AdamW: two moment tensors per parameter) dominates the file size — it
is roughly 2× the raw parameter bytes — which is why a checkpoint is ~12× larger than
the weights alone. Checkpoints live on **Google Drive**, never in Git.

## Gradient accumulation

Gradient accumulation decouples the *effective* batch (64 sequences) from what fits in
GPU memory (16 sequences). The loop scales each micro-step loss by `1/grad_accum`,
accumulates gradients over 4 micro-steps, then performs a single clipped optimizer step
and scheduler tick. This keeps the optimization math identical to a true batch of 64
while keeping peak activation memory at the 16-sequence level.

## T4-dependent values — measured by the pilot

These were **UNKNOWN until measured on a real T4**. The values below come from a real
pilot run on a free-Colab Tesla T4 (15.64 GB, torch 2.11.0+cu128 / CUDA 12.8) via
`scripts/pilot.py --config configs/v0.2.json --pilot-steps 40`. They are measurements,
not analytic guesses.

| Quantity | Measured value |
|---|---|
| GPU | Tesla T4, 15.64 GB |
| T4 throughput | **9,940.6 tokens/sec** (pilot benchmark, bfloat16 autocast) |
| T4 peak GPU memory | **5.28 GB** (of 15.64 GB — comfortable headroom) |
| AMP dtype actually used | `torch.bfloat16` (auto-selected by `resolve_dtype`; not fp16) |
| Estimated wall-clock, recommended run | **≈ 6.5 hours** = 233,991,110 tokens ÷ 9,940.6 tok/s ≈ 23,540 s |

The wall-clock estimate is derived from the measured pilot throughput; a full run's
sustained rate may differ somewhat from the short pilot benchmark, so treat ~6.5 h as an
informed estimate rather than a guarantee.

**Runtime caveat:** free T4 sessions time out and disconnect without warning, and a
~6.5-hour run will not fit in one session. The run **must** be resumable and checkpoint
frequently (every 500 steps here), so it spans several resumable sessions. The recovery
procedure is documented in [`RECOVERY.md`](RECOVERY.md).

## How a real run is launched (guarded)

Long training is never started casually. `scripts/train_v0_2.py` runs every safety gate
first (GPU present and is a T4, dataset hash, tokenizer hash, checkpoint compatibility,
Drive mounted, free disk) and then **refuses to train** unless `--confirm-long-run` is
also passed. Without that flag it performs a dry run of the gates and stops.

```bash
# dry run — checks gates, does NOT train
python scripts/train_v0_2.py --out-dir /content/drive/MyDrive/SHNU_LLM

# real run — only after a passing measured pilot and a deliberate decision
python scripts/train_v0_2.py --out-dir /content/drive/MyDrive/SHNU_LLM --confirm-long-run
```
