# SHNU-LLM v0.2.0 Evaluation Report

Prepared on the `v0.2.1-eval` branch. Every number below is labeled with its
source; anything not yet confirmed from an actual artifact in this branch's
run is marked **NOT VERIFIED** rather than guessed, per this project's
evaluation-phase rules (see branch working notes). This document evaluates
the frozen v0.2.0 checkpoint (tag `v0.2.0`, commit `9b374b9`); it does not
modify that release.

## A. Training status metrics

Source: `training_status.json` (lives on the Kaggle/Drive run directory, not
in this git repo — `.gitignore` excludes `runs/`, `checkpoints/`, `*.pt`) and
`checkpoint_best.pt` / `checkpoint_latest.pt` loaded directly with
`torch.load(..., weights_only=False)`.

| Field | Value | Status |
|---|---|---|
| Latest checkpoint step | 20,000 | Verified in an earlier session via the actual `checkpoint_latest.pt` payload (`step` key) loaded on Kaggle, and independently via the harness's own `"checkpoint": "<file> @ step <n>"` field in `eval_report.json` from the successful evaluation run |
| Training loss (final) | 2.9959 (training-status figure, provisional) | Provisional — not yet re-confirmed against `checkpoint_latest.pt`'s own `log`/step record in this branch |
| Validation loss (final, training-status) | 3.1919 | Provisional (see note below on why this differs from harness validation loss in §B) |
| Best validation loss | 3.182169 (provisional, previously reported) | **NOT VERIFIED** — must be read from `checkpoint_best.pt["best_val"]` directly |
| **True best checkpoint step** | previously reported inconsistently as 16,000 in one place and 19,000 in another | **NOT VERIFIED — open discrepancy.** Must be resolved by loading `checkpoint_best.pt` and reading its own `step` field; do not assume either prior figure. This is the single most important unresolved number in this report. |
| Sessions / resumes | 6 / 6 (provisional, previously reported) | **NOT VERIFIED** against `training_status.json`'s `session_count`/`resume_count` in this branch |
| `checkpoint_compatible` | true (provisional) | **NOT VERIFIED** in this branch — `verify_checkpoint_compatible()` must be re-run here |
| Tokens processed | 655,360,000 (20,000 steps × 32,768 tokens/step) | Arithmetically consistent with `configs/v0.2.json` (`batch_size=16, grad_accum_steps=4, block_size=512`) and the step count above; not independently re-derived from a logged counter in this branch |

**Why this section is provisional:** the Kaggle notebook editor used to run
the verification script hit a sustained platform-side outage during this
branch's work session (cell area rendering blank, keystrokes not
registering — confirmed across repeated fresh-tab and reload attempts). The
checkpoint-verification script is written and ready
(see the branch's working notes); it has not yet executed successfully in
this session. This section will be updated with the actual printed values
the moment it runs.

## B. Evaluation harness metrics

Source: `evaluation/evaluate.py` run against `checkpoint_latest.pt` (step
20,000) with `configs/v0.2.json` and `evaluation/eval_prompts.json`
(`eval_iters=100`, `seed=1337`). **These numbers are verified** — they come
from an actual completed run of this harness in an earlier session (Kaggle
notebook `SHNU_v02`, committed version that finished successfully, log
ending `full report -> .../eval_report.json`).

| Metric | Value |
|---|---|
| Validation loss (harness) | 3.175354 |
| Perplexity (harness) | 23.9353 |
| Train loss (harness, same run) | 3.013781 |
| `eval_iters` | 100 |
| `mean_repeated_ngram_frac` | 0.0963 |
| Bits-per-byte (BPB) | **NOT VERIFIED** — the `bits_per_byte()` function and its wiring into `evaluate.py` are new on this branch (see `src/shnu_llm/evaluation.py`); it has not yet been executed against the real checkpoint because of the Kaggle outage above |

**Why training-status validation loss (§A, 3.1919) and harness validation
loss (§B, 3.175354) differ, even though both describe "the same" checkpoint:**
they are not the same measurement. `training_status.json`'s validation loss
is whatever the training loop's own periodic `estimate_loss()` call last
recorded during training, using the training run's own `torch.Generator`
state and `eval_iters` setting at that point in the run. The harness's
validation loss comes from a separate, later invocation of
`validation_perplexity()` (in `evaluation.py`) with its own fixed seed
(`cfg.seed`) and its own `eval_iters` (100, from `eval_prompts.json`),
sampling a different set of random batches from the same validation split.
Because `PackedDataset.get_batch` draws a *random* subset of validation
windows rather than iterating the full split, two runs with different seeds
or iteration counts will land on slightly different loss estimates even
against an unchanged checkpoint and dataset. Neither figure is wrong; they
are estimates of the same underlying quantity from different random
samples.

## C. Context diagnostic

**Random-token diagnostic, NOT a quality metric.** `context_length_test()`
feeds uniformly random token ids (not real text) through the model at
increasing sequence lengths and checks only that the forward pass produces a
finite loss — i.e. a structural check that the model doesn't numerically
break at longer contexts, not a measurement of how well it understands long
context.

| Length | Loss finite | Loss |
|---|---|---|
| 8 | true | 14.1915 |
| 64 | true | 13.6064 |
| 256 | true | 13.378 |
| 512 | true | 13.488 |

(Source: same verified harness run as §B.) These losses are expected to sit
near `ln(vocab_size) = ln(16000) ≈ 9.68` times some factor above true
language-model loss, since the inputs are random tokens the model was never
trained to predict — they are not comparable to §B's validation loss.

## D. Generation samples

**Status: NOT YET ATTACHED in this document.** Greedy, T=0.7, and T=1.0
samples for all 8 fixed prompts were generated against the real step-20,000
checkpoint in an earlier session and reported directly to the user, but the
structured JSON files this report should cite
(`evaluation/results/v0.2.0/generations_t0.7.json`,
`generations_t1.0.json`, with fixed `seed=1337` and per-prompt reset) have
not yet been produced in *this* branch — that requires the same
Kaggle-editor access blocked in §A. This section will quote directly from
those JSON files once they exist, rather than being retyped from an earlier
report (to avoid any risk of transcription drift from the verbatim model
output).

Known qualitative finding from the earlier greedy-decoding harness run
(§B): the harness's own fixed-prompt greedy generations showed a repetition
degeneration pattern consistent with `mean_repeated_ngram_frac = 0.0963` —
i.e. roughly 1 in 10 trigrams across the 8 greedy completions was a repeat
of an earlier trigram in the same completion. This is the specific failure
mode decoding controls in this branch (§E) are meant to address.

## E. Decoding comparison (D0-D5)

**Status: NOT YET RUN.** Requires the real checkpoint (same Kaggle
dependency as §A/§D). The decoding controls themselves
(`repetition_penalty`, `no_repeat_ngram_size`, `top_p`, `min_new_tokens` in
`src/shnu_llm/decoding.py`, wired into `ShnuLM.generate` and
`sample_text`) are implemented and unit-tested on this branch (CPU-only,
tiny deterministic models — see `tests/test_decoding.py`, 12 new tests, all
passing alongside the existing 38 tests: 50/50 total). The comparison table
below will be filled in and `evaluation/results/v0.2.0/decoding_comparison.md`
generated once the checkpoint is reachable.

Planned configurations (seed=1337, max_new_tokens=60, fixed 8-prompt set):

| Config | Settings |
|---|---|
| D0 | greedy baseline (current `evaluate.py` behavior) |
| D1 | top_k=50, temperature=0.7 |
| D2 | top_p=0.9, temperature=0.7 |
| D3 | D1 + repetition_penalty=1.2 |
| D4 | D1 + no_repeat_ngram_size=3 |
| D5 | D2 + repetition_penalty=1.2 + no_repeat_ngram_size=3 |

For each: `mean_repeated_ngram_frac`, `distinct-1`, `distinct-2`, mean
generated length, EOS-hit rate, plus one honest one-line fluency
observation (descriptive, not promotional).

**Proposed default: not yet selected** — a recommendation requires the
actual D0-D5 results above; none is proposed here in their absence.

## F. Failure-mode analysis

Based on the one verified generation run available (§B, greedy decoding,
`mean_repeated_ngram_frac = 0.0963` across the 8 fixed prompts):

- **Repetition** is the dominant observed failure mode under greedy
  decoding — consistent with a ~34M-parameter model at this training budget
  falling into locally-high-probability repeated n-grams once a phrase
  becomes likely.
- **EOS behavior, malformed completions, short completions, degeneration**
  under sampling (T=0.7/1.0) and under the D0-D5 decoding controls: **NOT
  VERIFIED** — requires the generation runs in §D/§E to characterize with
  actual evidence rather than expectation.

## G. Frozen baseline block

**This block is the fixed reference point every future v0.3 experiment
report (including EXP-001, see
`evaluation/results/v0.2.0/EXP-001.md`) must compare against.** It will be
finalized (all NOT VERIFIED lines resolved) once the Kaggle-dependent work
in §A/§D/§E completes; until then, downstream comparisons should treat
these numbers as provisional and re-check this section before relying on
them.

| Metric | Value | Verified in this branch? |
|---|---|---|
| Checkpoint | `checkpoint_latest.pt`, step 20,000, tag `v0.2.0` @ `9b374b9` | Yes |
| Harness validation loss | 3.175354 | Yes |
| Harness perplexity | 23.9353 | Yes |
| `mean_repeated_ngram_frac` | 0.0963 | Yes |
| Bits-per-byte (BPB) | — | No — pending §B |
| Best checkpoint step | — | No — the 16k/19k discrepancy is unresolved, see §A |
| Best validation loss | — | No — pending §A |

## Checkpoint metadata note

`ShnuConfig.version` defaults to `"v0.1"` in `src/shnu_llm/config.py`, and
the v0.2.0 release's checkpoint config is believed to carry that same
string forward (a config field, not a training/architecture value) — this
looks like a cosmetic labeling inconsistency (the code was versioned "v0.1"
when the config dataclass's default was set, and the v0.2 training run
never overrode it), not a sign the v0.2.0 checkpoint is actually a v0.1
architecture; the harness successfully loads it with
`verify_checkpoint_compatible()` passing (per the successful evaluation run
in §B, which would have raised `SystemExit` on an actual architecture
mismatch). This branch does **not** rewrite the released checkpoint to
"fix" this. A source-level correction (e.g. an explicit
`version="v0.2"` in a *new* config used for future runs) is a candidate
follow-up, tracked here rather than made silently.
