# SHNU-LLM v0.2.0 Evaluation Report

This report evaluates the frozen v0.2.0 release (tag `v0.2.0`, commit `9b374b9`, checkpoint
`checkpoint_latest.pt` at step 20000). It does not modify that release, and no training was
performed. Every number below is traced to an artifact, and anything not confirmed from an
artifact is marked **NOT VERIFIED**.

Primary artifacts (all produced on Kaggle and byte-verified by SHA-256 after transfer):

| Artifact | SHA-256 | Bytes |
|---|---|---|
| `evaluation/results/v0.2.0/generations_t0.7.json` | `cf1effaa4b1f0e131b5f6ef80e37d9b4f7c8d168dd0e5ab60cf9afcd2a4d3252` | 3,264 |
| `evaluation/results/v0.2.0/generations_t1.0.json` | `0a1818873a855df065abbcde7a890300cf7aba090b5adea11213affb9d1b5ba2` | 3,329 |
| `evaluation/results/v0.2.0/bpb.json` | `3074ffeb5c5528203fd110cd231507249940010fa7f00853b8b3521789352840` | 540 |
| `evaluation/results/v0.2.0/decoding_comparison.json` | `e6f7b9d71c3a5338a3c016b306406fe6d0fbec21f025e723452a7203d3f7413d` | 21,130 |

## A. Checkpoint verification and training status

**Checkpoint selection.** The verification script scanned every attached Kaggle input folder
that contains `checkpoints/checkpoint_latest.pt`, and excluded `shnu-ckpt-17k`. Only one folder,
the v0.2.0 training notebook's own output bundle (`checkpoints/`, `tokenizer/`, `MANIFEST.json`), holds a
step-20000 `checkpoint_latest.pt`. Its SHA-256 matches both that folder's `MANIFEST.json` and
the recorded v0.2.0 prefix `9b472cf1`. The `shnu-llm-v02/run (1)` folder holds a step-6000
checkpoint and was not used.

| Field | Value | Source |
|---|---|---|
| Latest checkpoint step | 20000 | `checkpoint_latest.pt["step"]` |
| Latest checkpoint SHA-256 | `9b472cf1dd4d56f893d12d58fcf68fa8a22f41458bbb71d45d030dd867cdcf7e` (406,798,758 bytes) | computed; matches MANIFEST |
| **True best checkpoint step** | **19000** | `checkpoint_best.pt["step"]`; the MANIFEST also says 19000 |
| Best validation loss | 3.1821694374084473 | `checkpoint_best.pt["best_val"]`; equals the lowest value in the checkpoint's own eval log (46 evaluations) |
| Best checkpoint SHA-256 | `15731810e0680b47fb6ab3349baddf504a0a188ad57510f83335ed16d53975c1` (406,798,084 bytes) | computed; matches MANIFEST |
| Tokenizer SHA-256 | `1bf9397a6213d1a303c9d149d028fa2c456dd1a94da57c233b5b21588d1ac017` | computed; matches MANIFEST and the known value |
| Checkpoint compatible | true | `verify_checkpoint_compatible()` |
| State | completed, step 20000 of 20000 | `training_status.json` |
| Tokens processed | 655,360,000 (32,768 tokens per optimizer step) | `training_status.json` |
| Final train loss | 2.995904 | `training_status.json` |
| Final validation loss (training-status) | 3.191914 (perplexity 24.335) | `training_status.json` |
| Sessions / resumes / fresh_run | 6 / 6 / false | `training_status.json` |
| GPU / throughput | Tesla T4 / 8731.8 tokens/s | `training_status.json` |

**How the 16k/19k discrepancy resolves.** `training_status.json` reports
`best_validation_step: 16000` and, in the same file, `best_validation_loss: 3.182169`.
That loss belongs to step 19000, so the status file contradicts itself.

- **The checkpoint wins.** `checkpoint_best.pt` (step 19000), the bundle MANIFEST (step 19000), and the checkpoint's training log (lowest loss at step 19000) all agree. By the repository's own rule, the checkpoint is authoritative.
- **Cause, confirmed by reading the source.** `scripts/resume.py`'s status callback writes `best_validation_loss` on every eval event but never writes `best_validation_step`. That field is only set by `TrainingStatus.reconcile()` in `src/shnu_llm/status.py`, which last ran at step 17000 (`reconciliation.checkpoint_step: 17000`), when 16000 really was the best. Fixing this belongs in a future controlled change; it is not a data problem.
- **The best came late in training.** The last logged eval (step 19500) had validation loss 3.19944, and the final validation loss was 3.191914.

**Other status notes.**

- `last_error` is "KeyboardInterrupt (Colab disconnect / manual stop)" even though `state` is "completed". `scripts/resume.py` writes that string when a session is interrupted, and nothing clears it on a later successful completion. It is left over from an earlier interrupted session, not a failure of the final run.
- The checkpoint config and status file both carry `version: "v0.1"`. This is a metadata label only, since the architecture fields match v0.2. The released checkpoint was not modified.

## B. Evaluation-harness metrics

These come from the same procedure as `evaluation/evaluate.py`: `configs/v0.2.json`, seed 1337,
`eval_iters=100`, on the train and validation `.npy` streams (validation: 1,181,773 tokens,
SHA-256 `058b710795d4a8dd6b7655a635eda645d7009b2112cccd53e7712783e5eddcb8`).

| Metric | Value |
|---|---|
| Harness validation loss | 3.175354 |
| Harness perplexity | 23.9353 |
| Harness train loss | 3.013781 |
| `mean_repeated_ngram_frac` (greedy, 80 tokens, `evaluate.py` run of the SHNU_v02 notebook, version 5) | 0.0963 |
| **Bits per byte (BPB)** | **1.040343** |

The BPB run reproduced the harness validation loss exactly (3.175354), on CPU, in 1,169 seconds.

**How BPB was computed.**

- Formula: BPB = val_loss_nats × (total validation tokens / total UTF-8 bytes) / ln 2 = 3.175354 × (1,181,773 / 5,203,841) / 0.693147 = 1.040343.
- Byte counting: the full validation token stream was decoded with the run's own tokenizer (`tok.decode` over all validation ids) and the result was UTF-8 encoded.
- Caveat: `decode` drops special tokens, so the `<eos>` separators count as tokens but contribute no bytes. This biases tokens-per-byte slightly upward, and therefore BPB slightly upward.

**Why the training-status and harness validation losses differ (3.191914 vs 3.175354).** Both are
estimates over random batches drawn from the validation split, not full passes.

- The training-status value comes from the training loop's last `estimate_loss()` call. That call uses the training run's generator state.
- The harness value comes from a fresh `validation_perplexity()` call with seed 1337 and 100 iterations.
- Different random windows give slightly different estimates. Neither is wrong, but cross-version comparisons must use the same procedure, which means the harness value.

## C. Context-length diagnostic

**This is a random-token diagnostic, NOT a quality metric.** It feeds uniformly random tokens at
each length and checks only that the loss is finite.

| Length | Loss finite | Loss |
|---|---|---|
| 8 | true | 14.1915 |
| 64 | true | 13.6064 |
| 256 | true | 13.378 |
| 512 | true | 13.488 |

These losses sit above ln(16000) ≈ 9.68, the loss of a uniform guess. That is expected, because a
trained model is confidently wrong on random targets. They say nothing about long-context language
quality.

## D. Generation samples

The full verbatim text is in the JSON files; nothing below is edited. The prompts are
`evaluation/eval_prompts.json`, and the checkpoint is `checkpoint_latest.pt` at step 20000.

- **Greedy:** from `decoding_comparison.json` configuration D0 (60 tokens, EOS enabled). The original `evaluate.py` run used 80 tokens.
- **T=0.7 and T=1.0:** from `generations_t0.7.json` and `generations_t1.0.json`, both with top_k=50, 60 new tokens, seed 1337 reset per prompt, run on CPU. These two runs did not pass `eos_id`, so generation continued past sentence ends.

Representative excerpts, quoted exactly:

- **Greedy:** "The theory suggests that the theory of evolution is more likely to be influenced by the theory of evolution ."
- **T=0.7:** "The history of the Roman Empire . \n The Roman army was led by the Roman general Aubergeurs , who commanded a regiment of the Roman army in the first three years of the war . …"
- **T=1.0:** "The theory suggests that they are very much more likely to have originated earlier than before , due to the use of their longitudinal brachycephalian adaptations . …"

## E. Decoding comparison (D0–D5)

Settings: seed 1337, 60 new tokens, EOS enabled, CPU. See `evaluation/results/v0.2.0/decoding_comparison.md` for the
per-configuration observations.

| Config | rep-3gram | distinct-1 | distinct-2 | mean new tokens | EOS-hit rate |
|---|---|---|---|---|---|
| D0 greedy | 0.0837 | 0.8438 | 0.8820 | 16.62 | 0.875 |
| D1 top_k=50, T=0.7 | 0.0250 | 0.8797 | 0.9627 | 26.88 | 0.875 |
| D2 top_p=0.9, T=0.7 | 0.0204 | 0.8928 | 0.9623 | 19.50 | 1.000 |
| D3 D1 + rep_penalty 1.2 | 0.0019 | 0.8955 | 0.9804 | 33.38 | 0.750 |
| D4 D1 + no_repeat_ngram 3 | 0.0000 | 0.9104 | 0.9921 | 24.50 | 0.875 |
| D5 D2 + rep_penalty 1.2 + no_repeat_ngram 3 | 0.0000 | 0.9248 | 0.9914 | 25.62 | 0.875 |

These results supersede the earlier `evaluation/results/v0.2.0/decoding_d0_d5.json`, which counted
repetition over generated tokens only. Its sampled text is consistent with seeding once per
configuration rather than before every prompt. The details are in `decoding_comparison.md`.

**Proposed default: D4.** It is the smallest change from D1, it removes the measured trigram
repetition, and it matches D1 on 7 of 8 prompts. The recommendation is provisional because it rests
on 8 prompts and one seed.

## F. Failure-mode analysis (observed evidence only)

- **Repetition under greedy decoding.** The one 60-token greedy completion collapses into a loop ("the two groups of the three groups …", repeated-trigram fraction 0.5692). The harness value at 80 tokens is 0.0963.
- **Early EOS and short completions.** With EOS enabled, 75–100% of completions end before 60 tokens (mean 16.6–33.4 new tokens), usually right after " . \n". The likely cause is the per-record `<eos>` in the training stream, but that is **NOT VERIFIED**. No D0–D5 setting changes this.
- **Malformed completions.**
  - Prompts ending in "," often continue with ",000th", ",000 + ≈ XL2 + 1", or "XL2786453.112xHz", which look like fragments of WikiText number formatting.
  - One T=1.0 sample contains the Unicode replacement character "�", which comes from a partial byte-level token.
  - "@-@" artifacts come from WikiText's raw formatting.
- **Degeneration versus drift.** Sampling (D1–D5) avoids loops but drifts off topic within one or two sentences, and its factual content is unreliable. This is descriptive, not measured.

## G. Frozen baseline block (compare every v0.3 experiment against this)

The pre-registered v0.3 experiment `evaluation/results/v0.2.0/EXP-001.md` is a specification only and
has not been trained. It uses these values as its baseline.

| Metric | Value |
|---|---|
| Checkpoint | `checkpoint_latest.pt`, step 20000, SHA-256 `9b472cf1dd4d56f893d12d58fcf68fa8a22f41458bbb71d45d030dd867cdcf7e`, tag `v0.2.0` @ `9b374b9` |
| Tokenizer | 16,000-vocab byte-level BPE, SHA-256 `1bf9397a6213d1a303c9d149d028fa2c456dd1a94da57c233b5b21588d1ac017` |
| Validation data | 1,181,773 tokens, `.npy` SHA-256 `058b710795d4a8dd6b7655a635eda645d7009b2112cccd53e7712783e5eddcb8` |
| Harness validation loss | 3.175354 |
| Harness perplexity | 23.9353 |
| Bits per byte | 1.040343 |
| `mean_repeated_ngram_frac` (harness greedy, 80 tokens) | 0.0963 |
| Best checkpoint | step 19000, best_val 3.1821694 (training-loop estimate) |
| Training budget | 20,000 steps, 655,360,000 tokens, 6 sessions |
| Dataset SHA-256 (`1f9f77ec26e7477b7bf3c80c72a2e365749731f3b41b81af08ef16d29a4c0c80`, previously reported) | **NOT VERIFIED** in this evaluation. Only the validation `.npy` hash above was computed. |
