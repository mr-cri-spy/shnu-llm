# SHNU-LLM EXP-001 Closeout Report (v0.3)

**Verdict: FAIL. EXP-001 is not accepted and does not replace v0.2.0 as the baseline.**

This report closes EXP-001. It follows the structure of `docs/EVALUATION_v0.2.0.md`, as
required by §8 item 5 of the registration. The registration itself,
`evaluation/results/v0.3/EXP-001.md`, is left unchanged: it is the pre-registered
specification, and its thresholds are applied below exactly as written.

Every number here comes from a harness run or a hashed artifact. Anything not confirmed from
an artifact is marked **NOT VERIFIED**.

## 1. Objective

Test whether v0.2.0 is limited more by the amount of unique training data than by its
optimizer-step budget (registration §1).

## 2. Hypothesis

The same 20,000-step budget spent on a corpus with twice as many unique tokens
(WikiText-103 plus an equal-sized FineWeb-Edu slice) should produce a better model than
5.6 passes over WikiText-103.

## 3. Frozen configuration

These values are identical to v0.2.0, as enforced by `scripts/validate_exp001.py`.

| Group | Value |
|---|---|
| Architecture | 8 layers, 8 heads, d_model 512, block_size 512, vocab 16,000, 33,890,816 parameters |
| Tokenizer | `shnu_bpe.json`, SHA-256 `1bf9397a6213d1a303c9d149d028fa2c456dd1a94da57c233b5b21588d1ac017` |
| Seed | 1337 |
| Optimizer | AdamW (0.9, 0.95), weight decay 0.1, grad clip 1.0 |
| Learning-rate schedule | peak 3e-4, 500 warmup steps, cosine decay to 0.1 × peak |
| Budget | 20,000 steps × 32,768 tokens = 655,360,000 tokens |
| Evaluation | `evaluation/evaluate.py` (SHA-256 `d7d0583e…3591`), `evaluation/eval_prompts.json` (`bd156467…d0d0`), `configs/v0.2.json` (`3f539b3f…6e7d`); seed 1337, eval_iters 100, greedy 80 tokens |
| Code commit | `2d213943c66bb69d200c09b08237106d2ec840e9` (branch `v0.3-exp001`) |

## 4. Dataset composition

All files are in the Kaggle dataset `kirancrispy/shnu-v03-exp001-data`. Every SHA-256 below
was re-verified read-only on Kaggle on 2026-10-12.

| Stream | File | Tokens | SHA-256 |
|---|---|---|---|
| Mixed training corpus (50/50 by tokens) | `wikitext103_fineweb_edu_mix_train_v16000.npy` | 233,992,024 | `f33e196b…1ab01` |
| FineWeb-Edu training slice (`sample-10BT`, revision `87f09149…b8f9`) | `fineweb_edu_train_v16000.npy` | 116,996,469 | `74aff838…a95d5` |
| WikiText-103 validation (frozen v0.2.0 stream; also the training-loop validation copy) | `wikitext103_fineweb_edu_mix_val_v16000.npy` | 1,181,773 | `058b7107…ddcb8` |
| FineWeb-Edu held-out (disjoint documents, never trained on) | `fineweb_edu_heldout_v16000.npy` | 1,195,115 | `fce1c42c…21625` |
| Data manifest | `EXP001_DATA_MANIFEST.json` | — | `37c676e92be5d9d4f965b7f4a7f22e0db5ebb6ad64ceb9f53cf4247e5d6d225d` |

The WikiText-103 training stream is the v0.2.0 stream, reused byte-for-byte (SHA-256
`8b5ec7cf…4e923124`, 116,995,555 tokens). Expected epochs are 2.80 for EXP-001, against 5.60
for v0.2.0.

## 5. Training completion

| Field | Value | Source |
|---|---|---|
| State | `completed`, step 20,000 of 20,000 | `training_status.json` |
| Tokens processed | 655,360,000 | `training_status.json` |
| Sessions | 3 Kaggle sessions (2 resumes, from step 3000 and from step 14500) | `training_status.json`, run logs |
| Hardware | Tesla T4, GPU 0 only | `training_status.json` |
| PyTorch | 2.10.0+cu128 for steps 0–3000; 2.11.0+cu128 from the first resume onward and for evaluation | session logs |
| Final training-loop validation loss | 3.331575 (perplexity 27.9824) | `training_status.json` |

## 6. Final checkpoint

| Checkpoint | Step | Bytes | SHA-256 |
|---|---|---|---|
| `checkpoint_latest.pt` (acceptance checkpoint) | 20000 | 406,798,299 | `29d59e7a27b93290cbb2dec6673abadd55a203216e0c413b79dac1e289bac511` |
| `checkpoint_best.pt` (not used for acceptance) | 19500 | 406,797,691 | `cac3f125a6ba2811cb15b67bc7d0cc2b966bdebad5d495196d2376cd886daec3` |

Both are in the private Kaggle dataset `kirancrispy/shnu-v03-exp001-checkpoints`.

**Bookkeeping inconsistency (recorded, not investigated).**

- `training_status.json` reports `best_validation_step: 14500`. The `checkpoint_best.pt` file itself holds step 19500.
- Both record the same `best_val` of 3.335657.
- The final validation loss at step 20000 (3.331575) is lower than this stored best, yet the best was not updated.

Registration §7 says acceptance uses the final step-20000 checkpoint, so this inconsistency
does not affect the verdict.

## 7. Official v0.2.0 baseline

The baseline is the registered `checkpoint_latest.pt`, step 20000, SHA-256
`9b472cf1dd4d56f893d12d58fcf68fa8a22f41458bbb71d45d030dd867cdcf7e`. It was taken from the
SHNU_v02 notebook output bundle; its SHA-256 was verified before and after the run, and the
bundle's tokenizer hashes to the frozen tokenizer SHA.

| Split | Loss | Perplexity | BPB | `mean_repeated_ngram_frac` | Source |
|---|---|---|---|---|---|
| WikiText-103 validation | 3.175354 | 23.9353 | 1.040343 | 0.0963 | `docs/EVALUATION_v0.2.0.md` (frozen; not re-run) |
| FineWeb-Edu held-out | 6.091152 | 441.9303 | 2.197603 | 0.0963 | harness run, 2026-10-13, T4, commit `2d213943` |

The FineWeb-Edu held-out run reproduced the registered repetition value (0.0963) exactly.

## 8. EXP-001 results

These are harness runs on a Tesla T4 at commit `2d213943`, using `checkpoint_latest.pt`
(step 20000). The evaluation was run on 2026-10-12 and independently re-run on 2026-10-13.
The re-run matched exactly: every loss, perplexity, BPB, byte count, per-prompt repetition
value and context-length loss was identical.

| Split | Loss | Perplexity | BPB | `mean_repeated_ngram_frac` |
|---|---|---|---|---|
| WikiText-103 validation | 3.332190 | 27.9996 | 1.091728 | 0.5956 |
| FineWeb-Edu held-out | 3.458989 | 31.7848 | 1.247955 | 0.5956 |

Generation completed for all 8 prompts. The per-prompt repeated-trigram fractions were
0.7439, 0.5904, 0.6824, 0.6707, 0.5529, 0.4524, 0.5181 and 0.5542.

## 9. Acceptance thresholds (registration §7, unchanged)

| Metric | Pass condition |
|---|---|
| **Primary:** FineWeb-Edu held-out BPB | EXP-001 ≤ 0.95 × v0.2.0 on the same split |
| **Guardrail:** WikiText-103 validation BPB | ≤ 1.02 × 1.040343 = 1.06114986 |
| **Guardrail:** WikiText-103 perplexity | reported against 23.9353; covered by the BPB guardrail |
| **Guardrail:** `mean_repeated_ngram_frac` | ≤ 1.1 × 0.0963 = 0.10593 |

## 10. PASS/FAIL calculations

| Metric | v0.2.0 | Threshold | EXP-001 | Result |
|---|---|---|---|---|
| Primary: FineWeb-Edu held-out BPB | 2.197603 | ≤ 0.95 × 2.197603 = 2.08772285 | 1.247955 | **PASS** |
| Guardrail: WikiText-103 BPB | 1.040343 | ≤ 1.06114986 | 1.091728 | **FAIL** (+4.94% vs v0.2.0) |
| Guardrail: WikiText-103 perplexity | 23.9353 | covered by BPB guardrail | 27.9996 | reported (worse) |
| Guardrail: repetition | 0.0963 | ≤ 0.10593 | 0.5956 | **FAIL** (6.2 × v0.2.0) |

Registration §10 applies to this case: "If a guardrail fails, record the failure alongside
the primary result; EXP-001 does not replace v0.2.0 as the baseline."

For reference only, outside the registered criteria: WikiText-103 validation loss was 3.332190,
against 3.175354 for v0.2.0.

## 11. FineWeb-Edu result

EXP-001 substantially improved performance on the FineWeb-Edu distribution:

- loss went from 6.091152 to 3.458989;
- BPB went from 2.197603 to 1.247955, a 43.2% reduction.

This comparison is distribution-confounded. FineWeb-Edu was included in EXP-001 training,
while v0.2.0 was trained only on WikiText-103. The held-out documents are disjoint from the
training slice, but they come from the same distribution. The result therefore measures
in-distribution gain on FineWeb-Edu text. It is not a clean measure of general model
improvement, and it does not show that EXP-001 is generally more capable.

## 12. Generation failure

Under the registered decoding setup (greedy, 80 new tokens, EOS enabled), all eight EXP-001
generations fall into sentence-level loops: they state one sentence, then repeat it until the
token cap. For example, the first prompt produced:

> "The history of the United States is a very important part of the history of the United
> States. It is a very important part of the history of the United States. …"

Mean repeated-trigram fraction rose from 0.0963 (v0.2.0) to 0.5956.

## 13. Methodological caveats

These caveats qualify the evidence. None of them changes the verdict.

1. **Repetition comparability.** v0.2.0 often emitted EOS early. For example, its first prompt produced only "The history of the United States . \n", and its per-prompt fractions were [0, 0, 0.6706, 0, 0, 0, 0, 0.1]. EXP-001 usually ran to the full 80-token cap. The metric is therefore computed over very different lengths of generated text. This does not remove the failure: EXP-001's long outputs are visibly looping.
2. **FineWeb-Edu confound.** See §11.
3. **Isolated evaluation layout.** Each harness run used a new directory containing only the tokenizer and the `<dataset>_val_v16000.npy` file. With no train stream present, the harness's `train_loss` field is computed on validation data and is not meaningful. Before running, the validation batch positions were checked: they are identical to those drawn in the v0.2.0 layout, which included a WikiText train stream. Validation metrics therefore follow the baseline procedure.
4. **PyTorch version change.** Training moved from PyTorch 2.10.0 to 2.11.0 at the first resume. Its effect on the result is **NOT VERIFIED**.
5. **v0.2.0 WikiText-103 numbers were not re-run.** They come from the frozen v0.2.0 report.
6. **Artifact completeness (registration §8).** Two items were not met:
   - The v0.2.0 FineWeb-Edu `eval_report.json` was produced in a Kaggle session but was **not persisted**; its numbers are recorded here only.
   - An `export_run_bundle` `MANIFEST.json` was not produced for the EXP-001 checkpoints. The checkpoint dataset instead carries the uploader's `UPLOAD_MANIFEST.json` with SHA-256 values.

## 14. Final verdict

**FAIL.**

- **Primary metric:** the FineWeb-Edu held-out BPB passed.
- **Guardrails:** two of the pre-registered guardrails failed, WikiText-103 BPB (1.091728 > 1.06114986) and repetition (0.5956 > 0.10593).
- **Consequence:** under registration §10, EXP-001 is not accepted and is not treated as a successful model improvement.
- **Baseline:** v0.2.0 remains the baseline.
- **Secondary finding:** the FineWeb-Edu improvement is recorded as a secondary finding, subject to the confound in §11.

## 15. Evidence

| Evidence | Location |
|---|---|
| EXP-001 final evaluation (immutable) | private Kaggle dataset `kirancrispy/shnu-v03-exp001-final-eval-results`: `SUMMARY.json`, `wikitext103__evaluation__eval_report.json`, `fineweb_edu_heldout__evaluation__eval_report.json`, `eval_runner_exp001.log`, `SHA256_MANIFEST.json`. Uploaded privately, downloaded back and byte-verified by SHA-256 on 2026-10-13. |
| EXP-001 checkpoints, status and training logs | private Kaggle dataset `kirancrispy/shnu-v03-exp001-checkpoints` |
| Frozen data | Kaggle dataset `kirancrispy/shnu-v03-exp001-data` |
| v0.2.0 checkpoint | SHNU_v02 notebook output bundle (`checkpoints/checkpoint_latest.pt`, SHA-256 `9b472cf1…f7e`) |

## 16. Reproducibility

The evaluation is deterministic: fixed seed 1337, a fixed batch-draw generator, greedy
decoding, and frozen hashed inputs. The full EXP-001 evaluation was re-run from a fresh clone
and fresh output directories and matched the first run exactly.

To reproduce:

1. Check out commit `2d213943`.
2. Verify the pinned SHA-256 values of `evaluate.py`, `eval_prompts.json` and `configs/v0.2.json`.
3. Place the frozen tokenizer at `<out>/tokenizer/shnu_bpe.json`.
4. Place the frozen validation stream at `<out>/datasets/<name>_val_v16000.npy`.
5. Run:

```
python evaluation/evaluate.py --config configs/v0.2.json --out-dir <out> --dataset <name> --checkpoint <checkpoint_latest.pt>
```

Use a Tesla T4 with `CUDA_VISIBLE_DEVICES=0`.

## 17. Lessons learned (evidence-supported only)

1. A larger and more diverse training corpus did not automatically improve the frozen WikiText-103 benchmark: WikiText-103 BPB worsened from 1.040343 to 1.091728.
2. EXP-001 substantially improved FineWeb-Edu held-out performance, with BPB going from 2.197603 to 1.247955.
3. That improvement is partly confounded, because FineWeb-Edu was in EXP-001's training distribution and not in v0.2.0's.
4. Under the evaluated decoding setup the model developed severe generation repetition (0.5956 against 0.0963).
5. Validation loss alone was not enough to call the experiment a success.
6. The generation-quality guardrail mattered: it caught a serious failure that the FineWeb-Edu improvement does not show.
7. Although EXP-001 failed its acceptance test, it provides useful evidence for future work.

The cause of the repetition is **not** established by EXP-001.

## 18. Open questions (recorded, not answered)

- Why did FineWeb-Edu performance improve while WikiText-103 performance worsened?
- Why did generation repetition increase so sharply?
- Is the repetition caused by training dynamics, the corpus mixture, the EOS distribution, or another factor?
- Does a controlled corpus mixture improve generalization without degrading generation?
- Why does `checkpoint_best.pt` (step 19500) disagree with `training_status.json` (`best_validation_step` 14500), and why was the best not updated at step 20000?
