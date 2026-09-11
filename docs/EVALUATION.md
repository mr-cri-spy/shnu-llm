# SHNU-LLM — Evaluation Methodology

This document describes **how** SHNU-LLM is evaluated. All numbers in the notebook's
`final_report.json` are read directly from the run — nothing is hard-coded or fabricated.

## What we measure

| Metric | How | Meaning |
|---|---|---|
| Parameter count | `model.num_params()` (exact tensor sum) | model size |
| Vocabulary size | trained tokenizer `get_vocab_size()` | tokenizer capacity |
| Context length | `cfg.block_size` | max sequence the model attends over |
| Dataset size | chars/records after cleaning | training data scale |
| Training tokens | `batch × accum × block × steps` | optimization budget |
| Training / validation loss | mean cross-entropy over `eval_iters` batches | fit vs. generalization |
| **Perplexity** | `exp(validation_loss)` | standard LM quality proxy (lower = better) |
| Training time | wall clock of the train loop | efficiency |
| Tokens/sec | tokens processed / elapsed | throughput |
| Checkpoint size | file bytes | storage footprint |

## Sanity checks (must pass before trusting a run)

1. **Correct random init:** initial loss ≈ `ln(vocab_size)`. A correctly initialized
   softmax over `V` classes with no information has loss `ln(V)`. In the verified smoke
   run, initial loss **8.33** vs `ln(4000)` = **8.29** — as expected.
2. **Loss decreases:** training loss must fall monotonically-in-trend from the init value.
3. **Train/val gap:** a small, stable gap = healthy; a growing gap = overfitting;
   both high and flat = undertraining or a data/optimization bug.
4. **No NaN/Inf:** gradient clipping + AMP scaler guard against blow-ups; a NaN loss halts
   the analysis rather than being reported as success.
5. **Checkpoint round-trip:** save → reload into a *fresh* object → identical greedy output.
6. **Resume, not restart:** loading `checkpoint_latest.pt` continues from the saved step.

## Generation evaluation

We generate from multiple fixed prompts with greedy, temperature+top-k, and top-p decoding,
and save periodic samples to `SHNU_LLM/samples/step_*.txt`. Samples are **not cherry-picked** —
they are written on a fixed interval regardless of quality. For a small model, we look for:
local grammaticality, on-distribution vocabulary, and absence of degenerate loops under
sampling (greedy loops are expected at this scale).

## What perplexity does and does not tell us

Perplexity measures how well the model predicts held-out text from the **same distribution**.
It is a valid *relative* signal across SHNU-LLM versions trained/evaluated on the same data.
It is **not** a measure of reasoning, factuality, or instruction-following, and perplexities
are **not comparable** across different datasets or tokenizers. We therefore never compare
SHNU-LLM's perplexity to that of models trained on different corpora.

## Honesty rules for reporting

- "Training completed" only after the loop finishes and a checkpoint exists.
- "Model works" only after weights are loaded and text is generated in the same run.
- "Pretrained from scratch" only because weights are randomly initialized (verified via the
  `ln(V)` check) and trained through our loop.
- No fabricated losses, parameter counts, benchmark scores, times, or dataset sizes.

## Benchmarks (deliberately omitted in v0.1)

Standard benchmarks (HellaSwag, LAMBADA, etc.) require far more training than free-tier
compute allows to produce meaningful scores. Reporting near-random benchmark numbers would be
misleading, so v0.1 reports **loss and perplexity only**. A proper eval harness is planned for
a later version once the model is trained long enough for the scores to carry signal.

## v0.2 deterministic evaluation suite

v0.2 adds a reproducible evaluation suite (`src/shnu_llm/evaluation.py`, driven by
`evaluation/evaluate.py` with the fixed spec in `evaluation/eval_prompts.json`). It is
fully deterministic: a fixed seed (1337), greedy decoding for the reported generations,
and a fixed set of 8 prompts and context lengths. Running it twice on the same
checkpoint produces identical output.

| Component | What it reports |
|---|---|
| Validation loss / perplexity | mean cross-entropy over a fixed number of held-out batches (`eval_iters=100`), and `exp(loss)` |
| Fixed-prompt generation | greedy continuation of 8 fixed prompts (`max_new_tokens=80`) — deterministic and reproducible |
| Repetition metrics | per-generation `distinct_1`/`distinct_2`, repeated-n-gram fraction, and max consecutive repeat — a degeneration signal, plus a mean across the prompt set |
| Context-length test | a structural forward-pass check at lengths [8, 64, 256, 512] confirming the model produces finite loss across its full context window |

```bash
python evaluation/evaluate.py --config configs/v0.2.json \
    --out-dir /content/drive/MyDrive/SHNU_LLM --dataset wikitext103 \
    --checkpoint /content/drive/MyDrive/SHNU_LLM/checkpoints/checkpoint_latest.pt
```

The suite writes a full `eval_report.json` (and a compact console summary). Run without
`--checkpoint` for a structural dry run on a random-init model — useful for confirming
the harness works before any training exists.

> **Reported scores:** none yet. No v0.2 evaluation numbers are published here because
> v0.2 has **not been trained**. The suite above is the *methodology*; it will be run on
> a real checkpoint once one exists. No benchmark or perplexity score is claimed until it
> has actually been produced by a run.
