# SHNU-LLM v0.2.0 Evaluation Report

## Release

- Release: v0.2.0
- Checkpoint evaluated: step 20,000
- Model parameters: 33,890,816
- Architecture: decoder-only Transformer
- Layers: 8
- Hidden size: 512
- Attention heads: 8
- Context length: 512
- Vocabulary size: 16,000
- Tokenizer: SHNU BPE, trained from scratch
- Dataset: WikiText-103-raw-v1
- Training: random initialization, not fine-tuning

## Training checkpoint

The v0.2.0 training run completed at optimizer step 20,000.

Recorded training status:

- Tokens processed: 655,360,000
- Final recorded train loss: 2.995904
- Final recorded validation loss: 3.191914
- Final recorded perplexity: 24.335
- Best recorded validation loss: 3.182169
- Best checkpoint: step 19,000
- Device: NVIDIA Tesla T4
- Recorded throughput: 8,731.8 tokens/sec
- Sessions: 6
- Resumes: 6

The released latest checkpoint is step 20,000. The separate best checkpoint is step 19,000.

## Evaluation harness

The official evaluation harness was run against the recovered step-20,000 checkpoint.

Measured results:

| Metric | Value |
|---|---:|
| Validation loss | 3.175354 |
| Perplexity | 23.9353 |
| Train loss | 3.013781 |
| Evaluation iterations | 100 |
| Mean repeated n-gram fraction | 0.0963 |

These values are the measured evaluation-harness results for the recovered checkpoint.

## Context-length diagnostic

The context-length diagnostic used random token inputs and is not a language-quality metric.

| Context length | Loss |
|---:|---:|
| 8 | 14.1915 |
| 64 | 13.6064 |
| 256 | 13.3780 |
| 512 | 13.4880 |

These measurements are retained as a diagnostic only.

## Decoding evaluation

A separate D0-D5 decoding experiment was performed against the frozen step-20,000 checkpoint.

Evaluation conditions:

- Seed: 1337
- Maximum new tokens: 60
- Device: CPU
- Fixed prompts: 8
- Configurations: D0-D5

The complete comparison is documented in:

`evaluation/results/v0.2.0/decoding_comparison.md`

The raw machine-readable results are:

`evaluation/results/v0.2.0/decoding_d0_d5.json`

Summary:

| Config | Rep-3 | Distinct-1 | Distinct-2 | Mean length | EOS |
|---|---:|---:|---:|---:|---:|
| D0 | 0.0954 | 0.8438 | 0.8820 | 16.625 | 87.5% |
| D1 | 0.1558 | 0.6971 | 0.7952 | 35.750 | 75.0% |
| D2 | 0.0325 | 0.8040 | 0.9373 | 36.000 | 62.5% |
| D3 | 0.0000 | 0.8902 | 0.9864 | 30.375 | 75.0% |
| D4 | 0.0000 | 0.8339 | 0.9519 | 21.750 | 87.5% |
| D5 | 0.0000 | 0.9449 | 0.9972 | 25.000 | 87.5% |

D5 is the proposed decoding default for subsequent evaluation of this checkpoint:

- temperature 0.7
- top_p 0.9
- repetition penalty 1.2
- no-repeat 3-gram
- maximum 60 new tokens

This is a decoding proposal only. It is not evidence of improved model capability.

## Generation artifacts

The following fixed-seed generation artifacts are retained:

- `evaluation/results/v0.2.0/generations_t0.7.json`
- `evaluation/results/v0.2.0/generations_t1.0.json`
- `evaluation/results/v0.2.0/decoding_d0_d5.json`

The D0-D5 results were generated on CPU and should be described accordingly.

## BPB status

**NOT VERIFIED**

The BPB implementation is present in the evaluation code, but the original validation token stream/text artifact required to reproduce the metric was not recovered with the evaluation bundle.

No BPB number is reported here.

The missing validation artifact must be recovered before a BPB value is published.

## Reproducibility and artifact integrity

The recovered v0.2.0 model bundle contains:

- `checkpoint_latest.pt`
- `checkpoint_best.pt`
- tokenizer
- training status
- manifest

Latest checkpoint:

- Step: 20,000
- SHA-256: `9b472cf1dd4d56f893d12d58fcf68fa8a22f41458bbb71d45d030dd867cdcf7e`

Best checkpoint:

- Step: 19,000
- SHA-256: `15731810e0680b47fb6ab3349baddf504a0a188ad57510f83335ed16d53975c1`

Tokenizer SHA-256:

`1bf9397a6213d1a303c9d149d028fa2c456dd1a94da57c233b5b21588d1ac017`

Dataset SHA-256:

`1f9f77ec26e7477b7bf3c80c72a2e365749731f3b41b81af08ef16d29a4c0c80`

## Evaluation limitations

The reported validation and generation results do not establish factual accuracy or hallucination performance.

The decoding experiment uses only eight fixed prompts and is therefore diagnostic rather than a broad benchmark.

The context-length diagnostic uses random token inputs and must not be interpreted as model-quality evaluation.

BPB remains unverified.

No v0.3 training has been performed as part of this evaluation work.

## Release integrity

The v0.2.0 release checkpoint and release history are treated as immutable.

This evaluation work is performed on the `v0.2.1-eval` branch and does not modify the v0.2.0 release.
