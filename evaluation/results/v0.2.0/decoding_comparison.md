# SHNU-LLM v0.2.0 Decoding Comparison

## Purpose

This experiment evaluates decoding controls on the frozen SHNU-LLM v0.2.0 step-20,000 checkpoint.

No model weights were changed and no training was performed.

## Evaluation provenance

| Field | Value |
|---|---|
| Checkpoint | v0.2.0 step 20,000 |
| Seed | 1337 |
| Max new tokens | 60 |
| Device | CPU |
| Prompts | 8 fixed prompts |
| Configurations | D0-D5 |
| Model parameters | 33,890,816 |

## Configurations

| Config | Configuration |
|---|---|
| D0 | Greedy |
| D1 | Temperature 0.7 + Top-K 50 |
| D2 | Temperature 0.7 + Top-P 0.9 |
| D3 | D1 + repetition penalty 1.2 |
| D4 | D1 + no-repeat 3-gram |
| D5 | D2 + repetition penalty 1.2 + no-repeat 3-gram |

## Measured results

| Config | Rep-3 | Distinct-1 | Distinct-2 | Mean length | EOS |
|---|---:|---:|---:|---:|---:|
| D0 | 0.0954 | 0.8438 | 0.8820 | 16.625 | 87.5% |
| D1 | 0.1558 | 0.6971 | 0.7952 | 35.750 | 75.0% |
| D2 | 0.0325 | 0.8040 | 0.9373 | 36.000 | 62.5% |
| D3 | 0.0000 | 0.8902 | 0.9864 | 30.375 | 75.0% |
| D4 | 0.0000 | 0.8339 | 0.9519 | 21.750 | 87.5% |
| D5 | 0.0000 | 0.9449 | 0.9972 | 25.000 | 87.5% |

## Observations

D0 produces relatively short generations and frequently reaches EOS, but repetition remains visible for some prompts.

D1 produces longer generations, while its measured repeated 3-gram fraction is higher than D0.

D2 reduces measured repetition relative to D1 and increases Distinct-2, but has the lowest EOS hit rate among the six configurations.

D3 eliminates measured repeated 3-grams while retaining longer generations. Some generations reach the 60-token limit without EOS.

D4 eliminates measured repeated 3-grams and produces shorter average generations than D3, with 87.5% EOS.

D5 eliminates measured repeated 3-grams and records the highest Distinct-1 and Distinct-2 values in this experiment. Its EOS rate is 87.5%, with a 25-token mean generated length.

## Proposed decoding default

For subsequent evaluation of this checkpoint, D5 is the proposed default decoding configuration:

- temperature = 0.7
- top_p = 0.9
- repetition_penalty = 1.2
- no_repeat_ngram_size = 3
- max_new_tokens = 60

This proposal is limited to decoding behavior measured in this experiment. It does not establish that D5 improves the underlying model or training quality.

## Metric definitions and comparison boundary

The D0-D5 results are generation-only measurements from `decoding_d0_d5.json`.

The repeated 3-gram metric in this table should not be treated as directly equivalent to the repetition metric in the main evaluation harness unless the implementations are confirmed to use identical token spans and definitions.

Distinct-1 and Distinct-2 are lexical diversity measurements over generated output.

EOS is the percentage of the eight fixed prompts for which generation emitted the configured EOS token before reaching the maximum generation length.

## Reproducibility

The authoritative raw results are stored in:

`evaluation/results/v0.2.0/decoding_d0_d5.json`

The original temperature/top-k and temperature-1.0 generation artifacts are stored separately:

`evaluation/results/v0.2.0/generations_t0.7.json`

`evaluation/results/v0.2.0/generations_t1.0.json`

All reported D0-D5 values above were produced directly from the validated JSON artifact.

## Limitations

This is a small fixed-prompt decoding experiment with eight prompts.

It does not measure factuality, broad language-model quality, downstream task performance, or hallucination rate.

Generation was performed on CPU. These outputs should therefore be described as CPU decoding results rather than GPU generation results.

BPB remains NOT VERIFIED because the original validation text/token stream required to reproduce the metric is not currently available in the recovered evaluation artifacts.
