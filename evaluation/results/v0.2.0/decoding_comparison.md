# SHNU-LLM v0.2.0 — decoding comparison (D0–D5)

Every number here comes from `decoding_comparison.json`. That file was produced on
Kaggle by `kaggle_bpb_decoding_v5.py` and is byte-identical to the Kaggle output
(SHA-256 `e6f7b9d71c3a5338a3c016b306406fe6d0fbec21f025e723452a7203d3f7413d`,
21,130 bytes). No training was performed.

## Supersedes `decoding_d0_d5.json`

This comparison replaces the earlier D0–D5 run recorded in `decoding_d0_d5.json`. That file is
kept for history but should no longer be cited. The two runs differ in two ways:

- **Repetition metric.** D0 (greedy) produced identical text in both runs for all 8 prompts, but
  the earlier file's repetition values match trigram repeats counted over the generated tokens only.
  For example, 37 of 58 trigrams gives 0.6379, and 2 of 16 gives 0.125. This file uses the full
  re-encoded text, as `evaluate.py` does, which gives 0.5692 and 0.1 for the same completions.
- **Sampled text (D1–D5).** Only the first prompt's completion matches in each sampled
  configuration. That is consistent with the earlier run setting the seed once per configuration
  rather than before every prompt. This run resets seed 1337 before every prompt.

## Setup

| Field | Value |
|---|---|
| Checkpoint | v0.2.0 `checkpoint_latest.pt`, step 20000, SHA-256 `9b472cf1dd4d56f893d12d58fcf68fa8a22f41458bbb71d45d030dd867cdcf7e` |
| Prompts | the 8 fixed prompts in `evaluation/eval_prompts.json`, unchanged |
| Seed | 1337, reset before every prompt |
| max_new_tokens | 60 for every configuration, including D0. `evaluate.py` uses 80, so D0 is not numerically identical to the harness greedy run. |
| EOS | `eos_id = 3` (`<eos>`) passed to every configuration, so the EOS-hit rate is measurable |
| Device | CPU, torch 2.10.0+cpu. Sampled outputs depend on the device and RNG, so the same seed on a GPU would give different text. |
| Decoding code | `src/shnu_llm/decoding.py` and the extended `ShnuLM.generate` from branch `v0.2.1-eval` (commit `5beb26a`), copied verbatim into the Kaggle script because that branch is not on GitHub |

## How each metric is computed

- **mean_repeated_ngram_frac** is `repetition_metrics(n=3)` on the full decoded text (prompt plus completion) after re-encoding, which is how `evaluate.py` computes it. It is averaged over the 8 prompts.
- **distinct-1 and distinct-2** are the unique unigram and bigram ratios over the newly generated token ids only, averaged over prompts.
- **mean_new_tokens** is the number of newly generated tokens, counting the EOS token when one is emitted.
- **eos_hit_rate** is the fraction of prompts whose generation emitted `<eos>` before reaching 60 tokens.

## Results

| Config | Settings | rep-3gram frac | distinct-1 | distinct-2 | mean new tokens | EOS-hit rate |
|---|---|---|---|---|---|---|
| D0 | greedy (temperature 0) | 0.0837 | 0.8438 | 0.8820 | 16.62 | 0.875 |
| D1 | top_k=50, T=0.7 | 0.0250 | 0.8797 | 0.9627 | 26.88 | 0.875 |
| D2 | top_p=0.9, T=0.7 (no top_k) | 0.0204 | 0.8928 | 0.9623 | 19.50 | 1.000 |
| D3 | D1 + repetition_penalty=1.2 | 0.0019 | 0.8955 | 0.9804 | 33.38 | 0.750 |
| D4 | D1 + no_repeat_ngram_size=3 | 0.0000 | 0.9104 | 0.9921 | 24.50 | 0.875 |
| D5 | D2 + repetition_penalty=1.2 + no_repeat_ngram_size=3 | 0.0000 | 0.9248 | 0.9914 | 25.62 | 0.875 |

## One-line observation per configuration

- **D0:** Seven of eight completions stop after one short sentence; the one that runs to 60 tokens ("Scientists have discovered that") falls into a "the two groups of the three groups" loop (repeated-trigram fraction 0.5692).
- **D1:** The completions are more varied than D0's, but "The city of" repeats "the Cid Sanctuary" four times.
- **D2:** All 8 completions end at EOS, several after a single short clause, and "The theory suggests that" repeats "hemisphere" five times.
- **D3:** This gives the longest outputs (33.4 new tokens on average) with near-zero repetition, but the text still drifts off topic and includes malformed spans such as "XL2786453.112xHz".
- **D4:** This produces the same text as D1 on 7 of 8 prompts; on "The city of" it removes the repeated "Cid Sanctuary" phrase, and it has no repeated trigrams at all.
- **D5:** There are no repeated trigrams and it has the highest distinct-1, but completions wander, for example from a theory to a planet's surface, and "The city of" becomes a list of interstate numbers.

## Proposed default: D4 (top_k=50, temperature=0.7, no_repeat_ngram_size=3)

**Why D4:**

- It is the smallest change from D1, the top_k=50 / T=0.7 setting already used for the published T=0.7 samples.
- It brought the measured repeated-trigram fraction from 0.0250 (D1) to 0.0000.
- It raised distinct-2 from 0.9627 to 0.9921, the highest of the six configurations.
- It only intervenes when a loop is about to form: 7 of 8 completions are identical to D1.

**Why not D3 or D5:** `repetition_penalty` lowers the logit of every token already present, including "the", ",", ".", and "of". In D3 and D5 that changed most completions and produced more topic drift and malformed number strings in this sample. That is a qualitative judgment from 8 prompts, not a measured metric.

**What improved over D0/D1:** repetition loops, as described above.

**What did not improve (the tradeoffs that remain):**

- **Early EOS dominates every configuration.** The EOS-hit rate is 0.75–1.00, and the mean completion length is 16–33 of the 60 allowed tokens. Most completions end with " . \n" followed by EOS after one sentence. A likely cause is that the training stream inserts `<eos>` after every WikiText record (roughly one line or paragraph each), so the model has learned to stop after one line. This has not been verified. No decoding setting in D0–D5 addresses it. `min_new_tokens` exists in the new decoding module but was not part of this comparison.
- **Factual and topical quality are unchanged.** All configurations produce fluent-looking but unreliable content, as expected for a 34M-parameter model.
- **The hard block has side effects.** A 3-gram block can also stop legitimate repetitions, such as names or "I @-@ 94"-style patterns.
- **The evidence is thin.** This is 8 prompts and one seed on CPU. Treat the recommendation as provisional until it is confirmed on a larger prompt set or with more seeds.

`scripts/generate.py` and `scripts/chat.py` were not modified. Wiring in the default is left to the CTO, as specified.
