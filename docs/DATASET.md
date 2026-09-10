# SHNU-LLM v0.2 — Dataset

## Choice and rationale

v0.1 used Tiny Shakespeare (~1 MB) purely as a proof-of-concept. v0.2 trains on a
substantially larger, broad-domain, legally clear English corpus:

**WikiText-103-raw-v1** — Wikipedia "Good/Featured" articles, raw (unmodified) text.

| Criterion | Assessment |
|---|---|
| License / legal clarity | **CC BY-SA 3.0** — explicit, redistributable with attribution |
| Provenance | Curated Wikipedia articles; well-documented, widely used research corpus |
| Domain | Broad general English (encyclopedic), not single-author/archaic |
| Token count | ~100M tokens raw — enough to move a ~34M model well past the v0.1 regime |
| Free-compute feasibility | Streams/downloads free from HuggingFace; a bounded subset fits a free T4 |

It is chosen to optimize **data quality × legal clarity × token count × free-compute
feasibility** — not simply the largest corpus available. We deliberately avoid
corpora with murky provenance (e.g. raw web scrapes) for legal clarity, and avoid
multi-hundred-GB sets that cannot be trained on a free T4.

We do **not** use pretrained model weights, proprietary datasets, or paid APIs. The
corpus is third-party text under its own license; this project does not claim
ownership of it.

## Acquisition (reproducible)

`scripts/prepare_data.py` acquires the corpus:

```
python scripts/prepare_data.py --dataset wikitext103 \
    --out-dir /content/drive/MyDrive/SHNU_LLM --vocab-size 16000 --val-ratio 0.01
```

- **wikitext103** — HuggingFace `Salesforce/wikitext`, config `wikitext-103-raw-v1`,
  loaded **non-streaming**. (The deprecated `wikitext` dataset id raised an
  `hf://` streaming-URI error on newer `datasets`; using the current `Salesforce/wikitext`
  id with non-streaming is the fix.)
- **wikitext2** — a small WikiText-2 raw mirror over HTTPS, for CI / low-compute
  verification of the pipeline.
- **shakespeare** — Tiny Shakespeare (public domain), fallback only.

## Preparation pipeline

1. Acquire raw records; record source URL/id and a SHA-256 of the raw text.
2. Clean: normalize newlines; strip carriage returns and null bytes.
3. Filter: drop empty and too-short records (`--min-chars`, default 64).
4. Deduplicate: exact-duplicate records removed.
5. Deterministic train/validation split with **no overlap** (`--val-ratio`, seed-independent contiguous split).
6. Tokenize with the from-scratch byte-level BPE tokenizer (trained on the corpus itself).
7. Compute **exact** token counts after tokenization.
8. Save tokenized streams (`.npy`) and a `*_metadata.json` (source, license, date,
   SHA-256, clean stats, doc count, tokenizer vocab/hash, token counts) to Drive.

Tokenized streams and metadata live on Google Drive, **not** in Git.

## Verified pipeline run (WikiText-2 sample, for pipeline validation)

Running the pipeline on the reachable WikiText-2 raw sample produced real numbers,
confirming the pipeline end-to-end:

| Field | Value |
|---|---|
| License | CC BY-SA 3.0 |
| Raw records | 11,669 |
| Documents kept (after clean/dedup/filter) | 5,458 |
| Removed (short / duplicate) | 6,209 / 2 |
| Tokenizer vocab | 8,000 (roundtrip ✓, special tokens ✓) |
| Tokens total / train / val | 2,587,073 / 2,457,720 / 129,353 |
| Chars per token | 4.11 |

The **production** WikiText-103-raw run (vocab 16,000) is executed on Colab, where
HuggingFace is reachable; its exact token count is recorded in
`datasets/wikitext103_metadata.json` on Drive at prepare time.

## Evaluation set

A fixed held-out validation split (the `*_val_v*.npy` stream) is reused across
SHNU-LLM versions so perplexity is comparable within the same corpus + tokenizer.
Perplexity is **not** comparable across different datasets or tokenizers.
