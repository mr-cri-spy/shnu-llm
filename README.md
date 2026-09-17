# SHNU-LLM

A from-scratch language model research project.

**SHNU-LLM is initialized randomly and pretrained from scratch; pretrained LLM weights are not used.** The architecture, tokenizer, training process, and final weights are all produced within this repository. Open-source *software* (PyTorch, HuggingFace `tokenizers`, HuggingFace `datasets`) is used, but no pretrained language-model checkpoint of any kind.

SHNU-LLM v0.1 is a small model. Its purpose is to establish a correct, reproducible, and scalable from-scratch pretraining pipeline on modest compute — not to match large frontier models. See [Limitations](#limitations).

## Architecture

Decoder-only (GPT-style) Transformer implemented in `src/shnu_llm/model.py`:

| Component | Choice |
|---|---|
| Positional encoding | Rotary embeddings (RoPE) — no learned position parameters |
| Normalization | RMSNorm, pre-norm |
| Attention | Causal multi-head self-attention; explicit Q/K/V/O projections; optional grouped-query attention; `scaled_dot_product_attention` with `is_causal=True` |
| Feed-forward | SwiGLU |
| Residuals | Pre-norm residual connections; scaled initialization on residual projections |
| Output head | Weight-tied with the token embedding |
| Objective | Causal next-token cross-entropy |

## Model parameters

Configs live in `configs/`. Parameter counts are measured, not estimated.

| Config | Layers | d_model | Heads | Context | Vocab | Parameters |
|---|---|---|---|---|---|---|
| `configs/smoke.json` | 4 | 128 | 4 | 128 | 4,000 | 1,365,120 |
| `configs/t4.json` | 8 | 512 | 8 | 512 | 16,000 | 33,890,816 |
| `configs/v0.2.json` | 8 | 512 | 8 | 512 | 16,000 | 33,890,816 |

## Tokenizer

Byte-level BPE trained from scratch on the project corpus (`tokenizer/train_tokenizer.py`, HuggingFace `tokenizers`). Special tokens: `<pad> <unk> <bos> <eos>`. Byte-level coverage means no true out-of-vocabulary tokens. The TEXT → IDs → TEXT round-trip is asserted during training and in the test suite.

## Dataset

- **Primary (v0.2):** WikiText-103-raw-v1, human-written Wikipedia text — license **CC BY-SA 3.0**, loaded via HuggingFace `datasets` (`Salesforce/wikitext`, non-streaming).
- **Fallback:** Tiny Shakespeare (public domain) when the dataset hub is unreachable.

Full provenance, acquisition, and the preparation pipeline: [`docs/DATASET.md`](docs/DATASET.md).

Preprocessing: newline normalization; removal of empty, too-short, and exact-duplicate records; train/validation split with no overlap to avoid leakage. Third-party text retains its own license; this project does not claim ownership of it.

## Pretraining objective

Causal language modeling: predict token *t+1* from tokens *1..t* with cross-entropy loss. Training uses AdamW (decoupled weight decay on 2-D parameters only), a cosine learning-rate schedule with linear warmup, gradient clipping, gradient accumulation, and mixed precision on GPU.

## Training infrastructure

`scripts/train.py` runs the full pipeline: corpus load → clean → tokenizer → pack → model → pretrain → export. Checkpoints store model, optimizer, scheduler step, losses, config, and RNG state. Training **auto-resumes** from `checkpoint_latest.pt`, so a re-run after an interruption continues rather than restarting.

```bash
pip install -r requirements.txt
python scripts/train.py --config configs/smoke.json   # ~1-minute self-test
python scripts/train.py --config configs/t4.json      # ~34M-parameter run
```

## Hardware

Targets a single consumer/free GPU (e.g. an NVIDIA T4, 16 GB). CUDA is auto-detected; on Turing GPUs training uses fp16 with a gradient scaler (bf16 where supported). Micro-batches plus gradient accumulation keep memory within budget. CPU is supported for the smoke config.

## Evaluation

Reported from the run (`runs/<name>/evaluation/final_report.json`): parameter count, vocabulary, context length, dataset and license, training tokens and steps, training and validation loss, perplexity, GPU, training time, and throughput. Methodology: [`docs/EVALUATION.md`](docs/EVALUATION.md).

## Results

v0.1 reference run (single NVIDIA T4, bounded proof-of-concept): 33,890,816 parameters; initial loss 9.81 ≈ ln(16000)=9.68 (confirms correct random initialization); training loss 9.78 → 4.26; validation loss 5.29; perplexity ≈ 198. Full details and sample generations: [`docs/FINAL_REPORT.md`](docs/FINAL_REPORT.md). These numbers are specific to the corpus, tokenizer, and step budget of that run and are not comparable across datasets.

## v0.2 status (in progress — not yet trained)

v0.2 moves to the larger WikiText-103-raw corpus (~117M training tokens) with a 16,000
BPE tokenizer, and adds durable Google Drive persistence so a run survives free-tier
disconnects. The dataset and tokenizer are prepared and hash-verified; the engineering
for training, resume, safety, and evaluation is implemented and tested.

**Validation status:**

- The T4 pilot **has been run and passed** on a free-Colab Tesla T4: random init confirmed
  (initial loss ≈ ln(vocab)), loss decreased with no NaN/Inf, checkpoint save→reload→resume
  verified, measured throughput ≈ 9,940 tok/s and peak GPU memory ≈ 5.28 GB. See
  [`docs/TRAINING_PLAN.md`](docs/TRAINING_PLAN.md).
- v0.2 has **not** been trained. The pilot is a short pre-flight (a few dozen steps); no
  full v0.2 weights or final loss/perplexity numbers exist yet, and the long run is a
  separate, deliberately authorized step.

The deterministic training plan (tokens/step, recommended budget, checkpoint cadence,
storage), the fresh-runtime recovery procedure, and the safety gates are documented in
[`docs/TRAINING_PLAN.md`](docs/TRAINING_PLAN.md) and [`docs/RECOVERY.md`](docs/RECOVERY.md).
A crash-safe, checkpoint-authoritative training monitor writes a durable
`training_status.json` on Drive and is inspectable with `python scripts/status.py` — see
[`docs/MONITORING.md`](docs/MONITORING.md).
Long training is launched only through the guarded entrypoint `scripts/train_v0_2.py`,
which runs every safety gate and still refuses to train without `--confirm-long-run`.
For multi-session Colab runs there is a one-cell launcher/resumer
(`scripts/colab_launch.py`) that pulls the latest code, verifies it, mounts Drive, runs
the gates, reconciles the status, and resumes from the newest checkpoint — safe to
re-run after any disconnect; see [`docs/LAUNCHER.md`](docs/LAUNCHER.md).

The run can also continue on **other legitimate free GPUs** (e.g. Kaggle) through the
same shared launch core, one checkpoint format, and one resume path, so a session on
any platform resumes from the same newest valid checkpoint. Kaggle (which cannot mount
Google Drive) moves the checkpoint as a checksum-verified bundle that can never
overwrite a newer or valid checkpoint with an older or corrupt one. A safe benchmark
(`scripts/benchmark.py`) measures a runtime's throughput without ever touching the real
checkpoint. See [`docs/FREE_GPU_OPTIONS.md`](docs/FREE_GPU_OPTIONS.md) and
[`docs/MULTI_PLATFORM_TRAINING.md`](docs/MULTI_PLATFORM_TRAINING.md).

## Limitations

SHNU-LLM v0.1 is a small model trained on a small amount of text for a short schedule. It is **not** comparable to large frontier language models and will not answer general questions reliably. It produces locally fluent but often globally incoherent text, has no instruction-following, alignment, or factual grounding, and reflects the characteristics of its training corpus. It is a foundation and a pipeline, not a product.

## Roadmap

- **v0.1** — from-scratch pipeline, ~34M-parameter reference run *(complete)*
- **v0.2** — WikiText-103-raw corpus, 16K tokenizer, Google Drive persistence, deterministic eval suite and safety gates *(engineering complete; training not yet run)*
- **v0.3** — larger model as compute allows
- **v0.4** — improved tokenizer and data pipeline
- **v0.5** — training-stability and schedule tuning, standardized eval harness
- **v1.0** — larger-scale pretraining; a separate instruction-tuned line

## Reproduction

Deterministic seed (`seed=1337`) is set for Python, NumPy, and PyTorch; minor nondeterminism remains from GPU kernels. `notebooks/SHNU_LLM_From_Scratch.ipynb` runs the same pipeline top-to-bottom on Colab (`Runtime → T4 GPU → Run all`). Run tests with `pytest -q`.

## Repository layout

```
SHNU-LLM/
├── README.md
├── LICENSE
├── pyproject.toml
├── requirements.txt
├── src/shnu_llm/        # library: config, model, tokenizer, data, training, evaluation, preflight,
│                        #   status, platform, launcher_core, checkpoint_safety, benchmark
├── scripts/             # train.py, prepare_data.py, pilot.py, resume.py, train_v0_2.py, training_plan.py,
│                        #   colab_launch.py, kaggle_launch.py, benchmark.py, status.py
├── tokenizer/           # tokenizer training CLI
├── evaluation/          # deterministic evaluation CLI + fixed prompt spec
├── configs/             # smoke.json, t4.json, v0.2.json
├── notebooks/           # SHNU_LLM_From_Scratch.ipynb
├── tests/               # smoke, recovery, evaluation, and preflight tests
└── docs/                # DATASET, TRAINING_PLAN, RECOVERY, MONITORING, EVALUATION, FINAL_REPORT,
                         #   LAUNCHER, FREE_GPU_OPTIONS, MULTI_PLATFORM_TRAINING
```

Trained checkpoints and large model files are kept out of version control (see `.gitignore`); store them in Google Drive or a model registry.

## License

Code: MIT (see `LICENSE`). Trained weights and tokenizer are produced by this project. Training text retains each source's license (WikiText-103: CC BY-SA 3.0).
