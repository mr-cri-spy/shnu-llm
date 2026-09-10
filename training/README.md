# Training

Pretraining is driven by [`../scripts/train.py`](../scripts/train.py), which wires together
the library in `../src/shnu_llm/`.

```bash
python ../scripts/train.py --config ../configs/t4.json
```

## Checkpointing and recovery

Checkpoints are written under `runs/<name>/checkpoints/` and contain the model state,
optimizer state, current step, losses, configuration, and RNG state. Training resumes
automatically from `checkpoint_latest.pt`, so re-running after an interruption continues
from the last saved step instead of restarting.

Checkpoints and exported weights are intentionally excluded from version control (see
`../.gitignore`). Store large artifacts in Google Drive or a model registry. To persist
checkpoints to Google Drive from Colab, mount Drive and point `out_dir` at the mounted path.

## Reproducibility

A fixed seed (1337) is set for Python, NumPy, and PyTorch. Every run records its environment
and configuration in `runs/<name>/evaluation/final_report.json`.
