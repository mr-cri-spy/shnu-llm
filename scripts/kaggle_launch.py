#!/usr/bin/env python3
"""One-call SHNU-LLM v0.2 launcher / resumer for a fresh Kaggle notebook.

Kaggle gives a free T4 (2×) or P100 but **cannot mount Google Drive**, so the
canonical checkpoint has to be moved in and out as a verified bundle. This
launcher reuses the exact same platform-independent core
(``shnu_llm.launcher_core``) and the same resume system (``scripts/resume.py``)
as the Colab launcher — there is no second checkpoint format and no second
resume path. It only adds the Kaggle-specific storage bridge around it.

Durable state on Kaggle:

  * ``tokenizer`` + ``datasets`` (large, static) — attach as a **read-only
    Kaggle Dataset** and point ``--data-root`` at it (they are copied/linked into
    the working dir once). These never change during the run.
  * the **checkpoint** — moves via a checksum-verified tar.gz bundle
    (``shnu_llm.checkpoint_safety``). Import the newest bundle at the start of a
    session; export the updated bundle at the end. Exactly one checkpoint is
    authoritative at a time, and an older/corrupt bundle can never overwrite a
    newer/valid local checkpoint.

Safety: the GPU gate fails closed on **no CUDA** (never trains on CPU); a T4 is
not required (Kaggle may hand back a P100), but a GPU is. Nothing trains unless
``--confirm-long-run`` / ``confirm_long_run=True``.
"""
from __future__ import annotations

import os
import sys
import shutil

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.join(_HERE, "..")
if os.path.join(_REPO, "src") not in sys.path:
    sys.path.insert(0, os.path.join(_REPO, "src"))

import shnu_llm as S
from shnu_llm.preflight import PreflightError
from shnu_llm.launcher_core import run_launch
from shnu_llm.platform import KAGGLE


def stage_static_assets(out_dir, data_root, *, dataset="wikitext103", vocab_size=16000, verbose=True):
    """Make the tokenizer + pre-tokenized datasets available under ``out_dir``.

    ``data_root`` is where the attached (read-only) Kaggle Dataset is mounted,
    e.g. ``/kaggle/input/shnu-llm-v02-data``. Expected layout under it:
        tokenizer/shnu_bpe.json
        datasets/{dataset}_train_v{vocab_size}.npy
        datasets/{dataset}_val_v{vocab_size}.npy
        datasets/{dataset}_metadata.json
    Files are copied into ``out_dir`` only if missing (idempotent). No checkpoint
    is touched here. Returns the list of staged relative paths.
    """
    wanted = [
        "tokenizer/shnu_bpe.json",
        f"datasets/{dataset}_metadata.json",
        f"datasets/{dataset}_train_v{vocab_size}.npy",
        f"datasets/{dataset}_val_v{vocab_size}.npy",
    ]
    staged = []
    for rel in wanted:
        src = os.path.join(data_root, rel)
        dst = os.path.join(out_dir, rel)
        if os.path.exists(dst) and os.path.getsize(dst) > 0:
            continue
        if not os.path.exists(src):
            raise PreflightError(f"required static asset missing under data_root: {rel} "
                                 f"(looked in {src})")
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
        staged.append(rel)
    if verbose and staged:
        print(f"[kaggle] staged static assets: {staged}")
    return staged


def import_bundle(archive_path, out_dir, *, cfg=None, config_path=None, verbose=True):
    """Import a checksum-verified checkpoint bundle into ``out_dir`` (fail-closed).

    Delegates to ``checkpoint_safety.import_run_bundle`` with the v0.2 config
    (unless ``cfg``/``config_path`` overrides it) so the incoming checkpoint's
    architecture is verified and the never-older-over-newer / never-corrupt-over-
    valid rules are enforced.
    """
    import json
    if cfg is None:
        config_path = config_path or os.path.join(_REPO, "configs", "v0.2.json")
        cfg = S.ShnuConfig(**json.load(open(config_path)))
    cfg.out_dir = out_dir
    return S.import_run_bundle(archive_path, out_dir, cfg=cfg, verbose=verbose)


def export_bundle(out_dir, archive_path, *, dataset="wikitext103", verbose=True):
    """Export the current checkpoint(s) + tokenizer + status as a verified bundle.

    The bundle carries a MANIFEST.json with SHA-256 + size for every file, so the
    receiving side (Colab/Drive, or the next Kaggle session) re-verifies before
    adopting anything. Never mutates ``out_dir``.
    """
    man = S.export_run_bundle(out_dir, archive_path, dataset=dataset)
    if verbose:
        print(f"[kaggle] exported bundle -> {archive_path}")
        print(f"[kaggle]   archive sha256: {man.get('archive_sha256')}")
        print(f"[kaggle]   files: {[f['arcname'] for f in man['files']]}")
    return man


def launch(out_dir="/kaggle/working/SHNU_LLM", *, data_root=None, import_from=None,
           dataset="wikitext103", config_path=None, expected=None,
           confirm_long_run=False, require_gpu_gate=True, require_t4=False,
           min_free_gb=3.0, allow_fresh=False, repo_dir=None, verbose=True):
    """Kaggle launch: (stage assets) → (import bundle) → verify → gate → reconcile
    → (optionally) resume-train. Returns the launch report dict.

    ``allow_fresh`` defaults to **False** on Kaggle: without an imported
    checkpoint the launcher refuses to silently start a brand-new run over the
    real one. Set it True only for a deliberate fresh start.
    """
    repo_dir = repo_dir or _REPO
    os.makedirs(out_dir, exist_ok=True)

    if data_root:
        stage_static_assets(out_dir, data_root, dataset=dataset, verbose=verbose)
    if import_from:
        rep = import_bundle(import_from, out_dir, verbose=verbose)
        if verbose:
            print(f"[kaggle] import report: adopted={len(rep['adopted'])} "
                  f"kept={len(rep['kept'])} rejected={len(rep['rejected'])}")

    return run_launch(out_dir, platform=KAGGLE, dataset=dataset, config_path=config_path,
                      expected=expected, confirm_long_run=confirm_long_run,
                      require_gpu_gate=require_gpu_gate, require_t4=require_t4,
                      min_free_gb=min_free_gb, drive_root=out_dir,
                      allow_fresh=allow_fresh, repo_dir=repo_dir, verbose=verbose)


def main():
    import argparse
    ap = argparse.ArgumentParser(description="SHNU-LLM v0.2 launcher / resumer (Kaggle).")
    ap.add_argument("--out-dir", default="/kaggle/working/SHNU_LLM",
                    help="durable run root on the Kaggle working disk")
    ap.add_argument("--data-root", default=None,
                    help="mount of the attached read-only Kaggle Dataset (tokenizer + datasets)")
    ap.add_argument("--import-bundle", dest="import_from", default=None,
                    help="path to a checkpoint bundle (.tar.gz) to verify and import first")
    ap.add_argument("--export-bundle", dest="export_to", default=None,
                    help="after (or instead of) launching, export the current checkpoint bundle here")
    ap.add_argument("--dataset", default="wikitext103")
    ap.add_argument("--confirm-long-run", action="store_true",
                    help="REQUIRED to actually start/resume the long training run")
    ap.add_argument("--allow-fresh", action="store_true",
                    help="permit starting from scratch when no checkpoint exists")
    ap.add_argument("--require-t4", action="store_true", default=False,
                    help="insist on a T4 specifically (default: accept any CUDA GPU)")
    ap.add_argument("--min-free-gb", type=float, default=3.0)
    ap.add_argument("--export-only", action="store_true",
                    help="only export a bundle from --out-dir to --export-bundle, then exit")
    args = ap.parse_args()

    try:
        if args.export_only:
            if not args.export_to:
                ap.error("--export-only requires --export-bundle")
            export_bundle(args.out_dir, args.export_to, dataset=args.dataset)
            return
        launch(args.out_dir, data_root=args.data_root, import_from=args.import_from,
               dataset=args.dataset, confirm_long_run=args.confirm_long_run,
               require_t4=args.require_t4, min_free_gb=args.min_free_gb,
               allow_fresh=args.allow_fresh)
        if args.export_to:
            export_bundle(args.out_dir, args.export_to, dataset=args.dataset)
    except PreflightError as e:
        print(f"PREFLIGHT FAILED: {e}\nLauncher will NOT train. Fix the issue above and retry.")
        sys.exit(2)


if __name__ == "__main__":
    main()
