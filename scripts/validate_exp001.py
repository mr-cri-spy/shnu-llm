#!/usr/bin/env python3
"""Preflight validator for EXP-001 (v0.3). Read-only: it never trains, downloads,
or edits anything.

It checks that configs/v0.3-exp001.json changes exactly one experimental variable
(the training corpus) relative to the frozen v0.2.0 run, and that nothing in the
setup could overwrite the v0.2.0 checkpoint. Every problem is reported, and the
script exits non-zero if there are any. Values are never "fixed up" silently.

Usage:
    python scripts/validate_exp001.py
    python scripts/validate_exp001.py --out-dir /path/to/durable/v0.3/run
    python scripts/validate_exp001.py --tokenizer /path/to/tokenizer/shnu_bpe.json
    python scripts/validate_exp001.py --no-branch-check   # CI / detached checkouts only
"""
from __future__ import annotations

import os
import sys
import json
import hashlib
import argparse
import subprocess
from dataclasses import asdict, fields

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO, "src"))

DEFAULT_CONFIG = os.path.join(REPO, "configs", "v0.3-exp001.json")
V02_CONFIG = os.path.join(REPO, "configs", "v0.2.json")

# Frozen v0.2.0 facts (docs/EVALUATION_v0.2.0.md). Hard-coded on purpose, so that
# editing the v0.3 config cannot move the reference point.
FROZEN = {
    "branch": "v0.3-exp001",
    "experiment_id": "EXP-001",
    "changed_variables": ["training_corpus"],
    "architecture": {
        "n_layers": 8, "n_heads": 8, "n_kv_heads": 8, "d_model": 512, "d_ff": 1408,
        "block_size": 512, "vocab_size": 16000, "tie_embeddings": True,
        "rope_theta": 10000.0, "dropout": 0.0,
    },
    "parameter_count": 33890816,
    "tokenizer_sha256": "1bf9397a6213d1a303c9d149d028fa2c456dd1a94da57c233b5b21588d1ac017",
    "seed": 1337,
    "max_steps": 20000,
    "batch_size": 16,
    "grad_accum_steps": 4,
    "block_size": 512,
    "tokens_per_step": 32768,
    "lr": 3e-4,
    "warmup_steps": 500,
    "min_lr_ratio": 0.1,
    "betas": [0.9, 0.95],
    "weight_decay": 0.1,
    "grad_clip": 1.0,
    "baseline_commit": "9b374b9f63c7b8f61323617311095e738f6cae66",
    "baseline_checkpoint_sha256": "9b472cf1dd4d56f893d12d58fcf68fa8a22f41458bbb71d45d030dd867cdcf7e",
    "baseline_best_checkpoint_sha256": "15731810e0680b47fb6ab3349baddf504a0a188ad57510f83335ed16d53975c1",
    "v02_config_sha256": "3f539b3f2403936ce90bf570b83782f5b3c60ee59d5c1c861fb3b18d07236e7d",
    "baseline_eval": {"harness_val_loss": 3.175354, "harness_perplexity": 23.9353,
                      "bits_per_byte": 1.040343, "mean_repeated_ngram_frac": 0.0963},
    "v02_out_dirs": ["runs/shnu_v0.2", "SHNU_LLM", "/kaggle/working/SHNU_LLM",
                     "/content/drive/MyDrive/SHNU_LLM"],
    "v02_run_name": "shnu_v0.2",
}

# shnu_config fields allowed to differ from v0.2.0. They are run identity, not
# experimental variables: `version` labels the run, and `out_dir` must differ so
# the v0.2.0 run directory can never be written to.
IDENTITY_FIELDS = {"version", "out_dir"}

REQUIRED = {
    "config_schema_version": (),
    "experiment": ("id", "status", "branch", "changed_variables", "specification"),
    "baseline": ("commit", "config_sha256", "checkpoint_step", "checkpoint_sha256",
                 "best_checkpoint_step", "evaluation"),
    "architecture": ("n_layers", "n_heads", "n_kv_heads", "d_model", "d_ff", "block_size",
                     "vocab_size", "tie_embeddings", "rope_theta", "dropout",
                     "expected_parameter_count"),
    "tokenizer": ("filename", "sha256", "vocab_size", "retrain"),
    "training": ("max_steps", "batch_size", "grad_accum_steps", "block_size",
                 "effective_batch_sequences", "tokens_per_optimizer_step",
                 "total_tokens_processed", "seed", "lr_schedule", "optimizer"),
    "provenance": ("frozen_from_v0_2", "new_in_exp001", "unresolved_before_training"),
    "dataset": ("training_corpus", "dataset_key", "preparation_script", "manifest_filename",
                "components", "target_combined_train_tokens",
                "mixed_train_npy_filename", "mixed_train_npy_sha256", "validation_sets",
                "download_status"),
    "launcher": ("script",),
    "evaluation": ("harness", "prompts", "seed", "eval_iters", "metrics", "thresholds"),
    "output": ("run_name", "out_dir"),
    "shnu_config": (),
}


class ValidationError(Exception):
    """Raised when the EXP-001 preflight finds one or more problems."""


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_config(path: str) -> dict:
    """Parse the config JSON. Invalid JSON raises ValidationError."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError as e:
        raise ValidationError(f"config not found: {path}") from e
    except json.JSONDecodeError as e:
        raise ValidationError(f"config is not valid JSON: {path}: {e}") from e


def current_branch(repo: str = REPO):
    try:
        out = subprocess.run(["git", "-C", repo, "rev-parse", "--abbrev-ref", "HEAD"],
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or None
    except Exception:
        return None


def check_required(cfg: dict, errors: list) -> None:
    for section, keys in REQUIRED.items():
        if section not in cfg:
            errors.append(f"missing required section/field: {section}")
            continue
        for k in keys:
            if not isinstance(cfg[section], dict) or k not in cfg[section]:
                errors.append(f"missing required field: {section}.{k}")


def _eq(errors: list, label: str, actual, expected) -> None:
    if isinstance(expected, float) or isinstance(actual, float):
        ok = isinstance(actual, (int, float)) and not isinstance(actual, bool) \
            and abs(float(actual) - float(expected)) < 1e-12
    else:
        ok = actual == expected and type(actual) is type(expected)
    if not ok:
        errors.append(f"{label} = {actual!r}, expected frozen value {expected!r}")


def check_identity(cfg: dict, errors: list, *, branch=None, check_branch: bool = True) -> None:
    exp = cfg.get("experiment", {})
    _eq(errors, "experiment.id", exp.get("id"), FROZEN["experiment_id"])
    _eq(errors, "experiment.branch", exp.get("branch"), FROZEN["branch"])
    if exp.get("changed_variables") != FROZEN["changed_variables"]:
        errors.append(f"experiment.changed_variables = {exp.get('changed_variables')!r}; "
                      f"EXP-001 may change ONLY {FROZEN['changed_variables']!r}")
    if check_branch:
        b = branch if branch is not None else current_branch()
        if b != FROZEN["branch"]:
            errors.append(f"git branch is {b!r}, expected {FROZEN['branch']!r}")


def check_architecture(cfg: dict, errors: list) -> None:
    arch = cfg.get("architecture", {})
    for k, v in FROZEN["architecture"].items():
        _eq(errors, f"architecture.{k}", arch.get(k), v)
    _eq(errors, "architecture.expected_parameter_count",
        arch.get("expected_parameter_count"), FROZEN["parameter_count"])


def check_tokenizer(cfg: dict, errors: list, tokenizer_path=None) -> None:
    tok = cfg.get("tokenizer", {})
    _eq(errors, "tokenizer.sha256", tok.get("sha256"), FROZEN["tokenizer_sha256"])
    _eq(errors, "tokenizer.vocab_size", tok.get("vocab_size"), 16000)
    if tok.get("retrain") is not False:
        errors.append(f"tokenizer.retrain = {tok.get('retrain')!r}; the v0.2.0 tokenizer must be reused")
    if tokenizer_path is not None:
        if not os.path.isfile(tokenizer_path):
            errors.append(f"tokenizer file not found: {tokenizer_path}")
        else:
            actual = _sha256(tokenizer_path)
            if actual != FROZEN["tokenizer_sha256"]:
                errors.append(f"tokenizer file SHA-256 {actual} != frozen {FROZEN['tokenizer_sha256']}")


def check_training(cfg: dict, errors: list) -> None:
    tr = cfg.get("training", {})
    for k in ("seed", "max_steps", "batch_size", "grad_accum_steps", "block_size"):
        _eq(errors, f"training.{k}", tr.get(k), FROZEN[k])
    bs, ga, blk = tr.get("batch_size"), tr.get("grad_accum_steps"), tr.get("block_size")
    if all(isinstance(x, int) for x in (bs, ga, blk)):
        tps = bs * ga * blk
        if tps != FROZEN["tokens_per_step"]:
            errors.append(f"computed tokens/step {bs}*{ga}*{blk} = {tps} != {FROZEN['tokens_per_step']}")
        _eq(errors, "training.effective_batch_sequences", tr.get("effective_batch_sequences"), bs * ga)
    _eq(errors, "training.tokens_per_optimizer_step", tr.get("tokens_per_optimizer_step"),
        FROZEN["tokens_per_step"])
    _eq(errors, "training.total_tokens_processed", tr.get("total_tokens_processed"),
        FROZEN["tokens_per_step"] * FROZEN["max_steps"])
    sch = tr.get("lr_schedule", {}) or {}
    _eq(errors, "training.lr_schedule.lr", sch.get("lr"), FROZEN["lr"])
    _eq(errors, "training.lr_schedule.warmup_steps", sch.get("warmup_steps"), FROZEN["warmup_steps"])
    _eq(errors, "training.lr_schedule.min_lr_ratio", sch.get("min_lr_ratio"), FROZEN["min_lr_ratio"])
    if sch.get("type") != "cosine_with_linear_warmup":
        errors.append(f"training.lr_schedule.type = {sch.get('type')!r}, expected 'cosine_with_linear_warmup'")
    opt = tr.get("optimizer", {}) or {}
    if opt.get("type") != "AdamW":
        errors.append(f"training.optimizer.type = {opt.get('type')!r}, expected 'AdamW'")
    if opt.get("betas") != FROZEN["betas"]:
        errors.append(f"training.optimizer.betas = {opt.get('betas')!r}, expected {FROZEN['betas']!r}")
    _eq(errors, "training.optimizer.weight_decay", opt.get("weight_decay"), FROZEN["weight_decay"])
    _eq(errors, "training.optimizer.grad_clip", opt.get("grad_clip"), FROZEN["grad_clip"])


def check_shnu_config(cfg: dict, errors: list, v02_config_path: str = V02_CONFIG) -> None:
    """The ShnuConfig block must equal the v0.2.0 effective config (configs/v0.2.json
    plus ShnuConfig defaults) in every field except the identity fields."""
    import shnu_llm as S
    if not os.path.isfile(v02_config_path):
        errors.append(f"v0.2.0 config missing: {v02_config_path}")
        return
    actual_v02_sha = _sha256(v02_config_path)
    if actual_v02_sha != FROZEN["v02_config_sha256"]:
        errors.append(f"configs/v0.2.json has changed: SHA-256 {actual_v02_sha} != frozen "
                      f"{FROZEN['v02_config_sha256']} (the v0.2.0 config must not be modified)")
    _eq(errors, "baseline.config_sha256", cfg.get("baseline", {}).get("config_sha256"),
        FROZEN["v02_config_sha256"])
    block = cfg.get("shnu_config")
    if not isinstance(block, dict):
        errors.append("shnu_config must be an object of ShnuConfig fields")
        return
    known = {f.name for f in fields(S.ShnuConfig)}
    unknown = sorted(set(block) - known)
    if unknown:
        errors.append(f"shnu_config has fields ShnuConfig does not define: {unknown}")
        return
    missing = sorted(known - set(block))
    if missing:
        errors.append(f"shnu_config must list every ShnuConfig field explicitly; missing: {missing}")
    v02 = asdict(S.ShnuConfig(**json.load(open(v02_config_path))))
    try:
        new = asdict(S.ShnuConfig(**block))
    except Exception as e:
        errors.append(f"shnu_config does not construct a ShnuConfig: {e!r}")
        return
    # the config must be explicit: nothing may be silently derived by __post_init__
    for k in ("n_kv_heads", "d_ff"):
        if block.get(k) != new[k]:
            errors.append(f"shnu_config.{k}={block.get(k)!r} but ShnuConfig resolves it to {new[k]!r}")
    diffs = sorted(k for k in v02 if v02[k] != new[k])
    unexpected = [k for k in diffs if k not in IDENTITY_FIELDS]
    if unexpected:
        errors.append("shnu_config differs from v0.2.0 in non-identity fields "
                      + ", ".join(f"{k}: v0.2.0={v02[k]!r} v0.3={new[k]!r}" for k in unexpected))
    # the section values and the shnu_config block must agree
    arch, tr = cfg.get("architecture", {}), cfg.get("training", {})
    for k in FROZEN["architecture"]:
        if k in arch and arch[k] != new.get(k):
            errors.append(f"architecture.{k}={arch[k]!r} disagrees with shnu_config.{k}={new.get(k)!r}")
    pairs = {"max_steps": "max_steps", "batch_size": "batch_size", "grad_accum_steps": "grad_accum_steps",
             "seed": "seed", "block_size": "block_size", "eval_interval": "eval_interval",
             "eval_iters": "eval_iters", "checkpoint_interval": "checkpoint_interval"}
    for sec_k, cfg_k in pairs.items():
        if sec_k in tr and tr[sec_k] != new.get(cfg_k):
            errors.append(f"training.{sec_k}={tr[sec_k]!r} disagrees with shnu_config.{cfg_k}={new.get(cfg_k)!r}")


def check_parameter_count(cfg: dict, errors: list) -> int:
    import shnu_llm as S
    block = cfg.get("shnu_config")
    try:
        n = S.expected_param_count(S.ShnuConfig(**block))
    except Exception as e:
        errors.append(f"could not build the model to count parameters: {e!r}")
        return -1
    if n != FROZEN["parameter_count"]:
        errors.append(f"model parameter count {n:,} != frozen {FROZEN['parameter_count']:,}")
    return n


def check_provenance(cfg: dict, errors: list) -> None:
    """Every hash that is still null must be declared unresolved, and nothing frozen from
    v0.2 may be declared unresolved."""
    from shnu_llm import exp001_data as X
    prov = cfg.get("provenance") or {}
    unresolved = " ".join(prov.get("unresolved_before_training") or [])
    frozen = " ".join(prov.get("frozen_from_v0_2") or [])
    for label, value in X.pinned_hashes(cfg).items():
        if label == "inputs.wikitext103_train":
            continue
        if value is None and not unresolved:
            errors.append(f"{label} is null but provenance.unresolved_before_training is empty")
    if "wikitext103" in unresolved:
        errors.append("provenance.unresolved_before_training lists WikiText-103, but its hashes are frozen")
    for needed in ("architecture", "tokenizer", "wikitext103_val", "training", "evaluation"):
        if needed not in frozen:
            errors.append(f"provenance.frozen_from_v0_2 does not list {needed!r}")


def check_dataset(cfg: dict, errors: list) -> None:
    ds = cfg.get("dataset", {})
    if ds.get("training_corpus") != "wikitext103_plus_fineweb_edu":
        errors.append(f"dataset.training_corpus = {ds.get('training_corpus')!r}, "
                      "expected 'wikitext103_plus_fineweb_edu'")
    if ds.get("dataset_key") in (None, "", "wikitext103"):
        errors.append(f"dataset.dataset_key = {ds.get('dataset_key')!r}; it must differ from the "
                      "v0.2.0 key 'wikitext103' so the mixed streams never replace the v0.2.0 files")
    from shnu_llm import exp001_data as X
    fz = X.FROZEN_DATA
    vsets = ds.get("validation_sets") or {}
    val = vsets.get("wikitext103_val") or {}
    _eq(errors, "dataset.validation_sets.wikitext103_val.tokens", val.get("tokens"), fz["wikitext_val_tokens"])
    _eq(errors, "dataset.validation_sets.wikitext103_val.sha256", val.get("sha256"), fz["wikitext_val_sha256"])
    loop_val = vsets.get("training_loop_val") or {}
    _eq(errors, "dataset.validation_sets.training_loop_val.sha256", loop_val.get("sha256"),
        fz["wikitext_val_sha256"])
    _eq(errors, "dataset.dataset_key", ds.get("dataset_key"), fz["dataset_key"])
    if "min_chars" in ds:
        errors.append("dataset.min_chars must not exist: min_chars is an EXP-001 FineWeb-Edu preprocessing "
                      "parameter (dataset.components[fineweb_edu].preprocessing.min_chars), not a v0.2 value")
    _eq(errors, "dataset.validation_sets.wikitext103_val.size_bytes", val.get("size_bytes"),
        fz["wikitext_val_size_bytes"])
    _eq(errors, "dataset.validation_sets.wikitext103_val.dtype", val.get("dtype"), fz["wikitext_dtype"])
    names = X.file_names(fz["dataset_key"], fz["vocab_size"])
    _eq(errors, "dataset.manifest_filename", ds.get("manifest_filename"), X.MANIFEST_FILENAME)
    _eq(errors, "dataset.mixed_train_npy_filename", ds.get("mixed_train_npy_filename"), names["mix_train"])
    _eq(errors, "dataset.validation_sets.training_loop_val.filename", loop_val.get("filename"), names["mix_val"])
    comps = {c.get("name"): c for c in ds.get("components") or []}
    if set(comps) != {"wikitext103", "fineweb_edu"}:
        errors.append(f"dataset.components must be exactly wikitext103 and fineweb_edu, got {sorted(comps)}")
    else:
        wt, fw = comps["wikitext103"], comps["fineweb_edu"]
        _eq(errors, "dataset.components[wikitext103].train_tokens", wt.get("train_tokens"),
            fz["wikitext_train_tokens"])
        _eq(errors, "dataset.components[wikitext103].train_npy_sha256", wt.get("train_npy_sha256"),
            fz["wikitext_train_sha256"])
        _eq(errors, "dataset.components[wikitext103].train_npy_size_bytes", wt.get("train_npy_size_bytes"),
            fz["wikitext_train_size_bytes"])
        _eq(errors, "dataset.components[wikitext103].train_npy_dtype", wt.get("train_npy_dtype"),
            fz["wikitext_dtype"])
        _eq(errors, "dataset.components[wikitext103].provenance_class", wt.get("provenance_class"),
            "frozen_from_v0_2")
        _eq(errors, "dataset.components[fineweb_edu].provenance_class", fw.get("provenance_class"),
            "new_in_exp001")
        pre = fw.get("preprocessing") or {}
        _eq(errors, "dataset.components[fineweb_edu].preprocessing.min_chars", pre.get("min_chars"),
            fz["fineweb_min_chars"])
        _eq(errors, "dataset.components[fineweb_edu].preprocessing.provenance_class",
            pre.get("provenance_class"), "new_in_exp001")
        for k in ("revision", "source_export_sha256"):
            if k not in fw:
                errors.append(f"dataset.components[fineweb_edu].{k} must be present (null until pinned)")
        _eq(errors, "dataset.components[fineweb_edu].target_train_tokens",
            comps["fineweb_edu"].get("target_train_tokens"), fz["fineweb_target_train_tokens"])
        _eq(errors, "dataset.components[fineweb_edu].train_npy_filename",
            comps["fineweb_edu"].get("train_npy_filename"), names["fineweb_train"])
        _eq(errors, "dataset.target_combined_train_tokens", ds.get("target_combined_train_tokens"),
            fz["wikitext_train_tokens"] + fz["fineweb_target_train_tokens"])
    _eq(errors, "dataset.validation_sets.fineweb_edu_heldout.target_tokens",
        (vsets.get("fineweb_edu_heldout") or {}).get("target_tokens"), fz["fineweb_heldout_target_tokens"])
    _eq(errors, "dataset.validation_sets.fineweb_edu_heldout.filename",
        (vsets.get("fineweb_edu_heldout") or {}).get("filename"), names["fineweb_heldout"])
    check_provenance(cfg, errors)
    if (cfg.get("launcher") or {}).get("script") != "scripts/train_exp001.py":
        errors.append("launcher.script must be 'scripts/train_exp001.py'")
    if (cfg.get("output") or {}).get("run_name") != "shnu_v0.3_exp001":
        errors.append(f"output.run_name = {(cfg.get('output') or {}).get('run_name')!r}, expected 'shnu_v0.3_exp001'")


def check_baseline(cfg: dict, errors: list) -> None:
    b = cfg.get("baseline", {})
    _eq(errors, "baseline.commit", b.get("commit"), FROZEN["baseline_commit"])
    _eq(errors, "baseline.checkpoint_sha256", b.get("checkpoint_sha256"), FROZEN["baseline_checkpoint_sha256"])
    _eq(errors, "baseline.checkpoint_step", b.get("checkpoint_step"), 20000)
    _eq(errors, "baseline.best_checkpoint_step", b.get("best_checkpoint_step"), 19000)
    ev = b.get("evaluation", {}) or {}
    for k, v in FROZEN["baseline_eval"].items():
        _eq(errors, f"baseline.evaluation.{k}", ev.get(k), v)


def check_no_overwrite(cfg: dict, errors: list, out_dir=None) -> None:
    """Refuse any setup that could write into, or resume from, the v0.2.0 run."""
    out = cfg.get("output", {})
    sc = cfg.get("shnu_config", {}) or {}
    norm = lambda p: os.path.normpath(str(p)).rstrip("/")
    frozen_dirs = {norm(p) for p in FROZEN["v02_out_dirs"] + list(out.get("must_not_equal") or [])}
    for label, p in (("output.out_dir", out.get("out_dir")), ("shnu_config.out_dir", sc.get("out_dir")),
                     ("--out-dir", out_dir)):
        if p is not None and norm(p) in frozen_dirs:
            errors.append(f"{label} = {p!r} is a v0.2.0 run directory; refusing (would overwrite v0.2.0)")
    if out.get("out_dir") != sc.get("out_dir"):
        errors.append(f"output.out_dir {out.get('out_dir')!r} != shnu_config.out_dir {sc.get('out_dir')!r}")
    if out.get("run_name") == FROZEN["v02_run_name"]:
        errors.append(f"output.run_name = {out.get('run_name')!r} is the v0.2.0 run name")
    if out_dir is not None:
        ck_dir = os.path.join(out_dir, "checkpoints")
        if os.path.isdir(ck_dir):
            for fn in sorted(os.listdir(ck_dir)):
                if not fn.endswith(".pt"):
                    continue
                p = os.path.join(ck_dir, fn)
                sha = _sha256(p)
                if sha in (FROZEN["baseline_checkpoint_sha256"], FROZEN["baseline_best_checkpoint_sha256"]):
                    errors.append(f"{p} is a v0.2.0 checkpoint (SHA-256 {sha[:12]}…); a v0.3 run here "
                                  "would resume from or overwrite it")
                else:
                    errors.append(f"{p} already exists; EXP-001 must start from a fresh, empty "
                                  "checkpoints directory (random init, no resume)")


def validate(cfg_path: str = DEFAULT_CONFIG, *, out_dir=None, tokenizer_path=None,
             branch=None, check_branch: bool = True, count_params: bool = True) -> dict:
    """Run every check. Raises ValidationError listing ALL problems; returns a
    summary dict when everything passes."""
    cfg = load_config(cfg_path)
    errors: list = []
    check_required(cfg, errors)
    if errors:
        raise ValidationError("\n".join(errors))
    check_identity(cfg, errors, branch=branch, check_branch=check_branch)
    check_architecture(cfg, errors)
    check_tokenizer(cfg, errors, tokenizer_path)
    check_training(cfg, errors)
    check_shnu_config(cfg, errors)
    check_dataset(cfg, errors)
    check_baseline(cfg, errors)
    check_no_overwrite(cfg, errors, out_dir)
    params = check_parameter_count(cfg, errors) if count_params and not errors else None
    if errors:
        raise ValidationError("\n".join(errors))
    return {"config": os.path.relpath(cfg_path, REPO) if cfg_path.startswith(REPO) else cfg_path,
            "experiment": cfg["experiment"]["id"], "status": cfg["experiment"]["status"],
            "changed_variables": cfg["experiment"]["changed_variables"],
            "parameters": params,
            "tokens_per_step": cfg["training"]["tokens_per_optimizer_step"],
            "branch_checked": check_branch,
            "tokenizer_file_checked": tokenizer_path is not None,
            "out_dir_checked": out_dir is not None}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="EXP-001 preflight validator (read-only; never trains).")
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--out-dir", default=None, help="durable directory the v0.3 run would write to")
    ap.add_argument("--tokenizer", default=None, help="tokenizer file to hash-check against the frozen SHA-256")
    ap.add_argument("--no-branch-check", action="store_true",
                    help="skip the git branch check (for CI or detached checkouts)")
    args = ap.parse_args(argv)
    try:
        summary = validate(args.config, out_dir=args.out_dir, tokenizer_path=args.tokenizer,
                           check_branch=not args.no_branch_check)
    except ValidationError as e:
        print("EXP-001 PREFLIGHT: FAIL", file=sys.stderr)
        for line in str(e).splitlines():
            print(f"  - {line}", file=sys.stderr)
        return 1
    if args.no_branch_check:
        print("WARNING: git branch check skipped (--no-branch-check)")
    print("EXP-001 PREFLIGHT: PASS")
    print(json.dumps(summary, indent=2))
    print("No training was started. This validator only reads files.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
