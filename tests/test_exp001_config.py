"""Deterministic, CPU-only tests for the EXP-001 config and its preflight validator.

Nothing here trains, downloads, or touches a real checkpoint. Checkpoint-overwrite
cases use tiny stand-in files in a temp directory.
"""
from __future__ import annotations

import os
import sys
import copy
import json
import hashlib
import tempfile
import importlib.util

import pytest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO, "src"))
import shnu_llm as S

_spec = importlib.util.spec_from_file_location("validate_exp001",
                                               os.path.join(REPO, "scripts", "validate_exp001.py"))
V = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(V)

CFG_PATH = os.path.join(REPO, "configs", "v0.3-exp001.json")


def _cfg() -> dict:
    with open(CFG_PATH) as f:
        return json.load(f)


def _write(tmpdir, cfg) -> str:
    p = os.path.join(tmpdir, "cfg.json")
    with open(p, "w") as f:
        json.dump(cfg, f)
    return p


def _validate_dict(cfg, **kw):
    with tempfile.TemporaryDirectory() as d:
        return V.validate(_write(d, cfg), check_branch=False, count_params=False, **kw)


def _fails(cfg, needle, **kw):
    with pytest.raises(V.ValidationError) as ei:
        _validate_dict(cfg, **kw)
    assert needle in str(ei.value), str(ei.value)


# ---- the committed config -------------------------------------------------

def test_committed_config_is_valid_json_and_passes():
    summary = V.validate(CFG_PATH, check_branch=False, count_params=True)
    assert summary["changed_variables"] == ["training_corpus"]
    assert summary["parameters"] == 33890816
    assert summary["tokens_per_step"] == 32768
    assert summary["status"] == "PRE-TRAINING / NOT STARTED"


def test_branch_check_uses_expected_branch():
    V.validate(CFG_PATH, branch="v0.3-exp001", check_branch=True, count_params=False)
    with pytest.raises(V.ValidationError, match="git branch"):
        V.validate(CFG_PATH, branch="main", check_branch=True, count_params=False)


def test_shnu_config_block_matches_v02_except_identity_fields():
    from dataclasses import asdict
    v02 = asdict(S.ShnuConfig(**json.load(open(os.path.join(REPO, "configs", "v0.2.json")))))
    v03 = asdict(S.ShnuConfig(**_cfg()["shnu_config"]))
    assert {k for k in v02 if v02[k] != v03[k]} == {"version", "out_dir"}


def test_tokens_and_epoch_arithmetic():
    c = _cfg()
    tr = c["training"]
    assert tr["batch_size"] * tr["grad_accum_steps"] * tr["block_size"] == 32768
    assert tr["tokens_per_optimizer_step"] * tr["max_steps"] == tr["total_tokens_processed"] == 655360000
    ds = c["dataset"]
    wt = ds["components"][0]["train_tokens"]
    fw = ds["components"][1]["target_train_tokens"]
    assert wt + fw == ds["target_combined_train_tokens"]
    assert round(655360000 / wt, 4) == ds["expected_epochs"]["baseline_v0_2_0"]
    assert round(655360000 / (wt + fw), 4) == ds["expected_epochs"]["exp001"]


def test_thresholds_derive_from_frozen_baseline():
    c = _cfg()
    th, ev = c["evaluation"]["thresholds"], c["baseline"]["evaluation"]
    assert th["guardrail_wikitext_bpb_max"] == pytest.approx(ev["bits_per_byte"] * 1.02, abs=1e-9)
    assert th["guardrail_repetition_max"] == pytest.approx(ev["mean_repeated_ngram_frac"] * 1.1, abs=1e-9)


def test_config_records_frozen_file_hashes():
    c = _cfg()
    assert V._sha256(os.path.join(REPO, "configs", "v0.2.json")) == c["baseline"]["config_sha256"]
    assert V._sha256(os.path.join(REPO, c["evaluation"]["prompts"])) == c["evaluation"]["prompts_sha256"]
    assert V._sha256(os.path.join(REPO, c["evaluation"]["harness"])) == c["evaluation"]["harness_sha256"]


# ---- failures must be loud, never silently corrected ----------------------

def test_invalid_json_fails(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text("{not json")
    with pytest.raises(V.ValidationError, match="not valid JSON"):
        V.validate(str(p), check_branch=False, count_params=False)


def test_missing_required_field_fails():
    c = _cfg()
    del c["tokenizer"]["sha256"]
    _fails(c, "missing required field: tokenizer.sha256")


@pytest.mark.parametrize("key,value", [
    ("n_layers", 12), ("n_heads", 4), ("n_kv_heads", 4), ("d_model", 768),
    ("d_ff", 2048), ("block_size", 1024), ("vocab_size", 32000), ("tie_embeddings", False),
])
def test_architecture_change_fails(key, value):
    c = _cfg()
    c["architecture"][key] = value
    _fails(c, f"architecture.{key}")


def test_architecture_change_in_shnu_config_block_fails():
    c = _cfg()
    c["shnu_config"]["n_layers"] = 10
    _fails(c, "non-identity fields")


def test_tokenizer_sha_must_be_exact():
    c = _cfg()
    c["tokenizer"]["sha256"] = c["tokenizer"]["sha256"][:-1] + "0"
    _fails(c, "tokenizer.sha256")


def test_tokenizer_retrain_fails():
    c = _cfg()
    c["tokenizer"]["retrain"] = True
    _fails(c, "tokenizer.retrain")


def test_tokenizer_file_hash_mismatch_fails(tmp_path):
    f = tmp_path / "shnu_bpe.json"
    f.write_text("{}")
    _fails(_cfg(), "tokenizer file SHA-256", tokenizer_path=str(f))


@pytest.mark.parametrize("key,value", [
    ("seed", 42), ("max_steps", 40000), ("batch_size", 32), ("grad_accum_steps", 2), ("block_size", 256),
])
def test_training_budget_change_fails(key, value):
    c = _cfg()
    c["training"][key] = value
    _fails(c, f"training.{key}")


@pytest.mark.parametrize("path,value", [
    (("training", "lr_schedule", "lr"), 6e-4),
    (("training", "lr_schedule", "warmup_steps"), 1000),
    (("training", "lr_schedule", "min_lr_ratio"), 0.0),
    (("training", "optimizer", "weight_decay"), 0.0),
    (("training", "optimizer", "grad_clip"), 0.5),
    (("training", "optimizer", "betas"), [0.9, 0.999]),
])
def test_optimizer_and_schedule_changes_fail(path, value):
    c = _cfg()
    node = c
    for p in path[:-1]:
        node = node[p]
    node[path[-1]] = value
    _fails(c, ".".join(path))


def test_second_changed_variable_fails():
    c = _cfg()
    c["experiment"]["changed_variables"] = ["training_corpus", "learning_rate"]
    _fails(c, "may change ONLY")


def test_shnu_config_hidden_hyperparameter_change_fails():
    c = _cfg()
    c["shnu_config"]["lr"] = 1e-3
    _fails(c, "shnu_config differs from v0.2.0")


def test_unknown_shnu_config_field_fails():
    c = _cfg()
    c["shnu_config"]["hallucination_guard"] = True
    _fails(c, "does not define")


@pytest.mark.parametrize("bad_dir", ["runs/shnu_v0.2", "/kaggle/working/SHNU_LLM", "SHNU_LLM"])
def test_v02_output_dir_is_refused(bad_dir):
    c = _cfg()
    c["output"]["out_dir"] = bad_dir
    c["shnu_config"]["out_dir"] = bad_dir
    _fails(c, "v0.2.0 run directory")


def test_v02_run_name_is_refused():
    c = _cfg()
    c["output"]["run_name"] = "shnu_v0.2"
    _fails(c, "v0.2.0 run name")


def test_existing_v02_checkpoint_in_out_dir_is_refused(tmp_path, monkeypatch):
    ck = tmp_path / "checkpoints"
    ck.mkdir()
    fake = ck / "checkpoint_latest.pt"
    fake.write_bytes(b"stand-in checkpoint bytes")
    sha = hashlib.sha256(fake.read_bytes()).hexdigest()
    monkeypatch.setitem(V.FROZEN, "baseline_checkpoint_sha256", sha)
    with pytest.raises(V.ValidationError, match="is a v0.2.0 checkpoint"):
        V.validate(CFG_PATH, out_dir=str(tmp_path), check_branch=False, count_params=False)


def test_any_existing_checkpoint_in_out_dir_is_refused(tmp_path):
    ck = tmp_path / "checkpoints"
    ck.mkdir()
    (ck / "checkpoint_step_000500.pt").write_bytes(b"x")
    with pytest.raises(V.ValidationError, match="fresh, empty checkpoints directory"):
        V.validate(CFG_PATH, out_dir=str(tmp_path), check_branch=False, count_params=False)


def test_empty_out_dir_passes(tmp_path):
    V.validate(CFG_PATH, out_dir=str(tmp_path), check_branch=False, count_params=False)


def test_dataset_key_must_not_reuse_v02_key():
    c = _cfg()
    c["dataset"]["dataset_key"] = "wikitext103"
    _fails(c, "dataset.dataset_key")


def test_baseline_values_are_frozen():
    c = _cfg()
    c["baseline"]["evaluation"]["bits_per_byte"] = 1.0
    _fails(c, "baseline.evaluation.bits_per_byte")


def test_all_errors_reported_together():
    c = _cfg()
    c["training"]["seed"] = 1
    c["architecture"]["n_layers"] = 9
    with pytest.raises(V.ValidationError) as ei:
        _validate_dict(c)
    msg = str(ei.value)
    assert "training.seed" in msg and "architecture.n_layers" in msg


def test_validator_does_not_modify_config():
    before = V._sha256(CFG_PATH)
    V.validate(CFG_PATH, check_branch=False, count_params=False)
    assert V._sha256(CFG_PATH) == before


def test_cli_exit_codes(tmp_path, capsys):
    assert V.main(["--no-branch-check"]) == 0
    c = _cfg()
    c["training"]["max_steps"] = 1
    p = tmp_path / "bad.json"
    p.write_text(json.dumps(c))
    assert V.main(["--config", str(p), "--no-branch-check"]) == 1
    assert "FAIL" in capsys.readouterr().err


def test_deepcopy_does_not_leak_between_tests():
    a, b = _cfg(), _cfg()
    a["training"]["seed"] = 7
    assert b["training"]["seed"] == 1337 and copy.deepcopy(b) == _cfg()


# ---- provenance audit (2026-09-28) ------------------------------------------

def test_wikitext_train_stream_is_pinned_to_audited_hash():
    wt = _cfg()["dataset"]["components"][0]
    assert wt["name"] == "wikitext103" and wt["provenance_class"] == "frozen_from_v0_2"
    assert wt["train_npy_sha256"] == "8b5ec7cf9c41f482fe425403dcebd6508921feb1e04c244f3242b23e4e923124"
    assert wt["train_npy_size_bytes"] == 233991238 and wt["train_npy_dtype"] == "uint16"
    assert wt["train_tokens"] == 116995555
    # a .npy of 116,995,555 uint16 tokens is 233,991,110 data bytes + a 128-byte header
    assert wt["train_npy_size_bytes"] == 116995555 * 2 + 128
    assert wt["evidence"]["metadata_tokens"] == {"total": 118177328, "train": 116995555,
                                                  "val": 1181773, "val_ratio": 0.01}
    assert wt["evidence"]["eos_count_train"] + wt["evidence"]["eos_count_val"] == \
        wt["evidence"]["metadata_clean_stats"]["kept"]


def test_validation_split_is_final_one_percent():
    ev = _cfg()["dataset"]["components"][0]["evidence"]["metadata_tokens"]
    assert int(ev["total"] * ev["val_ratio"]) == ev["val"] == 1181773
    assert ev["total"] - ev["val"] == ev["train"]
    v = _cfg()["dataset"]["validation_sets"]["wikitext103_val"]
    assert v["size_bytes"] == 1181773 * 2 + 128


def test_min_chars_is_an_exp001_fineweb_parameter_not_a_v02_fact():
    c = _cfg()
    assert "min_chars" not in c["dataset"]
    pre = c["dataset"]["components"][1]["preprocessing"]
    assert pre["min_chars"] == 64 and pre["provenance_class"] == "new_in_exp001"
    assert "UNKNOWN" in c["dataset"]["components"][0]["evidence"]["min_chars_used_by_v0_2"]
    c["dataset"]["min_chars"] = 64
    _fails(c, "dataset.min_chars must not exist")


def test_provenance_sections_are_declared():
    prov = _cfg()["provenance"]
    assert "tokenizer" in prov["frozen_from_v0_2"]
    assert any("wikitext103_val" in x for x in prov["frozen_from_v0_2"])
    assert any("mixed_train_npy_sha256" in x for x in prov["new_in_exp001"])
    assert not any("wikitext103" in x for x in prov["unresolved_before_training"])
    assert any("source_export_sha256" in x for x in prov["unresolved_before_training"])


def test_unpinned_wikitext_hash_fails_validation():
    c = _cfg()
    c["dataset"]["components"][0]["train_npy_sha256"] = None
    _fails(c, "train_npy_sha256")
