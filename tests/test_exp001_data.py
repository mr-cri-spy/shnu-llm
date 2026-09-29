"""CPU-only, deterministic tests for EXP-001 data preparation, dataset-manifest
validation, the guarded launcher, and the TrainingStatus run name.

Nothing here downloads data or trains. A tiny tokenizer and tiny synthetic
"WikiText" and "FineWeb-Edu" corpora stand in for the real ones, and the frozen
hash and size values are overridden for the synthetic files only.
"""
from __future__ import annotations

import os
import sys
import json
import random
import inspect
import importlib.util
from dataclasses import asdict

import numpy as np
import pytest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO, "src"))
import shnu_llm as S
from shnu_llm import exp001_data as X


def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(REPO, "scripts", f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


T = _load("train_exp001")
R = _load("resume")
CFG_PATH = os.path.join(REPO, "configs", "v0.3-exp001.json")


# ---------------------------------------------------------------------------
# synthetic fixtures
# ---------------------------------------------------------------------------

def _words(seed, n_docs, n_words):
    rng = random.Random(seed)
    vocab = [f"w{i:02d}" for i in range(50)]
    return [" ".join(rng.choice(vocab) for _ in range(n_words)) for _ in range(n_docs)]


@pytest.fixture()
def synth(tmp_path):
    """A v0.2.0-like source dir (tokenizer + WikiText train/val streams), a FineWeb JSONL,
    and the frozen overrides that match those synthetic files."""
    src = tmp_path / "v02"
    (src / "datasets").mkdir(parents=True)
    tok_path = str(src / "tokenizer" / "shnu_bpe.json")
    wt_docs = _words(0, 120, 40)
    tok = S.train_tokenizer(iter(wt_docs + _words(1, 50, 40)), 300, tok_path)
    eos = tok.token_to_id("<eos>")
    stream = S.build_token_stream(tok, wt_docs, eos)
    train, val = S.train_val_split(stream, 0.1)
    np.save(src / "datasets" / "wikitext103_train_v16000.npy", train)
    np.save(src / "datasets" / "wikitext103_val_v16000.npy", val)
    fw = tmp_path / "fineweb.jsonl"
    fw_docs = _words(2, 200, 30)
    with open(fw, "w") as f:
        for i, t in enumerate(fw_docs):
            f.write(json.dumps({"id": f"<urn:doc:{i}>", "text": t}) + "\n")
            if i == 2:   # early, so they are reached before the targets are met
                f.write(json.dumps({"id": "<urn:doc:dup>", "text": fw_docs[0]}) + "\n")   # exact duplicate
                f.write(json.dumps({"id": "<urn:doc:short>", "text": "tiny"}) + "\n")      # too short
    tr_p, va_p = src / "datasets" / "wikitext103_train_v16000.npy", src / "datasets" / "wikitext103_val_v16000.npy"
    frozen = {
        "tokenizer_sha256": X.sha256_file(tok_path),
        "vocab_size": tok.get_vocab_size(),
        "wikitext_dtype": "uint16",
        "wikitext_train_tokens": len(train),
        "wikitext_train_sha256": X.sha256_file(str(tr_p)),
        "wikitext_train_size_bytes": os.path.getsize(tr_p),
        "wikitext_val_tokens": len(val),
        "wikitext_val_sha256": X.sha256_file(str(va_p)),
        "wikitext_val_size_bytes": os.path.getsize(va_p),
        "fineweb_target_train_tokens": len(train),
        "fineweb_heldout_target_tokens": 200,
        "fineweb_min_chars": 16,
    }
    return {"src": src, "tok_path": tok_path, "fw": str(fw), "frozen": frozen, "eos": eos,
            "train": train, "val": val, "tmp": tmp_path}


def _prepare(synth, run_dir):
    return X.prepare_exp001_data(
        wikitext_train_path=str(synth["src"] / "datasets" / "wikitext103_train_v16000.npy"),
        wikitext_val_path=str(synth["src"] / "datasets" / "wikitext103_val_v16000.npy"),
        tokenizer_path=synth["tok_path"],
        fineweb_docs=_load("prepare_exp001_data").iter_jsonl(synth["fw"], "id", "text"),
        out_datasets_dir=str(run_dir / "datasets"),
        fineweb_source={"mode": "local-jsonl", "local_file_sha256": X.sha256_file(synth["fw"])},
        frozen=synth["frozen"])


def _pinned_cfg(man, tokenizer_sha=None):
    cfg = json.load(open(CFG_PATH))
    ds = cfg["dataset"]
    comps = {c["name"]: c for c in ds["components"]}
    comps["wikitext103"]["train_npy_sha256"] = man["inputs"]["wikitext103_train"]["sha256"]
    comps["fineweb_edu"]["train_npy_sha256"] = man["files"]["fineweb_train"]["sha256"]
    comps["fineweb_edu"]["document_id_list_sha256"] = man["files"]["fineweb_train_doc_ids"]["sha256"]
    ds["mixed_train_npy_sha256"] = man["files"]["mix_train"]["sha256"]
    ds["validation_sets"]["fineweb_edu_heldout"]["sha256"] = man["files"]["fineweb_heldout"]["sha256"]
    ds["validation_sets"]["fineweb_edu_heldout"]["document_id_list_sha256"] = \
        man["files"]["fineweb_heldout_doc_ids"]["sha256"]
    comps["fineweb_edu"]["source_export_sha256"] = man["fineweb_source"]["local_file_sha256"]
    if tokenizer_sha:
        cfg["tokenizer"]["sha256"] = tokenizer_sha
    return cfg


def _synthetic_run_cfg(man, fz):
    """A config whose dataset section describes the synthetic files, for use with a
    patched FROZEN_DATA. Only numbers, hashes and file names change."""
    cfg = _pinned_cfg(man, tokenizer_sha=fz["tokenizer_sha256"])
    ds = cfg["dataset"]
    names = X.file_names(fz["dataset_key"], fz["vocab_size"])
    wt, fw = ds["components"]
    wt.update(train_tokens=fz["wikitext_train_tokens"], train_npy_sha256=fz["wikitext_train_sha256"],
              train_npy_size_bytes=fz["wikitext_train_size_bytes"])
    fw.update(target_train_tokens=fz["fineweb_target_train_tokens"], train_npy_filename=names["fineweb_train"])
    fw["preprocessing"]["min_chars"] = fz["fineweb_min_chars"]
    ds["target_combined_train_tokens"] = fz["wikitext_train_tokens"] + fz["fineweb_target_train_tokens"]
    ds["mixed_train_npy_filename"] = names["mix_train"]
    vs = ds["validation_sets"]
    vs["wikitext103_val"].update(tokens=fz["wikitext_val_tokens"], sha256=fz["wikitext_val_sha256"],
                                 size_bytes=fz["wikitext_val_size_bytes"])
    vs["training_loop_val"].update(sha256=fz["wikitext_val_sha256"], filename=names["mix_val"])
    vs["fineweb_edu_heldout"].update(target_tokens=fz["fineweb_heldout_target_tokens"],
                                     filename=names["fineweb_heldout"])
    return cfg


# ---------------------------------------------------------------------------
# TrainingStatus run identity
# ---------------------------------------------------------------------------

def _tiny_cfg(**kw):
    return S.ShnuConfig(vocab_size=64, n_layers=1, n_heads=2, d_model=16, block_size=16, **kw)


def test_status_default_run_name_unchanged(tmp_path):
    st = S.TrainingStatus.for_run(str(tmp_path))
    d = st.update(_tiny_cfg(), state="active")
    assert st.status_path.endswith(os.path.join("runs", "shnu_v0.2", "training_status.json"))
    assert d["run_name"] == "shnu_v0.2" == S.DEFAULT_RUN_NAME


def test_status_uses_exp001_run_name(tmp_path):
    st = S.TrainingStatus.for_run(str(tmp_path), run_name="shnu_v0.3_exp001")
    d = st.update(_tiny_cfg(version="v0.3"), state="active")
    assert st.status_path.endswith(os.path.join("runs", "shnu_v0.3_exp001", "training_status.json"))
    assert d["run_name"] == "shnu_v0.3_exp001"
    rec = st.reconcile(_tiny_cfg(version="v0.3"))
    assert rec["run_name"] == "shnu_v0.3_exp001"
    assert not os.path.exists(os.path.join(tmp_path, "runs", "shnu_v0.2"))
    assert "shnu_v0.3_exp001" in S.TrainingStatus.for_run(str(tmp_path / "x"), run_name="shnu_v0.3_exp001").render()


def test_resume_training_run_name_defaults_to_v02():
    sig = inspect.signature(R.resume_training)
    assert sig.parameters["run_name"].default == S.DEFAULT_RUN_NAME == "shnu_v0.2"


# ---------------------------------------------------------------------------
# config loading
# ---------------------------------------------------------------------------

def test_launcher_builds_shnu_config_from_block():
    cfg = json.load(open(CFG_PATH))
    sc = T.build_shnu_config(cfg, "/runs/exp001")
    v02 = asdict(S.ShnuConfig(**json.load(open(os.path.join(REPO, "configs", "v0.2.json")))))
    new = asdict(sc)
    assert {k for k in v02 if v02[k] != new[k]} == {"version", "out_dir"}
    assert sc.version == "v0.3" and sc.out_dir == "/runs/exp001"
    with pytest.raises(TypeError):
        S.ShnuConfig(**cfg)   # the full nested file is not a flat ShnuConfig


# ---------------------------------------------------------------------------
# data preparation
# ---------------------------------------------------------------------------

def test_prepare_builds_expected_files(synth):
    run = synth["tmp"] / "run"
    man = _prepare(synth, run)
    d = run / "datasets"
    f = man["files"]
    assert man["tokenizer"]["sha256"] == synth["frozen"]["tokenizer_sha256"]
    # mix val is a byte copy of the frozen WikiText validation stream
    assert f["mix_val"]["sha256"] == synth["frozen"]["wikitext_val_sha256"]
    assert f["mix_val"]["file"] == "wikitext103_fineweb_edu_mix_val_v%d.npy" % synth["frozen"]["vocab_size"]
    # mix = WikiText train + FineWeb train, every token exactly once
    mix = np.load(d / f["mix_train"]["file"])
    fw = np.load(d / f["fineweb_train"]["file"])
    assert len(mix) == len(synth["train"]) + len(fw) == f["mix_train"]["tokens"]
    assert np.array_equal(np.sort(mix), np.sort(np.concatenate([synth["train"], fw])))
    assert not np.array_equal(mix, np.concatenate([synth["train"], fw]))   # actually shuffled
    assert f["fineweb_train"]["tokens"] >= synth["frozen"]["fineweb_target_train_tokens"]
    assert f["fineweb_heldout"]["tokens"] >= synth["frozen"]["fineweb_heldout_target_tokens"]
    tr_ids = (d / "fineweb_edu_train_doc_ids.txt").read_text().split()
    ho_ids = (d / "fineweb_edu_heldout_doc_ids.txt").read_text().split()
    assert not set(tr_ids) & set(ho_ids)
    assert "<urn:doc:dup>" not in tr_ids + ho_ids and "<urn:doc:short>" not in tr_ids + ho_ids
    assert man["fineweb_clean_stats"]["removed_dup"] == 1 and man["fineweb_clean_stats"]["removed_short"] == 1


def test_prepare_is_deterministic(synth):
    a = _prepare(synth, synth["tmp"] / "a")
    b = _prepare(synth, synth["tmp"] / "b")
    for k in a["files"]:
        assert a["files"][k]["sha256"] == b["files"][k]["sha256"], k


def test_prepare_refuses_wrong_validation_stream(synth):
    bad = dict(synth["frozen"], wikitext_val_sha256="0" * 64)
    with pytest.raises(X.DataManifestError, match="WikiText validation SHA-256"):
        X.prepare_exp001_data(
            wikitext_train_path=str(synth["src"] / "datasets" / "wikitext103_train_v16000.npy"),
            wikitext_val_path=str(synth["src"] / "datasets" / "wikitext103_val_v16000.npy"),
            tokenizer_path=synth["tok_path"], fineweb_docs=iter([]),
            out_datasets_dir=str(synth["tmp"] / "r" / "datasets"), fineweb_source={}, frozen=bad)
    assert not (synth["tmp"] / "r" / "datasets").exists()


def test_prepare_refuses_wrong_tokenizer_and_token_counts(synth):
    bad = dict(synth["frozen"], tokenizer_sha256="1" * 64, wikitext_train_tokens=5)
    with pytest.raises(X.DataManifestError) as ei:
        X.prepare_exp001_data(
            wikitext_train_path=str(synth["src"] / "datasets" / "wikitext103_train_v16000.npy"),
            wikitext_val_path=str(synth["src"] / "datasets" / "wikitext103_val_v16000.npy"),
            tokenizer_path=synth["tok_path"], fineweb_docs=iter([]),
            out_datasets_dir=str(synth["tmp"] / "r" / "datasets"), fineweb_source={}, frozen=bad)
    assert "tokenizer SHA-256" in str(ei.value) and "WikiText train tokens" in str(ei.value)


def test_prepare_refuses_to_overwrite(synth):
    run = synth["tmp"] / "run"
    _prepare(synth, run)
    with pytest.raises(X.DataManifestError, match="refusing to overwrite"):
        _prepare(synth, run)


def test_prepare_fails_when_source_exhausted(synth):
    big = dict(synth["frozen"], fineweb_target_train_tokens=10 ** 9)
    with pytest.raises(X.DataManifestError, match="exhausted"):
        X.prepare_exp001_data(
            wikitext_train_path=str(synth["src"] / "datasets" / "wikitext103_train_v16000.npy"),
            wikitext_val_path=str(synth["src"] / "datasets" / "wikitext103_val_v16000.npy"),
            tokenizer_path=synth["tok_path"],
            fineweb_docs=_load("prepare_exp001_data").iter_jsonl(synth["fw"], "id", "text"),
            out_datasets_dir=str(synth["tmp"] / "r" / "datasets"), fineweb_source={}, frozen=big)


def test_split_and_mix_are_lossless():
    ids = np.array([5, 6, 3, 7, 3, 8, 9], dtype=np.uint16)      # eos = 3; trailing doc without eos
    assert X.split_at_eos(ids, 3) == [(0, 3), (3, 5), (5, 7)]
    m = X.mix_streams(ids, np.array([1, 3, 2, 2, 3], dtype=np.uint16), 3, 1337)
    assert sorted(m.tolist()) == sorted(ids.tolist() + [1, 3, 2, 2, 3])


# ---------------------------------------------------------------------------
# manifest validation
# ---------------------------------------------------------------------------

def test_manifest_validates_when_pinned(synth):
    run = synth["tmp"] / "run"
    man = _prepare(synth, run)
    X.validate_data_manifest(str(run / "datasets"), _pinned_cfg(man), frozen=synth["frozen"])


def test_manifest_missing_is_refused(tmp_path):
    with pytest.raises(X.DataManifestError, match="manifest missing"):
        X.validate_data_manifest(str(tmp_path), json.load(open(CFG_PATH)))


def test_unpinned_config_is_refused(synth):
    run = synth["tmp"] / "run"
    _prepare(synth, run)
    cfg = json.load(open(CFG_PATH))
    cfg["dataset"]["components"][1]["revision"] = None
    cfg["dataset"]["components"][1]["source_export_sha256"] = None
    cfg["dataset"]["components"][1]["train_npy_sha256"] = None
    cfg["dataset"]["components"][1]["document_id_list_sha256"] = None
    cfg["dataset"]["mixed_train_npy_sha256"] = None
    cfg["dataset"]["validation_sets"]["fineweb_edu_heldout"]["sha256"] = None
    cfg["dataset"]["validation_sets"]["fineweb_edu_heldout"]["document_id_list_sha256"] = None
    with pytest.raises(X.DataManifestError) as ei:
        X.validate_data_manifest(str(run / "datasets"), cfg, frozen=synth["frozen"])
    msg = str(ei.value)
    for label in ("files.mix_train", "files.fineweb_train", "files.fineweb_heldout",
                  "files.fineweb_train_doc_ids", "files.fineweb_heldout_doc_ids"):
        assert f"does not pin {label}" in msg
    assert "config pins no FineWeb-Edu source identity" in msg
    # the committed config pins the real v0.2.0 WikiText train hash, which the synthetic stream does not match
    assert "inputs.wikitext103_train: config pins 8b5ec7cf" in msg


def test_pinned_hash_mismatch_is_refused(synth):
    run = synth["tmp"] / "run"
    man = _prepare(synth, run)
    cfg = _pinned_cfg(man)
    cfg["dataset"]["mixed_train_npy_sha256"] = "f" * 64
    with pytest.raises(X.DataManifestError, match="files.mix_train: config pins"):
        X.validate_data_manifest(str(run / "datasets"), cfg, frozen=synth["frozen"])


def test_tampered_file_is_refused(synth):
    run = synth["tmp"] / "run"
    man = _prepare(synth, run)
    p = run / "datasets" / man["files"]["fineweb_heldout"]["file"]
    arr = np.load(p)
    arr[0] = (int(arr[0]) + 1) % 250
    np.save(p, arr)
    with pytest.raises(X.DataManifestError, match="fineweb_heldout: SHA-256 does not match"):
        X.validate_data_manifest(str(run / "datasets"), _pinned_cfg(man), frozen=synth["frozen"])


def test_wrong_validation_stream_in_manifest_is_refused(synth):
    run = synth["tmp"] / "run"
    man = _prepare(synth, run)
    wrong = dict(synth["frozen"], wikitext_val_sha256="a" * 64)
    with pytest.raises(X.DataManifestError, match="WikiText val SHA-256"):
        X.validate_data_manifest(str(run / "datasets"), _pinned_cfg(man), frozen=wrong)


def test_frozen_values_match_config():
    cfg = json.load(open(CFG_PATH))
    fz = X.FROZEN_DATA
    assert fz["tokenizer_sha256"] == cfg["tokenizer"]["sha256"]
    assert fz["wikitext_val_sha256"] == cfg["dataset"]["validation_sets"]["wikitext103_val"]["sha256"] \
        == "058b710795d4a8dd6b7655a635eda645d7009b2112cccd53e7712783e5eddcb8"
    assert fz["dataset_key"] == cfg["dataset"]["dataset_key"]
    assert fz["seed"] == cfg["training"]["seed"] == 1337


# ---------------------------------------------------------------------------
# launcher gates (no training: the train function is a recording stub)
# ---------------------------------------------------------------------------

def _run_dir_ready(synth, monkeypatch):
    run = synth["tmp"] / "run"
    (run / "tokenizer").mkdir(parents=True)
    (run / "tokenizer" / "shnu_bpe.json").write_bytes(open(synth["tok_path"], "rb").read())
    man = _prepare(synth, run)
    fz = dict(X.FROZEN_DATA, **synth["frozen"])
    monkeypatch.setattr(X, "FROZEN_DATA", fz)
    monkeypatch.setitem(T.V.FROZEN, "tokenizer_sha256", synth["frozen"]["tokenizer_sha256"])
    cfg_path = synth["tmp"] / "cfg.json"
    cfg_path.write_text(json.dumps(_synthetic_run_cfg(man, fz)))
    return run, str(cfg_path)


def test_launcher_preflight_passes_on_complete_run_dir(synth, monkeypatch):
    run, cfg_path = _run_dir_ready(synth, monkeypatch)
    ctx = T.preflight(cfg_path, str(run), check_branch=False)
    assert ctx["shnu_config"].version == "v0.3"
    assert ctx["shnu_config"].out_dir == str(run)


def test_launcher_requires_confirmation_and_passes_exp001_identity(synth, monkeypatch):
    run, cfg_path = _run_dir_ready(synth, monkeypatch)
    real = T.preflight
    monkeypatch.setattr(T, "preflight", lambda *a, **k: real(*a, **dict(k)))
    calls = []
    stub = lambda *a, **k: calls.append((a, k))
    base = ["--config", cfg_path, "--out-dir", str(run), "--no-branch-check"]
    assert T.main(base + ["--preflight-only"], _train_fn=stub) == 0
    assert T.main(base, _train_fn=stub) == 2
    assert calls == []
    assert T.main(base + ["--confirm-reviewed"], _train_fn=stub) == 0
    (args, kwargs), = calls
    shnu_cfg, out_dir, dataset_key = args
    assert dataset_key == "wikitext103_fineweb_edu_mix"
    assert kwargs == {"allow_fresh": True, "run_name": "shnu_v0.3_exp001"}
    assert shnu_cfg.version == "v0.3" and shnu_cfg.max_steps == 20000 and shnu_cfg.seed == 1337


def test_launcher_refuses_without_manifest(synth, monkeypatch, capsys):
    run, cfg_path = _run_dir_ready(synth, monkeypatch)
    os.remove(run / "datasets" / X.MANIFEST_FILENAME)
    calls = []
    assert T.main(["--config", cfg_path, "--out-dir", str(run), "--no-branch-check", "--confirm-reviewed"],
                  _train_fn=lambda *a, **k: calls.append(1)) == 1
    assert calls == [] and "manifest missing" in capsys.readouterr().err


def test_launcher_refuses_unpinned_config(synth, monkeypatch):
    run, cfg_path = _run_dir_ready(synth, monkeypatch)
    cfg = json.load(open(cfg_path))
    cfg["dataset"]["mixed_train_npy_sha256"] = None
    cfg["dataset"]["components"][1]["revision"] = None
    cfg["dataset"]["components"][1]["source_export_sha256"] = None
    cfg["dataset"]["components"][1]["train_npy_sha256"] = None
    cfg["dataset"]["components"][1]["document_id_list_sha256"] = None
    cfg["dataset"]["mixed_train_npy_sha256"] = None
    cfg["dataset"]["validation_sets"]["fineweb_edu_heldout"]["sha256"] = None
    cfg["dataset"]["validation_sets"]["fineweb_edu_heldout"]["document_id_list_sha256"] = None
    p = synth["tmp"] / "unpinned.json"
    p.write_text(json.dumps(cfg))
    with pytest.raises(T.LaunchRefused) as ei:
        T.preflight(str(p), str(run), check_branch=False)
    assert "does not pin files.mix_train" in str(ei.value)
    assert "no FineWeb-Edu source identity" in str(ei.value)


def test_launcher_refuses_wrong_wikitext_train_stream(synth, monkeypatch):
    run, cfg_path = _run_dir_ready(synth, monkeypatch)
    monkeypatch.setitem(X.FROZEN_DATA, "wikitext_train_sha256", "e" * 64)
    # 1) the config's pin no longer equals the frozen value -> config preflight refuses
    with pytest.raises(T.LaunchRefused, match=r"components\[wikitext103\].train_npy_sha256"):
        T.preflight(cfg_path, str(run), check_branch=False)
    # 2) config pin agrees with the frozen value, but the stream in the manifest is different
    cfg = json.load(open(cfg_path))
    cfg["dataset"]["components"][0]["train_npy_sha256"] = "e" * 64
    p = synth["tmp"] / "pinned_e.json"
    p.write_text(json.dumps(cfg))
    with pytest.raises(T.LaunchRefused, match="manifest WikiText train SHA-256"):
        T.preflight(str(p), str(run), check_branch=False)


def test_launcher_refuses_wrong_validation_stream(synth, monkeypatch):
    run, cfg_path = _run_dir_ready(synth, monkeypatch)
    monkeypatch.setitem(X.FROZEN_DATA, "wikitext_val_tokens", 1)
    with pytest.raises(T.LaunchRefused, match="wikitext103_val.tokens"):
        T.preflight(cfg_path, str(run), check_branch=False)


def test_launcher_refuses_changed_fineweb_export(synth, monkeypatch):
    run, cfg_path = _run_dir_ready(synth, monkeypatch)
    cfg = json.load(open(cfg_path))
    cfg["dataset"]["components"][1]["source_export_sha256"] = "d" * 64
    p = synth["tmp"] / "other_export.json"
    p.write_text(json.dumps(cfg))
    with pytest.raises(T.LaunchRefused, match="export SHA-256"):
        T.preflight(str(p), str(run), check_branch=False)


def test_launcher_refuses_wrong_tokenizer_file(synth, monkeypatch):
    run, cfg_path = _run_dir_ready(synth, monkeypatch)
    (run / "tokenizer" / "shnu_bpe.json").write_text("{}")
    with pytest.raises(T.LaunchRefused, match="tokenizer file SHA-256"):
        T.preflight(cfg_path, str(run), check_branch=False)


def test_launcher_refuses_incomplete_manifest(synth, monkeypatch):
    run, cfg_path = _run_dir_ready(synth, monkeypatch)
    mp = run / "datasets" / X.MANIFEST_FILENAME
    man = json.load(open(mp))
    del man["files"]["fineweb_heldout_doc_ids"]
    mp.write_text(json.dumps(man))
    with pytest.raises(T.LaunchRefused, match="manifest lacks files.fineweb_heldout_doc_ids"):
        T.preflight(cfg_path, str(run), check_branch=False)


def test_launcher_refuses_config_drift(synth, monkeypatch):
    run, cfg_path = _run_dir_ready(synth, monkeypatch)
    cfg = json.load(open(cfg_path))
    cfg["shnu_config"]["lr"] = 1e-3
    p = synth["tmp"] / "drift.json"
    p.write_text(json.dumps(cfg))
    with pytest.raises(T.LaunchRefused, match="non-identity fields"):
        T.preflight(str(p), str(run), check_branch=False)


def test_launcher_refuses_missing_tokenizer(tmp_path):
    with pytest.raises(T.LaunchRefused, match="tokenizer file not found"):
        T.preflight(CFG_PATH, str(tmp_path), check_branch=False)


def test_launcher_fresh_start_refuses_existing_checkpoint(synth, monkeypatch):
    run, cfg_path = _run_dir_ready(synth, monkeypatch)
    (run / "checkpoints").mkdir()
    (run / "checkpoints" / "checkpoint_latest.pt").write_bytes(b"x")
    with pytest.raises(T.LaunchRefused, match="fresh, empty checkpoints directory"):
        T.preflight(cfg_path, str(run), check_branch=False)


def test_launcher_resume_refuses_v02_checkpoint(synth, monkeypatch):
    import torch
    run, cfg_path = _run_dir_ready(synth, monkeypatch)
    (run / "checkpoints").mkdir()
    cfg = json.load(open(cfg_path))
    v02_like = dict(cfg["shnu_config"], version="v0.1")
    torch.save({"config": v02_like, "step": 20000, "model": {}}, run / "checkpoints" / "checkpoint_latest.pt")
    with pytest.raises(T.LaunchRefused, match="not an EXP-001 checkpoint"):
        T.preflight(cfg_path, str(run), resume=True, check_branch=False)
    sha = X.sha256_file(str(run / "checkpoints" / "checkpoint_latest.pt"))
    monkeypatch.setitem(T.V.FROZEN, "baseline_best_checkpoint_sha256", sha)
    with pytest.raises(T.LaunchRefused, match="is a v0.2.0 checkpoint"):
        T.preflight(cfg_path, str(run), resume=True, check_branch=False)


def test_launcher_resume_accepts_exp001_checkpoint(synth, monkeypatch):
    import torch
    run, cfg_path = _run_dir_ready(synth, monkeypatch)
    (run / "checkpoints").mkdir()
    cfg = json.load(open(cfg_path))
    torch.save({"config": cfg["shnu_config"], "step": 500, "model": {}}, run / "checkpoints" / "checkpoint_latest.pt")
    T.preflight(cfg_path, str(run), resume=True, check_branch=False)


def test_launcher_refuses_v02_out_dir():
    with pytest.raises(T.LaunchRefused):
        T.check_resume_state(json.load(open(CFG_PATH)), "/kaggle/working/SHNU_LLM")


def test_prepare_script_refuses_download_without_pinned_revision(synth, monkeypatch, capsys):
    P = _load("prepare_exp001_data")
    cfg = json.load(open(CFG_PATH))
    cfg["dataset"]["components"][1]["revision"] = None
    cfg["tokenizer"]["sha256"] = synth["frozen"]["tokenizer_sha256"]
    cfg_path = synth["tmp"] / "cfg.json"
    cfg_path.write_text(json.dumps(cfg))
    monkeypatch.setitem(P.V.FROZEN, "tokenizer_sha256", synth["frozen"]["tokenizer_sha256"])
    hub_called = []
    monkeypatch.setattr(P, "iter_hub", lambda fw: hub_called.append(1))
    rc = P.main(["--config", str(cfg_path), "--v02-datasets-dir", str(synth["src"] / "datasets"),
                 "--tokenizer", synth["tok_path"], "--out-dir", str(synth["tmp"] / "run"),
                 "--allow-download", "--no-branch-check"])
    assert rc == 1 and hub_called == []
    assert "revision is not pinned" in capsys.readouterr().err
    assert not (synth["tmp"] / "run" / "datasets").exists()


def test_prepare_script_refuses_wrong_tokenizer(synth, capsys):
    P = _load("prepare_exp001_data")
    rc = P.main(["--v02-datasets-dir", str(synth["src"] / "datasets"), "--tokenizer", synth["tok_path"],
                 "--out-dir", str(synth["tmp"] / "run"), "--fineweb-jsonl", synth["fw"], "--no-branch-check"])
    assert rc == 1 and "tokenizer file SHA-256" in capsys.readouterr().err


def test_prepare_refuses_wrong_wikitext_train_stream(synth):
    bad = dict(synth["frozen"], wikitext_train_sha256="c" * 64, wikitext_train_size_bytes=1)
    with pytest.raises(X.DataManifestError) as ei:
        X.prepare_exp001_data(
            wikitext_train_path=str(synth["src"] / "datasets" / "wikitext103_train_v16000.npy"),
            wikitext_val_path=str(synth["src"] / "datasets" / "wikitext103_val_v16000.npy"),
            tokenizer_path=synth["tok_path"], fineweb_docs=iter([]),
            out_datasets_dir=str(synth["tmp"] / "r" / "datasets"), fineweb_source={}, frozen=bad)
    assert "WikiText train SHA-256" in str(ei.value) and "WikiText train size" in str(ei.value)
    assert not (synth["tmp"] / "r").exists()


def test_frozen_wikitext_constants_match_config():
    cfg = json.load(open(CFG_PATH))
    wt = cfg["dataset"]["components"][0]
    assert X.FROZEN_DATA["wikitext_train_sha256"] == wt["train_npy_sha256"]
    assert X.FROZEN_DATA["wikitext_train_size_bytes"] == wt["train_npy_size_bytes"]
    assert X.FROZEN_DATA["wikitext_val_size_bytes"] == cfg["dataset"]["validation_sets"]["wikitext103_val"]["size_bytes"]
    assert X.FROZEN_DATA["fineweb_min_chars"] == cfg["dataset"]["components"][1]["preprocessing"]["min_chars"]
