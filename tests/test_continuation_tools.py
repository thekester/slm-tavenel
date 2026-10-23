"""Offline checks for continuation analyses, evaluation and MIR validation."""

import copy
import importlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from slm.config import TrainConfig


@pytest.fixture
def scripts(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module


@pytest.mark.parametrize("settings", [
    {"mir_mask_ratio_min": -0.1}, {"mir_mask_ratio_max": 1.1},
    {"mir_mask_ratio_min": 0.6, "mir_mask_ratio_max": 0.5},
    {"mir_mask_ratio_min": float("nan")}, {"mir_mask_ratio_max": float("nan")},
    {"mir_aux_weight": -0.1}, {"mir_aux_weight": float("nan")},
    {"mir_aux_weight": float("inf")}, {"mir_mask_token_id": -1}, {"mir_mask_token_id": 1.5},
])
def test_mir_rejects_invalid_settings(settings):
    with pytest.raises(ValueError, match="MIR"):
        TrainConfig(**settings)


def evaluation(values):
    sizes = [10, 30]
    return {"tokenizer_sha256": "tok", "text_sets": {"fsc": {
        "bits_per_byte": sum(x * n for x, n in zip(values, sizes)) / sum(sizes),
        "per_document": [{"text_sha256": str(i), "bytes": n, "bits_per_byte": x}
                         for i, (x, n) in enumerate(zip(values, sizes))],
    }}}


@pytest.mark.parametrize("module_name,function,key", [
    ("analyze_exp0016_25m_wiki", "paired_document_bpb", "delta_treatment_minus_control_bpb"),
    ("analyze_exp0020_21_mir", "paired", "mir_minus_control_bpb"),
    ("analyze_exp0022_seed_replication", "pair", "wiki10_minus_control_bpb"),
])
def test_paired_bpb_weights_bytes_and_rejects_misaligned_documents(scripts, module_name, function, key):
    fn = getattr(scripts(module_name), function)
    control, treatment = evaluation([1.0, 2.0]), evaluation([2.0, 4.0])
    def run(a, b):
        rng = 42 if function == "paired" else np.random.default_rng(42)
        return fn(a, b, rng, 1000)
    assert run(control, treatment)["fsc"][key] == pytest.approx(1.75)
    assert run(control, treatment) == run(control, treatment)
    bad = copy.deepcopy(treatment)
    bad["text_sets"]["fsc"]["per_document"].reverse()
    with pytest.raises(ValueError):
        run(control, bad)


def test_primary_text_evaluation_does_not_run_benchmarks(tmp_path, monkeypatch, scripts):
    module = scripts("evaluate_exp0016_25m_wiki")
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "TEXT_SETS", {"fsc_en_validation": "input.jsonl", "mot_dev": "absent.jsonl"})
    for name in ("checkpoint.pt", "input.jsonl", "tokenizer.json"):
        (tmp_path / name).write_text("{}")
    class Model:
        def eval(self):
            return self
        def num_params(self):
            return 10
    monkeypatch.setattr(module, "load_model", lambda *args: Model())
    monkeypatch.setattr(module.slm_tokenizer, "load", lambda *args: object())
    monkeypatch.setattr(module, "read_texts", lambda *args: ["synthetic"])
    monkeypatch.setattr(module, "bits_per_byte", lambda *args: {"bits_per_byte": 1.5, "nll_per_token": 2.0, "bytes": 9, "tokens": 3})
    def forbidden(*args, **kwargs):
        pytest.fail("primary-text-only attempted a benchmark evaluation")
    monkeypatch.setattr(module, "load_items", forbidden)
    out = tmp_path / "out.json"
    monkeypatch.setattr(sys, "argv", ["eval", "--checkpoint", str(tmp_path / "checkpoint.pt"),
                        "--tokenizer", str(tmp_path / "tokenizer.json"), "--out", str(out),
                        "--device", "cpu", "--primary-text-only"])
    module.main()
    result = json.loads(out.read_text())
    assert result["benchmarks"] == {}
    assert set(result["text_sets"]) == {"fsc_en_validation"}
    assert result["evaluation_mode"] == "text_only_primary"


def test_hal_audit_rejects_document_license_conflicts(tmp_path, monkeypatch, scripts):
    module = scripts("audit_fsc_cc0_hal")
    source, out = tmp_path / "input.jsonl", tmp_path / "audit.json"
    source.write_text("{}")
    monkeypatch.setattr(module, "load_documents", lambda *args: [{"id": "hal-1", "language": "en"}])
    monkeypatch.setattr(module, "query_hal", lambda *args: [{"halId_s": "hal-1", "licence_s": module.CC0_URL,
                        "fileLicenses_s": ["https://creativecommons.org/licenses/by/4.0/"]}])
    monkeypatch.setattr(sys, "argv", ["audit", str(source), "--output", str(out)])
    assert module.main() == 2
    assert json.loads(out.read_text())["counts"]["license_conflict"] == 1
