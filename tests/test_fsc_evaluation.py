"""Offline release checks; no production jobs or external downloads."""

import importlib
import json
from pathlib import Path
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import pytest


@pytest.fixture
def scripts(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module


@pytest.mark.parametrize("kind", ["checkpoint", "references"])
def test_evaluation_reuse_requires_matching_inputs(tmp_path, monkeypatch, scripts, kind):
    module = scripts(f"evaluate_fsc_{kind}")
    benchmark, validation = tmp_path / "demo.jsonl", tmp_path / "val.jsonl"
    benchmark.write_text('{"id":"q","label":0}\n')
    validation.write_text('{"id":"doc","text":"text"}\n')
    monkeypatch.setattr(module, "BENCHMARKS", [str(benchmark)])
    monkeypatch.setattr(module, "VAL_FILES", {"en": str(validation), "fr": str(validation)})
    model = SimpleNamespace(num_params=lambda: 5, cfg=SimpleNamespace(max_seq_len=256))
    calls = []
    def load(*args, **kwargs):
        calls.append((args, kwargs))
        return model if kind == "checkpoint" else (model, object())
    monkeypatch.setattr(module, "load_model" if kind == "checkpoint" else "load", load)
    monkeypatch.setattr(module, "multiple_choice", lambda *a, **kw: {
        "items": 1, "acc": 1.0, "per_item": [{"id": "q", "correct": True}]})
    monkeypatch.setattr(module, "bits_per_byte", lambda *a, **kw: {"bits_per_byte": 1.0, "tokens": 4})
    output = tmp_path / "output"
    if kind == "checkpoint":
        checkpoint, tokenizer = tmp_path / "model.pt", tmp_path / "tokenizer.json"
        checkpoint.write_bytes(b"checkpoint")
        tokenizer.write_text("{}")
        monkeypatch.setattr(module.slm_tokenizer, "load", lambda *a: object())
        evaluate = lambda: module.evaluate(checkpoint, tokenizer, output, "cpu")
    else:
        evaluate = lambda: module.evaluate("tiny1m", "cpu", output)
    evaluate()
    assert len(calls) == 1
    assert evaluate()["status"] == "already_complete"
    assert len(calls) == 1
    validation.write_text('{"id":"doc","text":"changed"}\n')
    evaluate()
    assert len(calls) == 2
    if kind == "checkpoint":
        checkpoint.write_bytes(b"new checkpoint")
    else:
        monkeypatch.setitem(module.REFERENCES, "tiny1m", {**module.REFERENCES["tiny1m"], "revision": "new"})
    evaluate()
    assert len(calls) == 3
    (output / "fsc-pilot-v1-mc.json").write_text("null")
    evaluate()
    assert len(calls) == 4


def test_chart_validates_hashes_and_supports_high_accuracy(tmp_path, monkeypatch, scripts):
    module = scripts("render_fsc_benchmark_chart")
    paths = [tmp_path / f"{i}.json" for i in range(2)]
    result = {"benchmarks": [{"path": "demo.jsonl", "sha256": "hash"}],
              "results": {task: {"acc": 0.9, "acc_stderr": 0.01} for task, _ in module.TASKS}}
    for path in paths:
        path.write_text(json.dumps(result))
    monkeypatch.setattr(module, "MODELS", [("A", paths[0], "#000000"), ("B", paths[1], "#ffffff")])
    output = tmp_path / "chart.svg"
    monkeypatch.setattr("sys.argv", ["render", "--out", str(output)])
    module.main()
    ET.parse(output)
    text = output.read_text()
    assert "100%" in text and "inherits the 10M training budget" in text
    result["benchmarks"][0]["sha256"] = "different"
    paths[1].write_text(json.dumps(result))
    with pytest.raises(ValueError, match="hashes differ"):
        module.main()
