"""Offline release checks; no production jobs or external downloads."""

import importlib
import json
import shutil
import subprocess
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
    readme = tmp_path / "README.md"
    prefix, suffix = "Keep this introduction.\n", "\nKeep this conclusion."
    readme.write_text(prefix + "<!-- BEGIN FSC SCALING BENCHMARK RESULTS -->\nPending\n<!-- END FSC SCALING BENCHMARK RESULTS -->" + suffix)
    paths = [tmp_path / f"{i}.json" for i in range(2)]
    result = {"benchmarks": [{"path": "demo.jsonl", "sha256": "hash"}],
              "results": {task: {"acc": 0.9, "acc_stderr": 0.01} for task, _ in module.TASKS}}
    for path in paths:
        path.write_text(json.dumps(result))
    monkeypatch.setattr(module, "MODELS", [("A", paths[0], "#000000"), ("B", paths[1], "#ffffff"),
                                         ("Unavailable", tmp_path / "missing.json", "#ff0000")])
    output = tmp_path / "chart.svg"
    monkeypatch.setattr("sys.argv", ["render", "--out", str(output), "--readme", str(readme)])
    module.main()
    ET.parse(output)
    text = output.read_text()
    assert "100%" in text and "All cross-model scores are exploratory" in text
    assert "different training data and compute" in text
    published = readme.read_text(encoding="utf-8")
    assert published.startswith(prefix) and published.endswith(suffix)
    assert "| A |" in published and "| B |" in published and "Unavailable" not in published
    assert "(chart.svg)" in published and "](runs/" not in published
    module.main()
    assert readme.read_text(encoding="utf-8") == published
    result["benchmarks"][0]["sha256"] = "different"
    paths[1].write_text(json.dumps(result))
    with pytest.raises(ValueError, match="hashes differ"):
        module.main()
    assert readme.read_text(encoding="utf-8") == published
    assert output.read_text() == text


@pytest.mark.parametrize("markers", ["no markers", "<!-- END FSC SCALING BENCHMARK RESULTS -->\n<!-- BEGIN FSC SCALING BENCHMARK RESULTS -->"])
def test_chart_rejects_invalid_readme_before_writing(tmp_path, monkeypatch, scripts, markers):
    module = scripts("render_fsc_benchmark_chart")
    readme, output = tmp_path / "README.md", tmp_path / "chart.svg"
    readme.write_text(markers)
    output.write_text("existing chart")
    monkeypatch.setattr("sys.argv", ["render", "--out", str(output), "--readme", str(readme)])
    with pytest.raises(ValueError, match="markers"):
        module.main()
    assert readme.read_text() == markers
    assert output.read_text() == "existing chart"


@pytest.mark.parametrize("completed", [["1m"], ["1m", "10m"]])
def test_queue_only_compares_available_local_runs(tmp_path, completed):
    powershell = shutil.which("powershell") or shutil.which("pwsh")
    if not powershell:
        pytest.skip("PowerShell is not installed")
    queue = Path(__file__).resolve().parents[1] / "scripts/evaluate_fsc_scaling.ps1"
    for name in completed:
        checkpoint = tmp_path / f"runs/fsc-scaling/{name}-gated-s1234/checkpoints/step_00015000.pt"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.touch()
    command = r'''
$ErrorActionPreference = 'Stop'
$global:commands = [System.Collections.Generic.List[string]]::new()
function python {
  $global:commands.Add(($args -join ' '))
  $global:LASTEXITCODE = 0
  if ($args[0] -eq '-c') { 'mock-pytorch-lib' }
}
& $env:FSC_TEST_QUEUE
$global:commands | ConvertTo-Json | Set-Content -LiteralPath 'commands.json' -Encoding utf8
'''
    import os
    result = subprocess.run([powershell, "-NoProfile", "-Command", command], cwd=tmp_path,
                            env={**os.environ, "FSC_TEST_QUEUE": str(queue)}, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    commands = json.loads((tmp_path / "commands.json").read_text(encoding="utf-8-sig"))
    comparisons = [command for command in commands if "scripts/compare_mc.py" in command]
    assert len(comparisons) == (1 if len(completed) == 1 else 4)
    assert all("25m" not in command for command in comparisons)
    if len(completed) == 1:
        assert "tiny1m" in comparisons[0] and "10m" not in comparisons[0]
