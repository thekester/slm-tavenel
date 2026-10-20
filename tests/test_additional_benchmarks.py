"""Supplementary benchmark checks using synthetic files and mocked processes."""

import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import xml.etree.ElementTree as ET

import pytest

from slm.benchmarks import convert
from slm.runlog import sha256_file


@pytest.fixture
def scripts(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module


def test_arc_challenge_and_winogrande_conversion():
    arc = [{"id": "q", "question": "Which?", "answerKey": "B",
            "choices": {"label": ["A", "B"], "text": ["one", "two"]}}]
    assert convert("arc_challenge", arc) == convert("arc_easy", arc)
    item = convert("winogrande", [{"sentence": "The box held the ball because _ was small.",
                                  "option1": "the box", "option2": "the ball", "answer": "2"}])[0]
    assert item["context"] == "The box held the ball because"
    assert item["choices"] == [" the box was small.", " the ball was small."]
    assert item["label"] == 1


@pytest.mark.parametrize("sentence,answer", [("No blank", "1"), ("_ and _", "1"), ("_ ran.", "")])
def test_winogrande_rejects_unlabeled_or_malformed_items(sentence, answer):
    with pytest.raises(ValueError):
        convert("winogrande", [{"sentence": sentence, "answer": answer, "option1": "a", "option2": "b"}])


def test_overlap_audit_counts_all_documents_and_detects_changed_benchmarks(tmp_path, monkeypatch, scripts):
    monkeypatch.chdir(tmp_path)
    benchmark = Path("data/benchmarks/example.jsonl")
    benchmark.parent.mkdir(parents=True)
    text = "one two three four five six seven eight nine ten eleven twelve thirteen"
    benchmark.write_text(json.dumps({"id": "q", "context": text, "choices": [" a", " b"], "label": 0}) + "\n")
    train = Path("train.jsonl")
    train.write_text("".join(json.dumps({"id": str(i), "text": value}) + "\n"
                             for i, value in enumerate(["unrelated", text, "another document"])))
    audit = scripts("audit_fsc_additional_contamination")
    monkeypatch.setattr(audit, "BENCHMARKS", [str(benchmark)])
    monkeypatch.setattr(audit, "TRAIN", [str(train)])
    audit.main()
    audit_path = Path("runs/fsc-additional/contamination-audit.json")
    report = json.loads(audit_path.read_text())
    assert report["documents_scanned"] == 3
    assert report["benchmarks"][0]["matched_train_documents"] == 1
    assert report["benchmarks"][0]["matched_evaluation_items"] == 1
    evaluator = scripts("evaluate_fsc_additional")
    monkeypatch.setattr(evaluator, "TASKS", ("example",))
    assert evaluator.validate_audit(audit_path)["sha256"] == sha256_file(audit_path)
    benchmark.write_text(benchmark.read_text() + "\n")
    with pytest.raises(ValueError, match="differs"):
        evaluator.validate_audit(audit_path)


def test_additional_chart_preserves_readme_and_rejects_bad_inputs(tmp_path, monkeypatch, scripts):
    module = scripts("render_fsc_additional_benchmarks")
    monkeypatch.chdir(tmp_path)
    readme = Path("README.md")
    readme.write_text("Before\n" + module.BEGIN + "\nPending\n" + module.END + "\nAfter")
    record = {"model": {"model_params": 1000000},
              "benchmarks": [{"path": "task.jsonl", "sha256": "abc"}],
              "results": {name: {"acc": chance, "acc_stderr": 0.01} for name, _, chance in module.TASKS}}
    paths = []
    for name in module.MODELS[:2]:
        path = Path("runs/fsc-additional") / name / "fsc-pilot-v2-mc.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(record))
        paths.append(path)
    module.main()
    output = Path("docs/fsc-additional-benchmark-comparison.svg")
    ET.parse(output)
    svg, published = output.read_text(encoding="utf-8"), readme.read_text(encoding="utf-8")
    assert "chance 25%" in svg and "chance 50%" in svg
    assert published.startswith("Before\n") and published.endswith("\nAfter")
    assert "](runs/" not in published
    readme.write_text("Missing markers")
    with pytest.raises(ValueError, match="markers"):
        module.main()
    assert output.read_text(encoding="utf-8") == svg
    readme.write_text(published, encoding="utf-8")
    record["benchmarks"][0]["sha256"] = "different"
    paths[1].write_text(json.dumps(record))
    with pytest.raises(ValueError, match="mismatch"):
        module.main()
    assert readme.read_text(encoding="utf-8") == published


def test_queue_fetches_before_auditing_and_does_not_refetch_before_scoring(tmp_path):
    shell = shutil.which("powershell") or shutil.which("pwsh")
    if not shell:
        pytest.skip("PowerShell is not installed")
    queue = Path(__file__).resolve().parents[1] / "scripts/queue_fsc_additional_benchmarks.ps1"
    for name in ("pythia-14m", "smollm2-135m", "tiny1m"):
        path = tmp_path / f"runs/reference/{name}/fsc-pilot-v1-mc.json"
        path.parent.mkdir(parents=True)
        path.write_text("{}")
    command = r'''
$ErrorActionPreference = 'Stop'
$global:commands = [System.Collections.Generic.List[string]]::new()
function Get-CimInstance { @() }
function python {
  $global:commands.Add(($args -join ' '))
  $global:LASTEXITCODE = 0
  if ($args[0] -eq '-c') { 'mock-lib' }
}
& $env:FSC_TEST_QUEUE
$global:commands | ConvertTo-Json | Set-Content commands.json -Encoding utf8
'''
    result = subprocess.run([shell, "-NoProfile", "-Command", command], cwd=tmp_path,
                            env={**os.environ, "FSC_TEST_QUEUE": str(queue)}, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    commands = json.loads((tmp_path / "commands.json").read_text(encoding="utf-8-sig"))
    assert "--fetch-only" in commands[1]
    assert "audit_fsc_additional_contamination.py" in commands[2]
    assert "--skip-fetch" in commands[3]
