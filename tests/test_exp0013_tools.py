"""Validate the queued pilot with synthetic inputs, without starting training."""

import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest

from test_exp0012_data import write_tokens


@pytest.fixture
def pilot(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    module = importlib.import_module("build_exp0013_1m_mixture")
    monkeypatch.setattr(module, "ROOT", tmp_path)
    fsc, mot, out = [tmp_path / name for name in ("fsc", "mot", "out")]
    record = write_tokens(fsc, [1, 1, 1, 0] * 5)
    record["splits"]["test"] = record["splits"]["val"]
    (fsc / "manifest.json").write_text(json.dumps(record))
    write_tokens(mot, [2, 2, 0, 3, 3, 0])
    tokenizer = tmp_path / "data/tokenizers/fsc-cc0-open-weights-v4/manifest.json"
    tokenizer.parent.mkdir(parents=True)
    tokenizer.write_text("{}")
    monkeypatch.setattr(sys, "argv", ["mix", "--fsc", str(fsc), "--mot", str(mot), "--out", str(out)])
    return module, fsc, mot, out


def test_mixture_preserves_heldouts_and_uses_only_complete_unique_articles(pilot):
    module, fsc, mot, out = pilot
    module.main()
    assert np.fromfile(out / "train_mot_00000.bin", dtype=np.uint16).tolist() == [2, 2, 0]
    for split in ("val", "test"):
        assert (out / f"{split}_fsc_00000.bin").read_bytes() == (fsc / "val_00000.bin").read_bytes()
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["mix"]["actual_mot_fraction"] == pytest.approx(3 / 23)
    assert manifest["mix"]["mot_repetitions"] == 1
    with pytest.raises(FileExistsError):
        module.main()


def test_mixture_rejects_corrupted_heldout_before_writing_output(pilot):
    module, fsc, mot, out = pilot
    (fsc / "val_00000.bin").write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="hash mismatch"):
        module.main()
    assert not out.exists()


def test_mixture_rejects_insufficient_unique_tokens_and_empty_stream(pilot):
    module, fsc, mot, out = pilot
    with pytest.raises(ValueError, match="Not enough unique"):
        module.end_on_document_boundary(np.array([1, 0]), 3, 0)
    with pytest.raises(ValueError, match="positive"):
        module.end_on_document_boundary(np.array([1, 0]), 0, 0)
    with pytest.raises(ValueError, match="complete document"):
        module.end_on_document_boundary(np.array([1, 0]), 1, 0)
    empty = out / "empty"
    write_tokens(empty, [])
    with pytest.raises(ValueError, match="no token shards"):
        module.read_stream(empty, "train")


def test_evaluator_resolves_inputs_from_repo_and_rejects_missing_language(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    module = importlib.import_module("evaluate_exp0013_1m")
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "FSC_SPLITS", {"fsc_dev": "fsc.jsonl"})
    monkeypatch.setattr(module, "MOT_SPLITS", {"mot_dev": "mot.jsonl"})
    monkeypatch.setattr(module, "BENCHMARKS", ["bench.jsonl"])
    for name in ("fsc.jsonl", "bench.jsonl", "checkpoint.pt"):
        (tmp_path / name).write_text("{}")
    tokenizer = tmp_path / "data/tokenizers/fsc-cc0-open-weights-v4/tokenizer.json"
    tokenizer.parent.mkdir(parents=True)
    tokenizer.write_text("{}")
    mot = tmp_path / "mot.jsonl"
    mot.write_text(''.join(json.dumps({"source": f"mot-v1.11-{lang}", "text": lang}) + "\n" for lang in ("en", "fr")))
    class Model:
        def eval(self):
            return self
        def num_params(self):
            return 10
    monkeypatch.setattr(module, "load_model", lambda *args: Model())
    monkeypatch.setattr(module.slm_tokenizer, "load", lambda path: object())
    def read_input(path):
        assert path.is_absolute() and path.exists()
        return ["synthetic"]
    monkeypatch.setattr(module, "read_texts", read_input)
    monkeypatch.setattr(module, "load_items", read_input)
    monkeypatch.setattr(module, "bits_per_byte", lambda *args: {"bpb": 1.0})
    monkeypatch.setattr(module, "multiple_choice", lambda *args, **kwargs: {"accuracy": 0.5})
    other = tmp_path / "other"
    other.mkdir()
    monkeypatch.chdir(other)
    out = tmp_path / "result.json"
    monkeypatch.setattr(sys, "argv", ["evaluate", "--checkpoint", str(tmp_path / "checkpoint.pt"), "--out", str(out), "--device", "cpu"])
    module.main()
    assert set(json.loads(out.read_text())["text_sets"]) == {"fsc_dev", "mot_dev_en", "mot_dev_fr"}
    with pytest.raises(ValueError, match="Unsupported"):
        module.mot_texts(mot, "de")
    mot.write_text(json.dumps({"source": "mot-v1.11-en", "text": "en"}) + "\n")
    with pytest.raises(ValueError, match="No fr records"):
        module.main()


def test_queue_obeys_most_recent_attempt_without_executing_queue():
    shell = shutil.which("pwsh") or shutil.which("powershell")
    if not shell:
        pytest.skip("PowerShell unavailable")
    queue = Path(__file__).resolve().parents[1] / "scripts/queue_exp0013_1m_after_exp0012.ps1"
    command = r'''
    $ErrorActionPreference = 'Stop'
    $tokens = $null; $errors = $null
    $ast = [System.Management.Automation.Language.Parser]::ParseFile($env:PILOT_QUEUE, [ref]$tokens, [ref]$errors)
    if ($errors.Count) { throw ($errors | Out-String) }
    $fn = $ast.Find({param($n) $n -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $n.Name -eq 'Get-PredecessorState'}, $true)
    Invoke-Expression $fn.Extent.Text
    $done = 'Both EXP-0012 held-out evaluations completed'
    $start = 'Starting matched FSC-only control'
    $failed = 'Control failed'
    @('', $done, "$done`n$start", "$done`n$start`n$failed", "$failed`n$start`n$done") | ForEach-Object { Get-PredecessorState $_ } | ConvertTo-Json
    '''
    result = subprocess.run([shell, "-NoProfile", "-NonInteractive", "-Command", command],
                            env={**os.environ, "PILOT_QUEUE": str(queue)}, capture_output=True, text=True, check=True)
    assert json.loads(result.stdout) == ["waiting", "complete", "waiting", "failed", "complete"]
