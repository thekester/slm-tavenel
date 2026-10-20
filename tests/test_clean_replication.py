"""Offline checks for paired statistics and train/validation separation."""

import importlib.util
import json
from pathlib import Path

import pytest

from slm.data.shards import split_of
from slm.runlog import sha256_file


def script(name):
    path = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def result(correct):
    return {
        "benchmarks": [{"path": "benchmark.jsonl", "sha256": "a" * 64}],
        "results": {"demo": {"per_item": [
            {"id": str(i), "label": 0, "correct": value}
            for i, value in enumerate(correct)
        ]}},
    }


def compare(tmp_path, baseline, treatment, draws=1000):
    paths = [tmp_path / "base.json", tmp_path / "treatment.json"]
    for path, data in zip(paths, (baseline, treatment)):
        path.write_text(json.dumps(data), encoding="utf-8")
    return script("compare_mc").compare(*(str(p) for p in paths), seed=42, draws=draws)


def test_paired_comparison_counts_and_reproducibility(tmp_path):
    baseline, treatment = result([False, True, True, False]), result([True, False, True, True])
    report = compare(tmp_path, baseline, treatment)
    assert report == compare(tmp_path, baseline, treatment)
    metrics = report["benchmarks"]["demo"]
    assert metrics["treatment_only_correct"] == 2
    assert metrics["baseline_only_correct"] == 1
    assert metrics["both_same_correctness"] == 1
    assert metrics["delta_treatment_minus_baseline"] == 0.25
    improved = compare(tmp_path, result([False] * 4), result([True] * 4))
    assert improved["benchmarks"]["demo"]["paired_bootstrap_95ci"] == [1.0, 1.0]


@pytest.mark.parametrize("mismatch", ["hash", "id", "label", "missing_hashes", "empty", "draws"])
def test_paired_comparison_rejects_invalid_inputs(tmp_path, mismatch):
    baseline, treatment = result([False]), result([True])
    if mismatch == "hash":
        treatment["benchmarks"][0]["sha256"] = "b" * 64
    elif mismatch in ("id", "label"):
        treatment["results"]["demo"]["per_item"][0][mismatch] = 9
    elif mismatch == "missing_hashes":
        del baseline["benchmarks"], treatment["benchmarks"]
    elif mismatch == "empty":
        baseline["results"]["demo"]["per_item"] = []
        treatment["results"]["demo"]["per_item"] = []
    with pytest.raises(ValueError):
        compare(tmp_path, baseline, treatment, draws=0 if mismatch == "draws" else 100)


def test_tokenizer_sample_and_bpb_holdout_are_disjoint(tmp_path, monkeypatch):
    records = [
        {"id": str(i), "source": f"wikipedia-{'en' if i % 2 else 'fr'}", "text": f"Document {i}"}
        for i in range(5000)
    ]
    source = tmp_path / "documents.jsonl"
    source.write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")
    sample, holdout = script("prepare_wiki_v1_tokenizer_sample"), script("prepare_bpb_holdout")
    for module in (sample, holdout):
        monkeypatch.setattr(module, "SOURCE", source)
    monkeypatch.setattr(sample, "SAMPLE", tmp_path / "sample.jsonl")
    monkeypatch.setattr(sample, "MANIFEST", tmp_path / "sample-manifest.json")
    monkeypatch.setattr(sample, "COUNT", 40)
    monkeypatch.setattr(holdout, "OUT", tmp_path)
    monkeypatch.setattr(holdout, "DOCS_PER_LANGUAGE", 3)
    sample.main()
    holdout.main()
    paths = [sample.SAMPLE, tmp_path / "bpb-val-en.jsonl", tmp_path / "bpb-val-fr.jsonl"]
    hashes = [sha256_file(path) for path in paths]
    train_ids = {json.loads(line)["id"] for line in paths[0].read_text().splitlines()}
    assert len(train_ids) == 40
    assert all(split_of(doc_id, 0.01) == "train" for doc_id in train_ids)
    for path in paths[1:]:
        ids = {json.loads(line)["id"] for line in path.read_text().splitlines()}
        assert len(ids) == 3 and ids.isdisjoint(train_ids)
        assert all(split_of(doc_id, 0.01) == "val" for doc_id in ids)
    sample.main()
    holdout.main()
    assert hashes == [sha256_file(path) for path in paths]


@pytest.mark.parametrize("dirty", [False, True, None])
def test_source_snapshot_records_actual_revision(tmp_path, monkeypatch, dirty):
    snapshot = script("snapshot_clean_run_sources")
    manifest = tmp_path / "data/processed/wiki-v1/manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({"code": {"commit": "pipeline-commit"}}), encoding="utf-8")
    output = tmp_path / "snapshot.json"
    monkeypatch.setattr(snapshot, "ROOT", tmp_path)
    monkeypatch.setattr(snapshot, "OUT", output)
    monkeypatch.setattr(snapshot, "code_revision", lambda root: {"commit": "current-commit", "dirty": dirty})
    snapshot.main()
    report = json.loads(output.read_text())
    assert report["base_pipeline_code_commit"] == "pipeline-commit"
    assert report["code_commit"] == "current-commit"
    assert report["working_tree_dirty"] is dirty
    assert report["files_sha256"] == [{"path": manifest.as_posix(), "sha256": sha256_file(manifest)}]
