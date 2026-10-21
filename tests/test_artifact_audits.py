"""Exercise provenance tools on synthetic artifacts without touching training data."""

import hashlib
import importlib
import io
import json
from pathlib import Path
import sys
import tarfile

import pytest


@pytest.fixture
def scripts(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module


def test_index_handles_null_provenance_external_artifacts_and_numeric_steps(tmp_path, monkeypatch, scripts):
    module = scripts("write_run_artifact_index")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.chdir(workspace)
    run = tmp_path / "external-run"
    checkpoints = run / "checkpoints"
    checkpoints.mkdir(parents=True)
    (checkpoints / "step_9.pt").write_bytes(b"older")
    checkpoint = checkpoints / "step_10.pt"
    checkpoint.write_bytes(b"checkpoint")
    (checkpoints / "step_99.pt").mkdir()
    (run / "run_manifest.json").write_text(json.dumps({"data": {"provenance": None}}))
    (run / "metrics.jsonl").write_text('{"step": 9}\n\n{"step": 10, "tokens": 100}\n')
    extra = workspace / "scores.json"
    extra.write_text("{}")
    monkeypatch.setattr(sys, "argv", ["index", str(run), "--include", str(extra), "--include", str(extra)])
    module.main()
    record = json.loads((run / "artifact_index.json").read_text())
    assert record["run"]["steps_completed"] == 10
    artifacts = {item["path"]: item for item in record["artifacts"]}
    assert len(artifacts) == 4
    assert artifacts[checkpoint.as_posix()]["sha256"] == hashlib.sha256(b"checkpoint").hexdigest()
    assert artifacts[checkpoint.as_posix()]["bytes"] == 10
    assert "scores.json" in artifacts
    assert not any("step_9.pt" in path for path in artifacts)


def test_index_rejects_missing_artifact_before_writing(tmp_path, monkeypatch, scripts):
    module = scripts("write_run_artifact_index")
    monkeypatch.chdir(tmp_path)
    (tmp_path / "checkpoints").mkdir()
    (tmp_path / "checkpoints/step_1.pt").write_bytes(b"checkpoint")
    (tmp_path / "run_manifest.json").write_text("{}")
    monkeypatch.setattr(sys, "argv", ["index", str(tmp_path), "--include", "missing.json"])
    with pytest.raises(FileNotFoundError, match="Additional artifact"):
        module.main()
    assert not (tmp_path / "artifact_index.json").exists()


def test_mot_audit_counts_metadata_and_text_defects_without_publishing_text(tmp_path, scripts):
    module = scripts("audit_mot_archives")
    path = tmp_path / "synthetic.tgz"
    private_text = "Article text that must stay out of the report."
    article = {"url": "https://example.org/a", "title": "A", "authors": ["Reuters"],
               "content_type": "news", "site_language": "en", "time_published": "2020-01-01",
               "paragraphs": [private_text, {"text": "\ufffd\u0001\n"}]}
    entries = {"eng/article/one.json": json.dumps(article).encode(),
               "eng/article/two.json": b"{}", "eng/article/broken.json": b"{",
               "eng/article/list.json": b"[]", "eng/metadata/ignored.json": b"{}"}
    with tarfile.open(path, "w:gz") as archive:
        for name, data in entries.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    report = module.audit_archive(path)
    assert report["article_records"] == 2
    assert report["malformed_article_records"] == 2
    assert report["text_character_count"] == len(private_text) + 3
    assert report["unicode_replacement_character_count"] == 1
    assert report["records_with_unicode_replacement_character"] == 1
    assert report["unexpected_control_character_count"] == 1
    assert report["missing_required_metadata_counts"] == dict.fromkeys(module.REQUIRED_METADATA, 1)
    assert report["byline_counts"] == {"Reuters": 1, "<missing>": 1}
    assert report["publication_year_counts"] == {"2020": 1, "<missing>": 1}
    assert report["agency_marker_byline_records"] == [{"record": "eng/article/one.json", "author": "Reuters"}]
    assert report["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert report["text_content_included"] is False
    assert private_text not in json.dumps(report)
