"""Offline checks for EXP-0010 provenance, scoring and publication tools."""

import gzip
import hashlib
import importlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

from slm.data.pipeline import build_step


@pytest.fixture
def scripts(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module


def test_hal_plain_and_gzip_inputs_group_by_metadata_language(tmp_path, monkeypatch, scripts):
    audit = scripts("audit_fsc_cc0_hal")
    paths = []
    for i, language in enumerate(["en", "fr", "en"]):
        path = tmp_path / (f"{i}.jsonl.gz" if i == 1 else f"{i}.jsonl")
        row = {"id": f"hal-{i}", "text": "Private source text", "source_metadata": {
            "language": language, "license": "cc0", "source": "hal"}}
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "wt", encoding="utf-8") as stream:
            stream.write(json.dumps(row) + "\n\n")
        paths.append(path)
    monkeypatch.setattr(audit, "query_hal", lambda batch, *args: [
        {"halId_s": row["id"], "licence_s": audit.CC0_URL} for row in batch])
    output = tmp_path / "filtered"
    report = tmp_path / "audit.json"
    monkeypatch.setattr(sys, "argv", ["audit", *map(str, paths), "--output", str(report),
                                     "--filtered-output-dir", str(output)])
    assert audit.main() == 0
    record = json.loads(report.read_text())
    assert record["counts"]["by_language"] == {"en": 2, "fr": 1}
    assert "Private source text" not in report.read_text()
    with gzip.open(output / "en.jsonl.gz", "rt", encoding="utf-8") as stream:
        assert [json.loads(line)["id"] for line in stream] == ["hal-0", "hal-2"]
    with pytest.raises(FileExistsError):
        audit.main()
    with pytest.raises(ValueError, match="language"):
        audit.document_language({"language": "../outside"}, paths[0])


def test_metadata_encoding_defect_is_rejected():
    step = build_step({"publication_risk_metadata": {"excluded_disciplines": [], "title_patterns": []}})
    doc = {"source_metadata": {"discipline": "Physics", "title": "Example", "author": "Name\ufffd"}}
    assert step(doc) == (None, ("replacement_character_in_source_metadata", None))


@pytest.mark.parametrize("dirty,commit", [(True, "expected"), (False, "wrong"), (False, "expected")])
def test_code_identity_guard(monkeypatch, scripts, tmp_path, dirty, commit):
    module = scripts("exp0010_integrity")
    head = module.EXPECTED_COMMIT if commit == "expected" else "other"
    monkeypatch.setattr(module.subprocess, "check_output", lambda args, **kw:
                        head if "rev-parse" in args else " M model.py" if dirty else "")
    if dirty or commit != "expected":
        with pytest.raises(ValueError, match="clean code worktree"):
            module.validate_code_root(tmp_path)
    else:
        module.validate_code_root(tmp_path)
    path = tmp_path / "artifact"
    path.write_bytes(b"original")
    expected = hashlib.sha256(b"original").hexdigest()
    module.validate_hash(path, expected)
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash differs"):
        module.validate_hash(path, expected)


def test_cached_initializations_must_be_paired(scripts):
    driver = scripts("run_exp0010_paired_replicates")
    cfgs = {arm: SimpleNamespace(attn_gate=arm) for arm in ["gated", "ungated"]}
    states = {arm: {"step": 0, "model_config": cfgs[arm].__dict__,
                    "model": {"weight": torch.ones(2)}} for arm in cfgs}
    driver.validate_initializations(states["gated"], states["ungated"], cfgs)
    states["ungated"]["model"]["weight"][0] = 0
    with pytest.raises(ValueError, match="not exactly paired"):
        driver.validate_initializations(states["gated"], states["ungated"], cfgs)


def test_document_scoring_counts_each_token_once_across_chunks(scripts):
    analysis = scripts("analyze_exp0010_paired_replicates")
    tokenizer = SimpleNamespace(token_to_id=lambda token: 0, encode=lambda text: SimpleNamespace(ids=[1] * 5))
    lengths = []
    def model(ids):
        assert not torch.is_grad_enabled()
        lengths.append(ids.shape[1])
        return torch.zeros((*ids.shape, 4)), None
    bits, nbytes = analysis.score_document(model, tokenizer, "hello", torch.device("cpu"), 2)
    assert lengths == [2, 2, 1]
    assert bits == pytest.approx(10)
    assert nbytes == 5
    assert analysis.bpb_for({("g", 1, "en", "x"): (10, 5), ("g", 1, "en", "y"): (20, 5)},
                            "g", 1, "en", ["x", "y"]) == 3


def test_figure_footer_uses_recorded_split_counts(tmp_path, monkeypatch, scripts):
    figures = scripts("render_exp0010_model_card_figures")
    manifest = tmp_path / "splits.json"
    manifest.write_text(json.dumps({"splits": [{"split": split, "language": lang, "documents": n}
        for split in ["train", "dev", "test"] for lang, n in [("en", 3), ("fr", 2)]]}))
    monkeypatch.setattr(figures, "SPLITS", manifest)
    target = tmp_path / "chart.svg"
    figures.render_data_split(target)
    svg = target.read_text(encoding="utf-8")
    assert "English 3 train" in svg and "French 2 train" in svg
    assert "748 train" not in svg and "481 train" not in svg


def test_source_manifest_omits_text_and_rejects_split_overlap(tmp_path, monkeypatch, scripts):
    export = scripts("export_open_weight_data_manifest")
    splits = tmp_path / "splits"
    splits.mkdir()
    row = {"id": "doc", "source_doc_id": "hal-1", "source_revision": "revision",
           "input_sha256": "input-hash", "license": "CC0-1.0", "text": "Private article text"}
    for lang in ["en", "fr"]:
        for split in ["train", "dev", "test"]:
            (splits / f"{lang}-{split}.jsonl").write_text(json.dumps(row) + "\n" if (lang, split) == ("en", "train") else "")
    documents = tmp_path / "documents.jsonl"
    documents.write_text(json.dumps(row) + "\n")
    out = tmp_path / "manifest.jsonl"
    monkeypatch.setattr(sys, "argv", ["export", "--documents", str(documents), "--splits", str(splits), "--out", str(out)])
    export.main()
    record = json.loads(out.read_text())
    assert record["source_record_uri"] == "https://hal.science/hal-1"
    assert record["processed_text_sha256"] == hashlib.sha256(row["text"].encode()).hexdigest()
    assert row["text"] not in out.read_text()
    (splits / "en-test.jsonl").write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="more than one split"):
        export.main()
