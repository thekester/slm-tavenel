"""Synthetic provenance and data-isolation regressions; no production queues or network."""

import importlib
import io
import json
from pathlib import Path
import sys
from urllib.parse import parse_qs, urlparse

import numpy as np
import pytest
from tokenizers import Tokenizer, models, pre_tokenizers

from slm.data.pipeline import build_step


@pytest.fixture
def scripts(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module


def test_metadata_filter_requires_fields_and_matches_case_insensitively():
    step = build_step({"publication_risk_metadata": {
        "excluded_disciplines": ["Life Sciences"], "title_patterns": [r"\bpatient(s)?\b"]}})
    for metadata, reason in [({}, "missing_discipline_metadata"),
                             ({"discipline": "Physics"}, "missing_title_metadata"),
                             ({"discipline": "Physics; LIFE SCIENCES", "title": "Example"}, "excluded_discipline"),
                             ({"discipline": "Physics", "title": "Study of PATIENTS"}, "human_subject_title_signal")]:
        kept, rejection = step({"source_metadata": metadata})
        assert kept is None and rejection[0] == reason
    doc = {"source_metadata": {"discipline": "Physics", "title": "Quantum states"}}
    assert step(doc) == (doc, None)


def test_hal_fetches_conflicting_versions_on_later_pages(monkeypatch, scripts):
    audit = scripts("audit_fsc_cc0_hal")
    starts = []
    def response(request, timeout):
        start = int(parse_qs(urlparse(request.full_url).query)["start"][0])
        starts.append(start)
        page = [{"halId_s": "hal-1", "licence_s": audit.CC0_URL if start == 0 else "other"}]
        return io.StringIO(json.dumps({"response": {"numFound": 2, "docs": page}}))
    monkeypatch.setattr(audit, "urlopen", response)
    records = audit.query_hal([{"id": "hal-1"}], timeout=1, retries=1)
    assert starts == [0, 1]
    assert [r["licence_s"] for r in records] == [audit.CC0_URL, "other"]


def test_hal_rejects_truncated_response(monkeypatch, scripts):
    audit = scripts("audit_fsc_cc0_hal")
    monkeypatch.setattr(audit, "urlopen", lambda *a, **kw:
                        io.StringIO(json.dumps({"response": {"numFound": 2, "docs": []}})))
    with pytest.raises(RuntimeError, match="Incomplete"):
        audit.query_hal([{"id": "hal-1"}], timeout=1, retries=1)


def test_split_is_deterministic_disjoint_and_does_not_guess_language(tmp_path, monkeypatch, scripts):
    split = scripts("split_open_weight_corpus")
    source = tmp_path / "input.jsonl"
    rows = [{"id": str(i), "language": language, "text": "Example document"}
            for language in ("en", "fr") for i in range(100)]
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))
    contents = []
    for attempt in range(2):
        out = tmp_path / str(attempt)
        monkeypatch.setattr(sys, "argv", ["split", "--input", str(source), "--output-dir", str(out)])
        split.main()
        contents.append({p.name: p.read_bytes() for p in out.glob("*.jsonl")})
        keys = [(r["language"], r["id"]) for data in contents[-1].values()
                for r in map(json.loads, data.splitlines())]
        assert len(keys) == len(set(keys)) == 200
    assert contents[0] == contents[1]
    with pytest.raises(FileExistsError):
        split.main()
    source.write_text('{"id": "x", "text": "no language"}\n')
    monkeypatch.setattr(sys, "argv", ["split", "--input", str(source), "--output-dir", str(tmp_path / "bad")])
    with pytest.raises(ValueError, match="unsupported language"):
        split.main()


def test_encoded_splits_keep_test_tokens_separate(tmp_path, monkeypatch, scripts):
    encode = scripts("encode_open_weight_splits")
    tokenizer = Tokenizer(models.WordLevel({"<|endoftext|>": 0, "train": 1, "dev": 2, "test": 3}))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tok = tmp_path / "tokenizer.json"
    tokenizer.save(str(tok))
    source = tmp_path / "splits"
    source.mkdir()
    for language in ("en", "fr"):
        for split in ("train", "dev", "test"):
            (source / f"{language}-{split}.jsonl").write_text(json.dumps({"text": split}) + "\n")
    manifest = source / "split_manifest.json"
    manifest.write_text("{}")
    out = tmp_path / "tokens"
    monkeypatch.setattr(sys, "argv", ["encode", "--tokenizer", str(tok), "--split-dir", str(source),
                                     "--split-manifest", str(manifest), "--out", str(out)])
    encode.main()
    record = json.loads((out / "manifest.json").read_text())
    for split, token in [("train", 1), ("val", 2), ("test", 3)]:
        assert record["splits"][split]["tokens"] == 4
        shard = out / record["splits"][split]["shards"][0]["file"]
        assert np.fromfile(shard, dtype=np.uint16).tolist() == [token, 0, token, 0]
    with pytest.raises(FileExistsError):
        encode.main()
