"""Exercise MOT preparation on tiny synthetic data without running training queues."""

import gzip
import hashlib
import importlib
import io
import json
from pathlib import Path
import sys
import tarfile

import numpy as np
import pytest


@pytest.fixture
def scripts(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module


def write_tokens(path, train, val=(1, 0)):
    path.mkdir(parents=True)
    record = {"dtype": "uint16", "vocab_size": 32, "eos_id": 0,
              "tokenizer_sha256": "same-tokenizer", "splits": {}}
    for split, tokens in [("train", train), ("val", val)]:
        shard = path / f"{split}_00000.bin"
        np.asarray(tokens, dtype=np.uint16).tofile(shard)
        record["splits"][split] = {"tokens": len(tokens), "docs": tokens.count(0), "shards": [
            {"file": shard.name, "tokens": len(tokens), "sha256": hashlib.sha256(shard.read_bytes()).hexdigest()}]}
    (path / "manifest.json").write_text(json.dumps(record))
    return record


def test_mix_trims_at_document_boundary_and_preserves_fsc_validation(tmp_path, monkeypatch, scripts):
    module = scripts("build_exp0012_mixture_shards")
    monkeypatch.setattr(module, "ROOT", tmp_path)
    fsc = tmp_path / "data/tokens/fsc-pilot-v1-scaling"
    write_tokens(fsc, [1, 1, 1, 0] * 5)
    tokenizer = tmp_path / "data/tokenizer/fsc-pilot-v1-train-only/manifest.json"
    tokenizer.parent.mkdir(parents=True)
    tokenizer.write_text("{}")
    mot = tmp_path / "mot"
    write_tokens(mot, [2, 2, 0, 3, 3, 0])
    out = tmp_path / "mix"
    monkeypatch.setattr(sys, "argv", ["mix", "--mot-tokens", str(mot), "--out", str(out)])
    module.main()
    # Target = 5 news tokens; the first complete article ends at token 3.
    assert np.fromfile(out / "train_mot_00000.bin", dtype=np.uint16).tolist() == [2, 2, 0]
    assert (out / "val_fsc_00000.bin").read_bytes() == (fsc / "val_00000.bin").read_bytes()
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["mix"]["actual_news_fraction"] == pytest.approx(3 / 23)
    assert manifest["mix"]["seed_used_for_reordering"] is False
    with pytest.raises(FileExistsError):
        module.main()


def test_stream_rejects_empty_or_wrong_counts(tmp_path, scripts):
    module = scripts("build_exp0012_mixture_shards")
    path = tmp_path / "empty"
    write_tokens(path, [])
    with pytest.raises(ValueError, match="empty"):
        module.load_stream(path, "train")
    record = write_tokens(tmp_path / "bad", [1, 0])
    record["splits"]["train"]["shards"][0]["tokens"] = 9
    (tmp_path / "bad/manifest.json").write_text(json.dumps(record))
    with pytest.raises(ValueError, match="token count"):
        module.load_stream(tmp_path / "bad", "train")
    with pytest.raises(ValueError, match="boundary"):
        module.trim_at_eos(np.array([1, 2, 0]), 2, 0)
    assert module.trim_at_eos(np.array([1, 0, 2, 0]), 4, 0).tolist() == [1, 0, 2, 0]


def test_extraction_skips_malformed_and_agency_records_without_leaking_text(tmp_path, monkeypatch, scripts):
    extract = scripts("extract_mot_articles")
    archive = tmp_path / "mot.tgz"
    rows = [{"title": "Title", "paragraphs": "Private article body", "authors": []},
            {"paragraphs": ["Agency body"], "authors": ["Reuters"]}, []]
    with tarfile.open(archive, "w:gz") as out:
        for i, row in enumerate(rows):
            content = json.dumps(row).encode()
            entry = tarfile.TarInfo(f"en/article/{i}.json")
            entry.size = len(content)
            out.addfile(entry, io.BytesIO(content))
    output, audit = tmp_path / "articles.jsonl", tmp_path / "audit.json"
    monkeypatch.setattr(sys, "argv", ["extract", str(archive), "--language", "en", "--out", str(output), "--audit", str(audit)])
    extract.main()
    record = json.loads(audit.read_text())
    assert record["counts"]["records_written"] == 1
    assert record["counts"]["malformed_records"] == 1
    assert record["counts"]["excluded_explicit_agency_byline"] == 1
    assert json.loads(output.read_text())["text"] == "Title\n\nPrivate article body"
    assert "Private article body" not in audit.read_text()


def test_temporal_splits_boundaries_duplicates_and_existing_output(tmp_path, monkeypatch, scripts):
    split = scripts("split_exp0012_mot_temporal")
    source, out = tmp_path / "documents.jsonl", tmp_path / "splits"
    rows = [{"id": str(i), "source": "mot-v1.11-en", "text": "Example",
             "source_metadata": {"time_published": year, "authors": []}}
            for i, year in enumerate(["2021-01-01", "2022", "2023", "2024", "unknown"])]
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))
    monkeypatch.setattr(sys, "argv", ["split", str(source), "--out", str(out)])
    split.main()
    assert [len((out / f"{name}.jsonl").read_text().splitlines()) for name in ["train", "dev", "test"]] == [1, 2, 1]
    assert json.loads((out / "split-audit.json").read_text())["counts"]["excluded_missing_publication_year"] == 1
    with pytest.raises(FileExistsError):
        split.main()
    rows[1]["id"] = rows[0]["id"]
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))
    monkeypatch.setattr(sys, "argv", ["split", str(source), "--out", str(tmp_path / "duplicates")])
    with pytest.raises(ValueError, match="unique"):
        split.main()
