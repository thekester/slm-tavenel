"""Offline FSC fixtures: metadata rejection, article grouping and corpus balance."""

import gzip
import importlib
import json
from pathlib import Path

import pytest

from slm.data.pipeline import MinHashStep, run
from slm.data.registry import Registry, evaluate
from slm.data.shards import split_of


@pytest.fixture
def scripts(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module


def page(doc_id="article", number=1, **overrides):
    return {"id": doc_id, "page": number, "text": "Some scientific text.",
            "language": "en", "license": "cc-by/4.0", "terms": "", "source": "Publisher",
            "title": "Title", "author": "Author", "publication_date": "2020",
            "doi": "example-doi", "discipline": "Physics", "word_count": 3, **overrides}


def parquet(path, rows):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    pq.write_table(pa.Table.from_pylist(rows), path)


@pytest.mark.parametrize("field", ["title", "author", "publication_date", "doi", "discipline"])
def test_audit_loads_and_rejects_conflicting_article_metadata(tmp_path, scripts, field):
    path = tmp_path / "shard.parquet"
    parquet(path, [page(), page(number=2, **{field: "different"}),
                   page("consistent"), page("consistent", number=2, source="PUBLISHER")])
    report = scripts("audit_french_science_documents").audit([path], tmp_path / "audit.json")
    assert report["excluded_document_ids"] == {"article": ["inconsistent_article_metadata"]}
    assert report["documents"]["consistent"]["unique_page_count"] == 2


def test_selector_filters_terms_licenses_and_preserves_provenance(tmp_path, scripts):
    path = tmp_path / "shard.parquet"
    parquet(path, [page("keep"), page("terms", terms="additional restriction"),
                   page("unknown", license="cc-by"), page("missing", source=""),
                   page("language", language="de")])
    report = scripts("select_french_science_commons").select([path], tmp_path / "selected")
    output = report["outputs"]["en:CC-BY-4.0"]
    assert output["records"] == 1
    with gzip.open(output["path"], "rt", encoding="utf-8") as stream:
        row = json.loads(stream.readline())
    assert row["source_metadata"]["doi"] == "example-doi"
    assert row["id"] == "keep#page-0001"
    assert all(report["counts"][reason] == 1 for reason in
               ("drop_nonempty_terms", "drop_license:cc-by", "drop_missing_provenance", "drop_language"))


def test_aggregation_sorts_pages_and_excludes_conflicting_duplicates(tmp_path, scripts):
    source, destination = tmp_path / "pages.gz", tmp_path / "articles.gz"
    pages = [page(number=2, text="Second"), page(text="First"), page(text="First"),
             page("conflict", text="One"), page("conflict", text="Two")]
    with gzip.open(source, "wt", encoding="utf-8") as stream:
        for row in pages:
            stream.write(json.dumps({"text": row["text"], "source_metadata": row}) + "\n")
    report = scripts("aggregate_french_science_pages").aggregate_one(source, destination, set())
    with gzip.open(destination, "rt", encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream]
    assert len(rows) == 1 and rows[0]["text"] == "First\n\nSecond"
    assert report["duplicate_pages_removed"] == 1
    assert report["conflicting_page_articles_excluded"] == 1


def test_balancing_rerun_cannot_retain_old_language_rows(tmp_path, scripts):
    source, output = tmp_path / "input.jsonl", tmp_path / "balanced"
    rows = [{"id": lang, "source": f"fsc-{lang}-cc0", "text": "one two"} for lang in ("en", "fr")]
    source.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    balance = scripts("balance_processed_fsc").balance
    first = balance(source, output, 42)
    assert first == balance(source, output, 42)
    source.write_text(json.dumps(rows[0]) + "\n", encoding="utf-8")
    report = balance(source, output, 42)
    for info in report["outputs"]:
        with gzip.open(info["path"], "rt", encoding="utf-8") as stream:
            assert stream.read() == ""


def test_fsc_split_and_token_budget_match_encoder(tmp_path, scripts):
    from tokenizers import Tokenizer, models, pre_tokenizers

    tokenizer = Tokenizer(models.WordLevel({"<|endoftext|>": 0, "[UNK]": 1, "word": 2}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tok_path = tmp_path / "tokenizer.json"
    tokenizer.save(str(tok_path))
    source = tmp_path / "documents.jsonl"
    rows = [{"id": f"fsc-{lang}-cc0/{i}", "source": f"fsc-{lang}-cc0", "text": "word word"}
            for lang in ("en", "fr") for i in range(100)]
    source.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    module = scripts("balance_fsc_by_tokens")
    assert all(module.split_of(row["id"], 0.2) == split_of(row["id"], 0.2) for row in rows)
    report = module.balance([source], tok_path, tmp_path / "balanced", 0.2, 42)
    seen = set()
    for output in report["outputs"]:
        chosen = [json.loads(line) for line in Path(output["path"]).read_text().splitlines()]
        assert all(split_of(row["id"], 0.2) == output["split"] for row in chosen)
        assert seen.isdisjoint(row["id"] for row in chosen)
        seen.update(row["id"] for row in chosen)
        metrics = report["license_balance"]["cc0"][output["split"]][output["language"]]
        assert metrics["selected_tokens"] == 3 * len(chosen)


def test_preserved_metadata_cannot_override_pipeline_provenance(tmp_path):
    raw = tmp_path / "input.jsonl"
    raw.write_text(json.dumps({"id": "doc", "text": "Text", "source_metadata": {"doi": "example"}}) + "\n")
    cfg = {"registry": "configs/data/sources.yaml", "output_dir": str(tmp_path / "out"),
           "inputs": [{"source": "synthetic-prose-v1", "path": str(raw),
                       "preserve_fields": ["source_metadata"]}], "steps": []}
    path = tmp_path / "pipeline.json"
    path.write_text(json.dumps(cfg))
    run(path)
    document = json.loads((tmp_path / "out/documents.jsonl").read_text())
    assert document["source_metadata"] == {"doi": "example"}
    assert document["id"] == "synthetic-prose-v1/doc"
    cfg["inputs"][0]["preserve_fields"] = ["license"]
    path.write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match="cannot overwrite"):
        run(path)


def test_minhash_word_limit_does_not_truncate_retained_text():
    step = MinHashStep(max_words=5, num_perm=32, bands=8)
    first = {"id": "a", "text": "one two three four five then original content"}
    second = {"id": "b", "text": "one two three four five but different content"}
    assert step(first) == (first, None)
    assert step(second)[1][0] == "near_duplicate"
    with pytest.raises(ValueError, match="positive"):
        MinHashStep(max_words=0)


def test_fsc_registry_candidates_remain_blocked():
    registry = Registry.load("configs/data/sources.yaml")
    for lang in ("en", "fr"):
        for license_key in ("cc0", "ccby4", "lo2"):
            source = registry[f"fsc-{lang}-{license_key}"]
            decision = evaluate(source)
            assert source.status == "candidate" and not decision.usable
            assert "license not verified by a reviewer" in decision.blockers
            if license_key == "lo2":
                assert "Licence-Ouverte-2.0: attribution required" in decision.obligations


@pytest.fixture
def final_manifest_inputs(tmp_path, scripts):
    module = scripts("write_fsc_final_manifest")
    balanced, tokenizer, tokens = (tmp_path / name for name in ("balanced", "tokenizer", "tokens"))
    balanced.mkdir()
    tokenizer.mkdir()
    (tokenizer / "tokenizer.json").write_text("{}")
    tok_hash = module.sha256_file(tokenizer / "tokenizer.json")
    (tokenizer / "manifest.json").write_text(json.dumps({"tokenizer_sha256": tok_hash}))
    for language in ("en", "fr"):
        for split in ("train", "val"):
            doc = balanced / f"{language}-{split}.jsonl"
            doc.write_text(json.dumps({"id": f"{language}-{split}", "text": "text"}) + "\n")
            path = tokens / language / split / "manifest.json"
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps({
                "tokenizer_sha256": tok_hash, "inputs": [{"path": str(doc), "sha256": module.sha256_file(doc)}],
                "val_fraction": 0,
                "splits": {"train": {"docs": 1, "tokens": 10 if language == "en" else 12, "shards": []},
                           "val": {"docs": 0, "tokens": 0, "shards": []}},
            }))
    return module, balanced, tokenizer, tokens


def test_final_manifest_joins_verified_counts(final_manifest_inputs):
    module, balanced, tokenizer, tokens = final_manifest_inputs
    manifest = module.build_manifest(balanced, tokenizer, tokens, 0.2)
    assert manifest["validation_fraction"] == 0.2
    assert manifest["splits"]["fr"]["val"]["documents"] == 1
    assert manifest["language_token_balance"]["train"] == {
        "en_tokens": 10, "fr_tokens": 12, "absolute_difference": 2,
        "difference_fraction_of_mean": 2 / 11,
    }


@pytest.mark.parametrize("mismatch", ["documents", "tokenizer_file", "encoded_tokenizer", "resplit", "empty"])
def test_final_manifest_rejects_incompatible_inputs(final_manifest_inputs, mismatch):
    module, balanced, tokenizer, tokens = final_manifest_inputs
    path = tokens / "en/train/manifest.json"
    manifest = module.read(path)
    if mismatch == "documents":
        (balanced / "en-train.jsonl").write_text('{"id":"changed","text":"different"}\n')
    elif mismatch == "tokenizer_file":
        (tokenizer / "tokenizer.json").write_text('{"changed":true}')
    elif mismatch == "encoded_tokenizer":
        manifest["tokenizer_sha256"] = "different"
    elif mismatch == "resplit":
        manifest["val_fraction"] = 0.05
        manifest["splits"]["val"]["docs"] = 1
    else:
        manifest["splits"]["train"]["tokens"] = 0
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        module.build_manifest(balanced, tokenizer, tokens)
