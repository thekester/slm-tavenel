"""Find article ids whose page-level license/provenance metadata is inconsistent."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

from select_french_science_commons import REVISION, SUBSET, sha256_file


def audit(paths: list[Path], out_path: Path) -> dict:
    docs: dict[str, dict] = {}
    pages = 0
    for path in paths:
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(
            columns=["id", "language", "license", "terms", "source", "page",
                     "title", "author", "publication_date", "doi", "discipline"],
            batch_size=8192,
        ):
            for row in batch.to_pylist():
                pages += 1
                doc_id = str(row.get("id") or "").strip()
                if not doc_id:
                    continue
                record = docs.get(doc_id)
                if record is None:
                    docs[doc_id] = {
                        "languages": {str(row.get("language") or "").strip().lower()},
                        "licenses": {str(row.get("license") or "").strip().lower()},
                        "sources": {str(row.get("source") or "").strip()},
                        "source_keys": {str(row.get("source") or "").strip().casefold()},
                        "metadata": {
                            field: {str(row.get(field) or "").strip()}
                            for field in ("title", "author", "publication_date", "doi", "discipline")
                        },
                        "has_nonempty_terms": bool((row.get("terms") or "").strip()),
                        "pages": 1,
                        "page_sum": int(row.get("page") or 0),
                        "page_min": int(row.get("page") or 0),
                        "page_max": int(row.get("page") or 0),
                        "page_numbers": {int(row.get("page") or 0)},
                        "shards": {path.name},
                    }
                else:
                    record["languages"].add(str(row.get("language") or "").strip().lower())
                    record["licenses"].add(str(row.get("license") or "").strip().lower())
                    source = str(row.get("source") or "").strip()
                    record["sources"].add(source)
                    record["source_keys"].add(source.casefold())
                    for field in ("title", "author", "publication_date", "doi", "discipline"):
                        record["metadata"][field].add(str(row.get(field) or "").strip())
                    record["has_nonempty_terms"] |= bool((row.get("terms") or "").strip())
                    record["pages"] += 1
                    page = int(row.get("page") or 0)
                    record["page_sum"] += page
                    record["page_min"] = min(record["page_min"], page)
                    record["page_max"] = max(record["page_max"], page)
                    record["page_numbers"].add(page)
                    record["shards"].add(path.name)

    reasons: Counter[str] = Counter()
    excluded = {}
    for doc_id, record in docs.items():
        flags = []
        if len(record["languages"]) != 1:
            flags.append("mixed_language_metadata")
        if len(record["licenses"]) != 1:
            flags.append("mixed_license_metadata")
        if len(record["source_keys"]) != 1 or "" in record["source_keys"]:
            flags.append("incomplete_or_mixed_source_metadata")
        if any(len(values) != 1 for values in record["metadata"].values()):
            flags.append("inconsistent_article_metadata")
        if record["has_nonempty_terms"]:
            flags.append("nonempty_additional_terms")
        if len(record["shards"]) != 1:
            flags.append("article_spans_multiple_shards")
        if flags:
            excluded[doc_id] = flags
            reasons.update(flags)

    manifest = {
        "kind": "slm.document_metadata_audit",
        "dataset": "PleIAs/French-Science-Commons",
        "revision": REVISION,
        "subset": SUBSET,
        "inputs": [{"path": str(path), "sha256": sha256_file(path)} for path in paths],
        "pages_scanned": pages,
        "documents_scanned": len(docs),
        "documents_excluded": len(excluded),
        "exclusion_reasons": dict(sorted(reasons.items())),
        "excluded_document_ids": excluded,
        "documents": {
            doc_id: {
                "page_count": record["pages"],
                "page_min": record["page_min"],
                "page_max": record["page_max"],
                "unique_page_count": len(record["page_numbers"]),
                "duplicate_page_rows": record["pages"] - len(record["page_numbers"]),
                "missing_page_numbers_within_range": max(0, record["page_max"] - record["page_min"] + 1 - len(record["page_numbers"])),
                "source_labels": sorted(record["sources"]),
                "shard_count": len(record["shards"]),
            }
            for doc_id, record in docs.items()
        },
        "selection_rule": "exclude whole article if its page records disagree on language, license, source, title, author, publication date, DOI or discipline, have missing source metadata, contain any additional terms, or span multiple shards",
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(manifest, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    paths = sorted(Path(args.input_dir).glob("**/*.parquet"))
    result = audit(paths, Path(args.output))
    print(json.dumps({k: result[k] for k in ("pages_scanned", "documents_scanned", "documents_excluded", "exclusion_reasons")}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
