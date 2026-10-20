"""Join consecutive licensed page rows into article-level training documents."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def aggregate_one(source: Path, destination: Path, excluded_ids: set[str]) -> dict:
    destination.parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    pages = 0
    docs = 0
    words = 0
    chars = 0
    current_id: str | None = None
    current_meta: dict | None = None
    current_page_texts: dict[int, str] = {}
    current_source_labels: set[str] = set()
    current_words = 0
    excluded_articles = 0
    excluded_pages = 0
    conflicting_page_articles = 0
    duplicate_pages_removed = 0

    def flush(out) -> None:
        nonlocal docs, words, chars, current_id, current_meta, current_page_texts, current_source_labels, current_words, excluded_articles, excluded_pages, conflicting_page_articles, duplicate_pages_removed
        if current_id is None or current_meta is None:
            return
        if current_id in excluded_ids:
            excluded_articles += 1
            excluded_pages += len(current_page_texts)
            return
        if getattr(flush, "page_conflict", False):
            conflicting_page_articles += 1
            excluded_pages += len(current_page_texts)
            return
        page_numbers = sorted(current_page_texts)
        text = "\n\n".join(current_page_texts[p] for p in page_numbers)
        metadata = {**current_meta, "page_numbers": page_numbers, "page_count": len(page_numbers),
                    "source_labels": sorted(current_source_labels)}
        result = {
            "id": current_id,
            "text": text,
            "source_content_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "source_metadata": metadata,
        }
        out.write((json.dumps(result, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8"))
        docs += 1
        words += current_words
        chars += len(text)

    with source.open("rb") as raw_in, destination.open("wb") as raw_out:
        with gzip.GzipFile(fileobj=raw_in, mode="rb") as reader, gzip.GzipFile(
            filename="", fileobj=raw_out, mode="wb", mtime=0, compresslevel=1
        ) as writer:
            for line in reader:
                row = json.loads(line)
                metadata = row["source_metadata"]
                doc_id = str(metadata["id"])
                if current_id is not None and doc_id != current_id:
                    flush(writer)
                    seen.add(current_id)
                    if doc_id in seen:
                        raise ValueError(f"non-consecutive pages for article {doc_id!r} in {source}")
                    current_id, current_meta, current_texts, current_pages, current_words = None, None, [], [], 0
                if current_id is None:
                    current_id = doc_id
                    current_meta = dict(metadata)
                    current_meta.pop("page", None)
                    current_page_texts = {}
                    current_source_labels = set()
                    current_words = 0
                    flush.page_conflict = False
                elif current_id not in excluded_ids and (
                    any(current_meta.get(k) != metadata.get(k) for k in ("language", "license", "doi"))
                    or str(current_meta.get("source") or "").casefold() != str(metadata.get("source") or "").casefold()
                ):
                    raise ValueError(f"page metadata conflict within article {doc_id!r} in {source}")
                page = int(metadata["page"])
                page_text = row["text"]
                current_source_labels.add(str(metadata.get("source") or ""))
                if page in current_page_texts:
                    if current_page_texts[page] != page_text:
                        flush.page_conflict = True
                    else:
                        duplicate_pages_removed += 1
                else:
                    current_page_texts[page] = page_text
                    current_words += int(metadata.get("word_count") or 0)
                pages += 1
            flush(writer)

    return {
        "input": str(source),
        "input_sha256": sha256(source),
        "output": str(destination),
        "output_sha256": sha256(destination),
        "pages": pages,
        "articles": docs,
        "source_word_count_sum": words,
        "characters": chars,
        "excluded_articles_for_metadata_conflicts": excluded_articles,
        "excluded_pages_for_metadata_conflicts": excluded_pages,
        "conflicting_page_articles_excluded": conflicting_page_articles,
        "duplicate_pages_removed": duplicate_pages_removed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--exclude-audit", required=True, help="document_metadata_audit.json from audit_french_science_documents.py")
    args = parser.parse_args()
    audit = json.loads(Path(args.exclude_audit).read_text(encoding="utf-8"))
    excluded_ids = set(audit["excluded_document_ids"])
    inputs = []
    for directory in args.input_dir:
        root = Path(directory)
        for source in sorted(root.glob("**/*.jsonl.gz")):
            inputs.append((root, source))
    results = []
    for root, source in inputs:
        relative = Path(root.name) / source.relative_to(root)
        destination = Path(args.output_dir) / relative
        result = aggregate_one(source, destination, excluded_ids)
        results.append(result)
        print(f"{source.name}: {result['pages']:,} pages -> {result['articles']:,} articles", flush=True)
    manifest = {"kind": "slm.document_aggregation", "exclude_audit": args.exclude_audit,
                "excluded_document_ids": len(excluded_ids), "source_document_ids": len(audit["documents"]), "inputs": results}
    manifest_path = Path(args.output_dir) / "aggregation_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
        f.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
