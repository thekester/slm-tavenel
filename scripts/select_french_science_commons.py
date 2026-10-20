"""Select traceable open-license pages from pinned French-Science-Commons shards.

This is a conservative source selector, not a rights approval. It only keeps French and
English rows from the ``french_open_science`` directory whose per-page license has a
specific allow-listed identifier and whose additional-terms field is empty. Raw page text
and attribution/provenance fields are retained for the repository's normal data pipeline.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pyarrow.parquet as pq

DATASET = "PleIAs/French-Science-Commons"
REVISION = "bfa114c09d7c28dcd0a8a47fe445c64b4a3c6156"
SUBSET = "french_open_science"
LICENSE_MAP = {
    "cc0": "CC0-1.0",
    "cc-by/4.0": "CC-BY-4.0",
    "etalab-2.0": "Licence-Ouverte-2.0",
}
LANGUAGES = {"en", "fr"}
METADATA_FIELDS = (
    "id", "title", "author", "publication_date", "doi", "language", "license",
    "terms", "discipline", "source", "page", "word_count", "supra",
)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def code_revision() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def iter_rows(paths: Iterable[Path]):
    for path in paths:
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=1024):
            yield path, batch.to_pylist()


def select(inputs: list[Path], output_dir: Path, revision: str = REVISION) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[tuple[str, str], Any] = {}
    counts: Counter[str] = Counter()
    words: Counter[str] = Counter()
    chars: Counter[str] = Counter()
    source_counts: dict[str, Counter[str]] = {}

    def output_for(key: tuple[str, str]):
        if key not in outputs:
            lang, license_id = key
            path = output_dir / lang / f"{license_id.lower().replace('-', '_')}.jsonl.gz"
            path.parent.mkdir(parents=True, exist_ok=True)
            raw = path.open("wb")
            compressed = gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0)
            outputs[key] = (path, raw, compressed)
        return outputs[key][2]

    input_rows = 0
    try:
        for parquet_path, rows in iter_rows(inputs):
            for row in rows:
                input_rows += 1
                language = (row.get("language") or "").strip().lower()
                raw_license = (row.get("license") or "").strip().lower()
                license_id = LICENSE_MAP.get(raw_license)
                terms = (row.get("terms") or "").strip()
                source = (row.get("source") or "").strip()
                doc_id = str(row.get("id") or "").strip()
                page = row.get("page")
                text = row.get("text") or ""

                if language not in LANGUAGES:
                    counts["drop_language"] += 1
                    continue
                if license_id is None:
                    counts[f"drop_license:{raw_license or 'missing'}"] += 1
                    continue
                if terms:
                    counts["drop_nonempty_terms"] += 1
                    continue
                if not source or not doc_id or page is None:
                    counts["drop_missing_provenance"] += 1
                    continue
                if not text.strip():
                    counts["drop_empty_text"] += 1
                    continue

                key = (language, license_id)
                result = {
                    "id": f"{doc_id}#page-{int(page):04d}",
                    "text": text,
                    "source_content_sha256": sha256_text(text),
                    "source_metadata": {field: row.get(field) for field in METADATA_FIELDS},
                }
                line = (json.dumps(result, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
                output_for(key).write(line)
                counts[f"keep:{language}:{license_id}"] += 1
                words[f"{language}:{license_id}"] += int(row.get("word_count") or 0)
                chars[f"{language}:{license_id}"] += len(text)
                source_counts.setdefault(f"{language}:{license_id}", Counter())[source] += 1
                counts[f"selected_from_file:{parquet_path.name}"] += 1
    finally:
        for _path, raw, compressed in outputs.values():
            compressed.close()
            raw.close()

    output_records = {}
    for (lang, license_id), (path, _raw, _gz) in outputs.items():
        output_records[f"{lang}:{license_id}"] = {
            "path": str(path),
            "records": counts[f"keep:{lang}:{license_id}"],
            "word_count_sum_from_source": words[f"{lang}:{license_id}"],
            "characters": chars[f"{lang}:{license_id}"],
            "sha256": sha256_file(path),
            "top_source_labels": source_counts[f"{lang}:{license_id}"].most_common(20),
        }

    manifest = {
        "kind": "slm.source_selection",
        "dataset": DATASET,
        "revision": revision,
        "subset": SUBSET,
        "selection": {
            "languages": sorted(LANGUAGES),
            "raw_license_to_spdx": LICENSE_MAP,
            "require_empty_terms": True,
            "required_metadata": ["id", "page", "source"],
            "unit": "page; article id and page number are retained for article-level splits",
        },
        "input_rows": input_rows,
        "inputs": [{"path": str(path), "sha256": sha256_file(path)} for path in inputs],
        "outputs": output_records,
        "counts": dict(sorted(counts.items())),
        "code_revision": code_revision(),
        "created": datetime.now(timezone.utc).isoformat(),
        "disclaimer": "Candidate subset only. Per-document metadata is retained; it has not been independently rights- or privacy-reviewed.",
    }
    with (output_dir / "selection_manifest.json").open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(manifest, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True, help="Pinned french_open_science parquet shard; repeatable")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--revision", default=REVISION)
    args = parser.parse_args()
    manifest = select([Path(p) for p in args.input], Path(args.output_dir), args.revision)
    print(json.dumps({"input_rows": manifest["input_rows"], "counts": manifest["counts"], "outputs": manifest["outputs"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
