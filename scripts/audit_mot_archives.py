"""Inventory MOT article metadata without emitting or storing article text.

The report is a provenance aid, not a legal clearance: bylines cannot establish
who owns every part of an article or whether its text is exclusively VOA-produced.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import tarfile
from collections import Counter
from pathlib import Path
from typing import Any
from datetime import datetime, timezone


AGENCY_MARKERS = re.compile(
    r"\b(?:afp|ap|associated press|reuters|agence france[- ]presse|dpa|ansa|efe)\b",
    re.IGNORECASE,
)
REQUIRED_METADATA = ("url", "content_type", "site_language", "time_published", "title", "authors")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def text_strings(value: Any):
    if isinstance(value, str):
        yield value
    elif isinstance(value, list):
        for item in value:
            yield from text_strings(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from text_strings(item)


def audit_archive(path: Path) -> dict[str, Any]:
    records = 0
    malformed = 0
    missing = Counter()
    types = Counter()
    years = Counter()
    author_values = Counter()
    agency_examples: list[dict[str, str]] = []
    text_characters = 0
    replacement_characters = 0
    replacement_records = 0
    control_characters = 0

    with tarfile.open(path, mode="r:gz") as archive:
        for member in archive:
            normalized = member.name.replace("\\", "/")
            if not member.isfile() or "/article/" not in normalized or not normalized.endswith(".json"):
                continue
            stream = archive.extractfile(member)
            if stream is None:
                malformed += 1
                continue
            try:
                record = json.load(stream)
            except (UnicodeDecodeError, json.JSONDecodeError):
                malformed += 1
                continue
            if not isinstance(record, dict):
                malformed += 1
                continue

            records += 1
            record_replacements = 0
            for text in text_strings(record.get("paragraphs", [])):
                text_characters += len(text)
                record_replacements += text.count("\ufffd")
                control_characters += sum(
                    1 for char in text if ord(char) < 32 and char not in "\t\n\r"
                )
            replacement_characters += record_replacements
            replacement_records += record_replacements > 0
            for key in REQUIRED_METADATA:
                value = record.get(key)
                if value is None or value == "" or value == []:
                    missing[key] += 1
            types[str(record.get("content_type") or "<missing>")] += 1
            published = str(record.get("time_published") or "")
            years[published[:4] if re.match(r"^\d{4}", published) else "<missing>"] += 1

            authors = record.get("authors")
            if not isinstance(authors, list):
                authors = []
            labels = [str(author).strip() for author in authors if str(author).strip()]
            if not labels:
                author_values["<missing>"] += 1
            else:
                for label in labels:
                    author_values[label] += 1
                    if AGENCY_MARKERS.search(label) and len(agency_examples) < 100:
                        agency_examples.append({"record": normalized, "author": label})

    return {
        "archive": path.name,
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "article_records": records,
        "malformed_article_records": malformed,
        "text_character_count": text_characters,
        "unicode_replacement_character_count": replacement_characters,
        "records_with_unicode_replacement_character": replacement_records,
        "unexpected_control_character_count": control_characters,
        "missing_required_metadata_counts": dict(sorted(missing.items())),
        "content_type_counts": dict(sorted(types.items())),
        "publication_year_counts": dict(sorted(years.items())),
        "byline_counts": dict(sorted(author_values.items(), key=lambda item: (-item[1], item[0]))),
        "agency_marker_byline_records": agency_examples,
        "agency_marker_byline_records_capped_at": 100,
        "text_content_included": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archives", nargs="+", type=Path, help="Pinned MOT .tgz files to inspect")
    parser.add_argument("--output", type=Path, required=True, help="Path for aggregate JSON metadata report")
    args = parser.parse_args()

    reports = [audit_archive(path) for path in args.archives]
    script_path = Path(__file__).resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "schema": "slm.mot_metadata_audit.v1",
                "performed_at_utc": datetime.now(timezone.utc).isoformat(),
                "python_version": platform.python_version(),
                "script": script_path.name,
                "script_sha256": sha256_file(script_path),
                "archives": reports,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote MOT metadata and text-quality audit for {len(reports)} archive(s) to {args.output}")
    for report in reports:
        print(
            f"{report['archive']}: {report['article_records']} articles, "
            f"{report['malformed_article_records']} malformed, "
            f"{len(report['byline_counts'])} byline labels, "
            f"{len(report['agency_marker_byline_records'])} agency-marker records (max 100 listed)"
        )


if __name__ == "__main__":
    main()
