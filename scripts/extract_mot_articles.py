"""Extract MOT v1.11 article JSON as local JSONL while retaining supplied provenance fields.

The output includes source article text and must remain local; it is not a release artifact.
The audit JSON contains counts and hashes but never article text.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import tarfile
from collections import Counter
from pathlib import Path

AGENCY = re.compile(r"\b(?:AFP|AP|Associated Press|Reuters|DPA|ANSA|EFE|Agence France[- ]Presse)\b", re.I)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--language", choices=("en", "fr"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    counts: Counter[str] = Counter()

    with tarfile.open(args.archive, "r:gz") as archive, args.out.open("w", encoding="utf-8", newline="\n") as out:
        for member in archive:
            path = member.name.replace("\\", "/")
            if not member.isfile() or "/article/" not in path or not path.endswith(".json"):
                continue
            counts["article_records_seen"] += 1
            stream = archive.extractfile(member)
            if stream is None:
                counts["unreadable_records"] += 1
                continue
            try:
                record = json.load(stream)
            except (UnicodeDecodeError, json.JSONDecodeError):
                counts["malformed_records"] += 1
                continue
            if not isinstance(record, dict):
                counts["malformed_records"] += 1
                continue
            authors = record.get("authors") if isinstance(record.get("authors"), list) else []
            if AGENCY.search(" ".join(str(x) for x in authors)):
                counts["excluded_explicit_agency_byline"] += 1
                continue
            paragraphs = record.get("paragraphs") or []
            if isinstance(paragraphs, str):
                paragraphs = [paragraphs]
            if not isinstance(paragraphs, list):
                counts["malformed_records"] += 1
                continue
            paragraphs = [p.strip() for p in paragraphs if isinstance(p, str) and p.strip()]
            title = str(record.get("title") or "").strip()
            text = "\n\n".join(([title] if title else []) + paragraphs)
            if not text.strip():
                counts["empty_records"] += 1
                continue
            article_id = str(record.get("filename") or Path(path).stem)
            metadata = {
                "url": record.get("url"),
                "time_published": record.get("time_published"),
                "title": title,
                "authors": authors,
                "section": record.get("section"),
                "site_language": record.get("site_language"),
                "predicted_language": record.get("predicted_language"),
                "source_member": path,
            }
            out.write(json.dumps({"id": article_id, "text": text, "source_metadata": metadata}, ensure_ascii=False) + "\n")
            counts["records_written"] += 1
            counts[f"records_missing_byline_{args.language}"] += int(not authors)
            counts["text_characters_written"] += len(text)

    report = {
        "schema": "slm.mot_extract_audit.v1",
        "language": args.language,
        "archive": str(args.archive),
        "archive_sha256": sha256(args.archive),
        "output": str(args.out),
        "output_sha256": sha256(args.out),
        "counts": dict(counts),
        "article_text_included": False,
        "author_policy": "missing source bylines remain missing; no author inferred",
        "agency_policy": "exclude only explicit agency markers in supplied author/byline values",
    }
    args.audit.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"language": args.language, "counts": report["counts"], "audit": str(args.audit)}, indent=2))


if __name__ == "__main__":
    main()
