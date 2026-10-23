"""Export retained source provenance without including source article text."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--documents", type=Path, required=True, help="processed retained documents JSONL")
    parser.add_argument("--splits", type=Path, required=True, help="directory containing language split JSONL files")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    split_by_id: dict[str, tuple[str, str]] = {}
    for language in ("en", "fr"):
        for split in ("train", "dev", "test"):
            path = args.splits / f"{language}-{split}.jsonl"
            with path.open(encoding="utf-8") as stream:
                for line in stream:
                    if line.strip():
                        row = json.loads(line)
                        if row["id"] in split_by_id:
                            raise ValueError(f"Document appears in more than one split: {row['id']}")
                        split_by_id[row["id"]] = (language, split)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    count = 0
    with args.documents.open(encoding="utf-8") as source, args.out.open("w", encoding="utf-8", newline="\n") as output:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            doc_id = row["id"]
            if doc_id in seen:
                raise ValueError(f"Duplicate retained document id: {doc_id}")
            seen.add(doc_id)
            if doc_id not in split_by_id:
                raise ValueError(f"Retained document missing from split manifest inputs: {doc_id}")
            language, split = split_by_id[doc_id]
            metadata = row.get("source_metadata") or {}
            source_id = str(row["source_doc_id"])
            if source_id != str(metadata.get("id", source_id)):
                raise ValueError(f"Source ID mismatch for {doc_id}")
            if row.get("license") != "CC0-1.0":
                raise ValueError(f"Unexpected source license in v4 retained corpus: {doc_id}")
            item = {
                "dataset": "PleIAs/French-Science-Commons",
                "dataset_revision": row["source_revision"],
                "dataset_subset": "french_open_science",
                "dataset_record_id": doc_id,
                "source_record_id": source_id,
                "source_record_uri": f"https://hal.science/{source_id}",
                # Preserve the source's declared attribution metadata verbatim where present.
                # These fields help trace the corpus; they do not independently verify
                # author identity or prove that the submitter held the relevant rights.
                "source_title": metadata.get("title"),
                "declared_author": metadata.get("author"),
                "publication_date": metadata.get("publication_date"),
                "doi": metadata.get("doi"),
                "language": language,
                "declared_dataset_license": row["license"],
                "split": split,
                "source_text_sha256_before_processing": row["input_sha256"],
                "processed_text_sha256": sha256_text(row["text"]),
                "hal_license_metadata_audit": "training_record/hal-license-audit.json",
            }
            output.write(json.dumps(item, ensure_ascii=True, sort_keys=True) + "\n")
            count += 1

    if count != len(split_by_id):
        raise ValueError(f"Split files contain {len(split_by_id)} rows but {count} processed documents were exported")
    print(json.dumps({"documents": count, "manifest": str(args.out)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
