"""Make deterministic, temporal MOT train/dev/test files from the filtered documents."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("documents", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError("Temporal split output must be empty to preserve existing splits")
    args.out.mkdir(parents=True, exist_ok=True)
    outputs = {name: (args.out / f"{name}.jsonl").open("w", encoding="utf-8", newline="\n")
               for name in ("train", "dev", "test")}
    counts: Counter[str] = Counter()
    chars: Counter[str] = Counter()
    bylines: Counter[str] = Counter()
    try:
        rows = []
        seen_ids = set()
        with args.documents.open(encoding="utf-8") as source:
            for line in source:
                if not line.strip():
                    continue
                row = json.loads(line)
                identifier = row.get("id")
                if not isinstance(identifier, str) or not identifier or identifier in seen_ids:
                    raise ValueError("Temporal split requires unique, nonempty document IDs")
                seen_ids.add(identifier)
                if row.get("source") not in {"mot-v1.11-en", "mot-v1.11-fr"}:
                    raise ValueError(f"Unknown MOT source for {identifier}")
                meta = row.get("source_metadata") or {}
                stamp = str(meta.get("time_published") or "")
                year = int(stamp[:4]) if len(stamp) >= 4 and stamp[:4].isdigit() else None
                if year is None:
                    counts["excluded_missing_publication_year"] += 1
                    continue
                split = "train" if year <= 2021 else ("dev" if year <= 2023 else "test")
                row["experiment_split"] = split
                row["publication_year"] = year
                key = hashlib.sha256(str(row["id"]).encode()).hexdigest()
                rows.append((split, key, row))
        for split, _, row in sorted(rows, key=lambda item: (item[0], item[1])):
            outputs[split].write(json.dumps(row, ensure_ascii=False) + "\n")
            lang = "fr" if row.get("source") == "mot-v1.11-fr" else "en"
            counts[f"{split}_{lang}_documents"] += 1
            chars[split] += len(row["text"])
            meta = row.get("source_metadata") or {}
            bylines[f"{lang}_present"] += int(bool(meta.get("authors")))
            bylines[f"{lang}_missing"] += int(not meta.get("authors"))
    finally:
        for f in outputs.values():
            f.close()

    report = {
        "schema": "slm.exp0012_mot_temporal_split.v1",
        "input": str(args.documents),
        "input_sha256": digest(args.documents),
        "split_rule": "train publication year <= 2021; dev 2022-2023; test >= 2024; missing year excluded",
        "order": "within each split, records sorted by SHA-256 of stable source document ID",
        "counts": dict(counts),
        "byline_counts": dict(bylines),
        "characters_by_split": dict(chars),
        "outputs": {name: {"path": str(args.out / f"{name}.jsonl"),
                           "sha256": digest(args.out / f"{name}.jsonl")} for name in outputs},
        "article_text_included": False,
        "missing_authors_policy": "preserve as missing; no author inferred",
    }
    (args.out / "split-audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"counts": dict(counts), "byline_counts": dict(bylines),
                      "characters_by_split": dict(chars), "audit": str(args.out / 'split-audit.json')}, indent=2))


if __name__ == "__main__":
    main()
