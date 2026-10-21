"""Create deterministic, language-stratified document train/dev/test splits."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dev-fraction", type=float, default=0.05)
    parser.add_argument("--test-fraction", type=float, default=0.05)
    parser.add_argument("--salt", default="slm-tavenel-open-weight-cc0-split-v1")
    args = parser.parse_args()
    if args.dev_fraction <= 0 or args.test_fraction <= 0 or args.dev_fraction + args.test_fraction >= 1:
        parser.error("dev/test fractions must be positive and sum to less than one")

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("Split output directory must be empty to avoid mixing corpus versions")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {name: {} for name in ("train", "dev", "test")}
    handles: dict[tuple[str, str], object] = {}
    paths: dict[tuple[str, str], Path] = {}
    counts: dict[tuple[str, str], Counter] = {}
    seen_ids: set[str] = set()
    try:
        with args.input.open("r", encoding="utf-8") as source:
            for line_number, line in enumerate(source, 1):
                row = json.loads(line)
                identifier = str(row.get("id", ""))
                metadata = row.get("source_metadata") or {}
                language = str(row.get("language") or metadata.get("language") or
                               ("en" if str(row.get("source", "")).endswith("-en") else
                                "fr" if str(row.get("source", "")).endswith("-fr") else "")).lower()
                if not identifier or language not in {"en", "fr"}:
                    raise ValueError(f"Missing ID or unsupported language at line {line_number}")
                key = f"{language}:{identifier}"
                if key in seen_ids:
                    raise ValueError(f"Duplicate document ID in input: {key}")
                seen_ids.add(key)
                digest = hashlib.sha256(f"{args.salt}:{key}".encode("utf-8")).digest()
                value = int.from_bytes(digest[:8], "big") / 2**64
                split = "dev" if value < args.dev_fraction else (
                    "test" if value < args.dev_fraction + args.test_fraction else "train")
                pair = (language, split)
                if pair not in handles:
                    path = args.output_dir / f"{language}-{split}.jsonl"
                    paths[pair] = path
                    handles[pair] = path.open("w", encoding="utf-8", newline="\n")
                    counts[pair] = Counter()
                handles[pair].write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                counts[pair]["documents"] += 1
                counts[pair]["words"] += len(str(row.get("text", "")).split())
                counts[pair]["characters"] += len(str(row.get("text", "")))
    finally:
        for handle in handles.values():
            handle.close()

    splits = []
    for (language, split), path in sorted(paths.items()):
        splits.append({"language": language, "split": split, "path": str(path),
                       "sha256": sha256_file(path), "bytes": path.stat().st_size,
                       **counts[(language, split)]})
    manifest = {
        "kind": "slm.open_weight_document_splits",
        "input": str(args.input),
        "input_sha256": sha256_file(args.input),
        "split_salt": args.salt,
        "hash": "SHA-256(salt + ':' + language + ':' + document_id), first 64 bits mapped to [0,1)",
        "dev_fraction": args.dev_fraction,
        "test_fraction": args.test_fraction,
        "document_unit": "complete aggregated article/document; never split pages across sets",
        "splits": splits,
    }
    manifest_path = args.output_dir / "split_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    summary = {"output": str(manifest_path), "input_documents": len(seen_ids),
               "splits": [{k: row[k] for k in ("language", "split", "documents", "words", "sha256")} for row in splits]}
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
