"""Balance a processed FSC pilot by language within each license class."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from collections import defaultdict
from contextlib import ExitStack
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def license_key(source: str) -> str:
    return source.removeprefix("fsc-en-").removeprefix("fsc-fr-")


def balance(input_path: Path, output_dir: Path, seed: int) -> dict:
    groups: dict[tuple[str, str], list[tuple[str, int, str]]] = defaultdict(list)
    with input_path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            source = row["source"]
            language = source[4:6]
            lic = license_key(source)
            words = len(row["text"].split())
            groups[(language, lic)].append((row["id"], words, source))

    selected: dict[tuple[str, str], set[str]] = {}
    report = {}
    for lic in sorted({key[1] for key in groups}):
        en, fr = groups[("en", lic)], groups[("fr", lic)]
        en_total, fr_total = sum(x[1] for x in en), sum(x[1] for x in fr)
        target = min(en_total, fr_total)
        for language, items in (("en", en), ("fr", fr)):
            ordered = sorted(items, key=lambda x: hashlib.sha256(f"{seed}:{lic}:{language}:{x[0]}".encode()).digest())
            chosen: set[str] = set()
            total = 0
            for doc_id, words, _ in ordered:
                if total + words <= target:
                    chosen.add(doc_id)
                    total += words
            selected[(language, lic)] = chosen
            report.setdefault(lic, {})[language] = {
                "available_documents": len(items),
                "available_whitespace_words": sum(x[1] for x in items),
                "selected_documents": len(chosen),
                "selected_whitespace_words": total,
                "discarded_documents": len(items) - len(chosen),
            }
        report[lic]["target_whitespace_words_per_language"] = target
        report[lic]["language_word_difference"] = abs(
            report[lic]["en"]["selected_whitespace_words"] - report[lic]["fr"]["selected_whitespace_words"]
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    paths = {(language, lic): output_dir / language / f"{lic}.jsonl.gz"
             for language, lic in selected}
    stale = set(output_dir.glob("**/*.jsonl.gz")) - set(paths.values())
    if stale:
        raise ValueError("output directory contains groups absent from this input; use a fresh output directory")
    with ExitStack() as stack:
        streams = {}
        for key, path in paths.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            raw = stack.enter_context(path.open("wb"))
            streams[key] = stack.enter_context(
                gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0, compresslevel=1)
            )
        stream = stack.enter_context(input_path.open(encoding="utf-8"))
        for line in stream:
            row = json.loads(line)
            source = row["source"]
            language, lic = source[4:6], license_key(source)
            if row["id"] not in selected[(language, lic)]:
                continue
            streams[(language, lic)].write((json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8"))

    for path in sorted(paths.values()):
        outputs.append({"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size})
    manifest = {
        "kind": "slm.processed_balanced_sample",
        "input": str(input_path),
        "input_sha256": sha256_file(input_path),
        "seed": seed,
        "selection": "SHA-256 order within each language/license group; greedily include documents without exceeding the smaller-language whitespace-word total",
        "note": "Document boundaries are retained. Per-language token totals will be recomputed with the frozen tokenizer.",
        "license_balance": report,
        "outputs": outputs,
    }
    (output_dir / "balance_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=20261020)
    args = parser.parse_args()
    result = balance(Path(args.input), Path(args.output_dir), args.seed)
    print(json.dumps(result["license_balance"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
