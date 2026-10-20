"""Create a deterministic EN/FR sample balanced within each permitted license class."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterator

REVISION = "bfa114c09d7c28dcd0a8a47fe445c64b4a3c6156"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def records(path: Path) -> Iterator[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def language_and_license(path: Path) -> tuple[str, str]:
    lang = path.parent.name
    lic = path.stem.replace(".jsonl", "")
    return lang, lic


def balance(input_dir: Path, output_dir: Path, seed: int = 20261020) -> dict:
    files = sorted(input_dir.glob("**/*.jsonl.gz"))
    if not files:
        raise FileNotFoundError(f"no aggregated JSONL files under {input_dir}")

    rows: dict[tuple[str, str], list[tuple[str, int]]] = defaultdict(list)
    input_hashes = []
    for path in files:
        lang, lic = language_and_license(path)
        input_hashes.append({"path": str(path), "sha256": sha256_file(path)})
        for row in records(path):
            count = len(row["text"].split())
            rows[(lang, lic)].append((row["id"], count))

    licenses = sorted({lic for _, lic in rows})
    selection: dict[tuple[str, str], set[str]] = {}
    report = {}
    for lic in licenses:
        en_rows = rows.get(("en", lic), [])
        fr_rows = rows.get(("fr", lic), [])
        if not en_rows or not fr_rows:
            report[lic] = {"status": "excluded_from_balanced_mix", "reason": "license not present in both languages"}
            continue
        en_total = sum(n for _, n in en_rows)
        fr_total = sum(n for _, n in fr_rows)
        target = min(en_total, fr_total)
        for lang, items in (("en", en_rows), ("fr", fr_rows)):
            ordered = sorted(items, key=lambda item: hashlib.sha256(f"{seed}:{lic}:{lang}:{item[0]}".encode()).digest())
            chosen: set[str] = set()
            total = 0
            for doc_id, count in ordered:
                if total + count <= target:
                    chosen.add(doc_id)
                    total += count
            selection[(lang, lic)] = chosen
            report.setdefault(lic, {})[lang] = {
                "available_documents": len(items),
                "available_whitespace_words": sum(n for _, n in items),
                "selected_documents": len(chosen),
                "selected_whitespace_words": total,
            }
        report[lic]["target_whitespace_words_per_language"] = target

    counters = defaultdict(lambda: {"documents": 0, "whitespace_words": 0})
    grouped_paths: dict[tuple[str, str], list[Path]] = defaultdict(list)
    for path in files:
        grouped_paths[language_and_license(path)].append(path)
    files_out = []
    for (lang, lic), group_paths in sorted(grouped_paths.items()):
        chosen = selection.get((lang, lic), set())
        output = output_dir / lang / f"{lic}.jsonl.gz"
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("wb") as raw, gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0, compresslevel=1) as zipped:
            for path in group_paths:
                for row in records(path):
                    if row["id"] not in chosen:
                        continue
                    zipped.write((json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8"))
                    counters[(lang, lic)]["documents"] += 1
                    counters[(lang, lic)]["whitespace_words"] += len(row["text"].split())
        files_out.append({"inputs": [str(path) for path in group_paths], "output": str(output),
                          **counters[(lang, lic)], "sha256": sha256_file(output)})

    manifest = {
        "kind": "slm.balanced_source_sample",
        "dataset": "PleIAs/French-Science-Commons",
        "revision": REVISION,
        "subset": "french_open_science",
        "seed": seed,
        "selection": "deterministic SHA-256 order within each license and language, greedily selected to equal whitespace-word totals across EN/FR",
        "word_count_note": "Whitespace-separated word counts are a sampling proxy; actual token totals will be measured after training the frozen tokenizer.",
        "license_balance": report,
        "outputs": files_out,
        "input_hashes": input_hashes,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "balance_manifest.json").open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(manifest, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=20261020)
    args = parser.parse_args()
    result = balance(Path(args.input_dir), Path(args.output_dir), args.seed)
    print(json.dumps(result["license_balance"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
