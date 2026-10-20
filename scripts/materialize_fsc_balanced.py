"""Materialize balanced processed FSC gzip groups as UTF-8 JSONL inputs."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def materialize(input_dir: Path, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {}
    for language in ("en", "fr"):
        output = output_dir / f"{language}.jsonl"
        docs = words = chars = 0
        with output.open("w", encoding="utf-8", newline="\n") as out:
            for source_path in sorted((input_dir / language).glob("*.jsonl.gz")):
                with gzip.open(source_path, "rt", encoding="utf-8") as stream:
                    for line in stream:
                        row = json.loads(line)
                        out.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                        docs += 1
                        words += len(row["text"].split())
                        chars += len(row["text"])
        report[language] = {"documents": docs, "whitespace_words": words, "characters": chars,
                            "path": str(output), "sha256": sha256_file(output), "bytes": output.stat().st_size}
    manifest = {"kind": "slm.fsc_materialized_documents", "inputs": [str(p) for p in sorted(input_dir.glob("**/*.jsonl.gz"))],
                "languages": report}
    (output_dir / "materialization_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    print(json.dumps(materialize(Path(args.input_dir), Path(args.output_dir)), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
