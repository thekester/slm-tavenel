"""Assemble language token shards while keeping the sealed test split out of training data."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import shutil
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_shards(root: Path, split: str, output: Path, prefix: str) -> list[dict]:
    source = root / prefix / split
    candidates = sorted(source.glob("train_*.bin"))
    if not candidates:
        raise FileNotFoundError(f"No token shards found under {source}")
    result = []
    split_prefix = "train" if split == "train" else "val"
    source_manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    source_shards = {row["file"]: row for row in source_manifest["splits"]["train"]["shards"]}
    for i, path in enumerate(candidates):
        target = output / f"{split_prefix}_{prefix}_{i:05d}.bin"
        shutil.copyfile(path, target)
        result.append({"source": str(path), "source_sha256": sha256_file(path),
                       "path": str(target), "sha256": sha256_file(target), "bytes": target.stat().st_size,
                       "tokens": source_shards[path.name]["tokens"],
                       "docs": source_manifest["splits"]["train"]["docs"] if len(candidates) == 1 else None})
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--token-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tokenizer-manifest", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("Token output directory must be empty to avoid stale training shards")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train = [item for language in ("en", "fr") for item in copy_shards(args.token_root, "train", args.output_dir, language)]
    dev = [item for language in ("en", "fr") for item in copy_shards(args.token_root, "dev", args.output_dir, language)]
    # The test token shards are deliberately not copied into the trainer's data directory.
    tokenizer_info = json.loads(args.tokenizer_manifest.read_text(encoding="utf-8"))
    for split_rows in (train, dev):
        for row in split_rows:
            row["file"] = Path(row["path"]).name
    token_manifest = {
        "kind": "slm.tokens",
        "created": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "dtype": "uint16",
        "eos_id": 0,
        "vocab_size": tokenizer_info["actual_vocab_size"],
        "tokenizer_sha256": tokenizer_info["tokenizer_sha256"],
        "val_fraction": 0.05,
        "inputs": [{"path": str(args.split_manifest), "sha256": sha256_file(args.split_manifest)}],
        "splits": {
            "train": {"tokens": sum(row["tokens"] for row in train), "docs": sum(row.get("docs") or 0 for row in train),
                      "shards": [{"file": row["file"], "tokens": row["tokens"], "sha256": row["sha256"]} for row in train]},
            "val": {"tokens": sum(row["tokens"] for row in dev), "docs": sum(row.get("docs") or 0 for row in dev),
                    "shards": [{"file": row["file"], "tokens": row["tokens"], "sha256": row["sha256"]} for row in dev]},
        },
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(token_manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    report = {
        "kind": "slm.open_weight_training_tokens",
        "tokenizer_manifest": str(args.tokenizer_manifest),
        "tokenizer_manifest_sha256": sha256_file(args.tokenizer_manifest),
        "split_manifest": str(args.split_manifest),
        "split_manifest_sha256": sha256_file(args.split_manifest),
        "train_shards": train,
        "dev_shards": dev,
        "test_included": False,
        "token_manifest_sha256": sha256_file(args.output_dir / "manifest.json"),
        "notes": "Language-specific documents are tokenized independently and then sampled jointly; test files remain separate.",
    }
    out = args.output_dir / "assembly_manifest.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output_dir), "train_shards": len(train), "dev_shards": len(dev),
                      "train_tokens": sum(row["tokens"] for row in train),
                      "dev_tokens": sum(row["tokens"] for row in dev),
                      "test_included": False}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
