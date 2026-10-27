#!/usr/bin/env python3
"""Encode the pinned HPLT-rich corpus with the tokenizer used by the 100M parent."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from slm.data.shards import encode_fixed_splits

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=ROOT / "data/processed/slm-tavenel-growth-candidate-v2")
    parser.add_argument("--tokenizer", type=Path, default=ROOT / "data/tokenizer/fsc-pilot-v1-train-only/tokenizer.json")
    parser.add_argument("--tokenizer-manifest", type=Path,
                        default=ROOT / "data/tokenizer/fsc-pilot-v1-train-only/manifest.json")
    parser.add_argument("--out", type=Path,
                        default=ROOT / "data/tokens/slm-tavenel-growth-candidate-v2-fsc-tokenizer")
    args = parser.parse_args()
    corpus_manifest_path = args.corpus / "manifest.json"
    corpus_manifest = json.loads(corpus_manifest_path.read_text(encoding="utf-8"))
    if corpus_manifest.get("status") != "candidate_not_legal_clearance":
        raise RuntimeError("Unexpected corpus review status")
    tokenizer_manifest = json.loads(args.tokenizer_manifest.read_text(encoding="utf-8"))
    expected_tokenizer_sha = tokenizer_manifest["tokenizer_sha256"]
    actual_tokenizer_sha = sha256(args.tokenizer)
    if actual_tokenizer_sha != expected_tokenizer_sha:
        raise RuntimeError("Tokenizer file does not match its pinned manifest")
    for split in ("train", "dev", "test"):
        path = args.corpus / f"{split}.jsonl.gz"
        expected = corpus_manifest["files"][split]["sha256"]
        if sha256(path) != expected:
            raise RuntimeError(f"Corpus {split} split hash mismatch")

    manifest = encode_fixed_splits(
        args.tokenizer,
        {split: [args.corpus / f"{split}.jsonl.gz"] for split in ("train", "dev", "test")},
        args.out,
        shard_tokens=100_000_000,
    )
    manifest["corpus_candidate_manifest_sha256"] = sha256(corpus_manifest_path)
    manifest["tokenizer_manifest_sha256"] = sha256(args.tokenizer_manifest)
    manifest["tokenizer_sha256"] = actual_tokenizer_sha
    manifest["encoding_note"] = "Re-encoded with the tokenizer lineage recorded for the EXP-0024 100M parent."
    (args.out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": args.out.relative_to(ROOT).as_posix() if args.out.is_relative_to(ROOT) else str(args.out),
        "splits": {name: {"docs": info["docs"], "tokens": info["tokens"], "shards": len(info["shards"])}
                   for name, info in manifest["splits"].items()},
        "tokenizer_sha256": actual_tokenizer_sha,
        "corpus_candidate_manifest_sha256": manifest["corpus_candidate_manifest_sha256"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
