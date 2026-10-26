#!/usr/bin/env python3
"""Tokenize the pinned growth-candidate-v2 corpus using its fixed source splits."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from slm.data.shards import encode_fixed_splits

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "data/processed/slm-tavenel-growth-candidate-v2"
TOKENS = ROOT / "data/tokens/slm-tavenel-growth-candidate-v2"
TOKENIZER = ROOT / "exports/slm-tavenel-1m-exp0010-release-candidate-v4/tokenizer.json"
TOKENIZER_SHA = "93c7b9364de2c42315571c851353862e63830bfbeb220dc14dde80e6ca82cfcb"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    corpus_manifest = json.loads((CORPUS / "manifest.json").read_text(encoding="utf-8"))
    if corpus_manifest.get("status") != "candidate_not_legal_clearance":
        raise RuntimeError("Unexpected corpus status")
    if sha256(TOKENIZER) != TOKENIZER_SHA:
        raise RuntimeError("Pinned SLM-Tavenel tokenizer hash mismatch")
    for split in ("train", "dev", "test"):
        file = CORPUS / f"{split}.jsonl.gz"
        expected = corpus_manifest["files"][split]["sha256"]
        if sha256(file) != expected:
            raise RuntimeError(f"Corpus {split} split hash mismatch")

    manifest = encode_fixed_splits(
        TOKENIZER,
        {
            "train": [CORPUS / "train.jsonl.gz"],
            "dev": [CORPUS / "dev.jsonl.gz"],
            "test": [CORPUS / "test.jsonl.gz"],
        },
        TOKENS,
        shard_tokens=100_000_000,
    )
    manifest["corpus_candidate_manifest_sha256"] = sha256(CORPUS / "manifest.json")
    (TOKENS / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": TOKENS.relative_to(ROOT).as_posix(),
        "splits": {name: {"docs": info["docs"], "tokens": info["tokens"], "shards": len(info["shards"])}
                   for name, info in manifest["splits"].items()},
        "tokenizer_sha256": manifest["tokenizer_sha256"],
        "corpus_candidate_manifest_sha256": manifest["corpus_candidate_manifest_sha256"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
