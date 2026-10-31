#!/usr/bin/env python3
"""Encode an EN/FR educational corpus candidate with the pinned SmolLM2 tokenizer (EXP-0034)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
# torch (via slm) must load before pyarrow on Windows, or c10.dll fails with WinError 1114.
from slm.data.shards import encode_fixed_splits  # noqa: E402
from slm.runlog import sha256_file as hash_file  # noqa: E402

CORPUS = ROOT / "data/processed/slm-tavenel-edu-corpus-v1"
TOKENIZER = ROOT / "data/tokenizers/smollm2-f8027fd0"
OUT = ROOT / "data/tokens/slm-tavenel-edu-corpus-v1-smollm2"


def encode(corpus: Path, out: Path) -> None:
    corpus_manifest = json.loads((corpus / "manifest.json").read_text(encoding="utf-8"))
    tokenizer_manifest = json.loads((TOKENIZER / "manifest.json").read_text(encoding="utf-8"))
    if hash_file(TOKENIZER / "tokenizer.json") != tokenizer_manifest["tokenizer_sha256"]:
        raise RuntimeError("Tokenizer file does not match its pinned manifest")
    for split in ("train", "dev", "test"):
        if hash_file(corpus / f"{split}.jsonl.gz") != corpus_manifest["files"][split]["sha256"]:
            raise RuntimeError(f"Corpus {split} split hash mismatch")
    manifest = encode_fixed_splits(TOKENIZER / "tokenizer.json",
                                   {s: [corpus / f"{s}.jsonl.gz"] for s in ("train", "dev", "test")},
                                   out, shard_tokens=200_000_000)
    manifest["corpus_candidate_manifest_sha256"] = hash_file(corpus / "manifest.json")
    manifest["tokenizer_manifest_sha256"] = hash_file(TOKENIZER / "manifest.json")
    manifest["tokenizer_source"] = {k: tokenizer_manifest[k] for k in ("source", "revision", "license")}
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({s: {"docs": v["docs"], "tokens": v["tokens"]} for s, v in manifest["splits"].items()}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=CORPUS)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    encode(args.corpus, args.out)


if __name__ == "__main__":
    main()
