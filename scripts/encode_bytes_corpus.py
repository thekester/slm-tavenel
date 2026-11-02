#!/usr/bin/env python3
"""Encode a corpus candidate as raw UTF-8 bytes for byte-level models (EXP-0040, H-Net pilot).

Each document becomes its bytes shifted by one (byte b -> id b + 1) followed by id 0 (end of text),
stored as uint16 shards with the same manifest layout as the BPE token shards, and the byte
"tokenizer" definition (data/tokenizers/bytes-v1/tokenizer.json) hashed like any tokenizer.
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from slm.runlog import code_revision, sha256_file  # noqa: E402

TOKENIZER = ROOT / "data/tokenizers/bytes-v1/tokenizer.json"


def encode_split(src: Path, out: Path, split: str, shard_tokens: int) -> dict:
    buf, buffered, shards, tokens, docs = [], 0, [], 0, 0

    def flush() -> None:
        nonlocal buf, buffered
        if not buf:
            return
        arr = np.concatenate(buf)
        name = f"{split}_{len(shards):05d}.bin"
        arr.tofile(out / name)
        shards.append({"file": name, "tokens": int(arr.size), "sha256": sha256_file(out / name)})
        buf, buffered = [], 0

    with gzip.open(src, "rt", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            text = json.loads(line).get("text")
            if not isinstance(text, str) or not text:
                continue
            ids = np.empty(len(text.encode("utf-8")) + 1, dtype=np.uint16)
            ids[:-1] = np.frombuffer(text.encode("utf-8"), dtype=np.uint8)
            ids[:-1] += 1
            ids[-1] = 0
            buf.append(ids)
            buffered += ids.size
            tokens += ids.size
            docs += 1
            if buffered >= shard_tokens:
                flush()
    flush()
    return {"tokens": tokens, "docs": docs, "shards": shards}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=ROOT / "data/processed/slm-tavenel-edu-corpus-v1")
    parser.add_argument("--out", type=Path, default=ROOT / "data/tokens/slm-tavenel-edu-corpus-v1-bytes")
    parser.add_argument("--shard-tokens", type=int, default=1_000_000_000)
    args = parser.parse_args()
    corpus_manifest = json.loads((args.corpus / "manifest.json").read_text(encoding="utf-8"))
    for split in ("train", "dev", "test"):
        if sha256_file(args.corpus / f"{split}.jsonl.gz") != corpus_manifest["files"][split]["sha256"]:
            raise RuntimeError(f"Corpus {split} split hash mismatch")
    args.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    splits = {}
    for source, name in (("train", "train"), ("dev", "val"), ("test", "test")):
        splits[name] = encode_split(args.corpus / f"{source}.jsonl.gz", args.out, name, args.shard_tokens)
        print(name, splits[name]["docs"], "docs", splits[name]["tokens"], "bytes", round(time.time() - t0), "s", flush=True)
    manifest = {
        "kind": "slm.tokens.bytes", "dtype": "uint16", "eos_id": 0, "vocab_size": 257,
        "tokenizer_sha256": sha256_file(TOKENIZER), "tokenizer": "data/tokenizers/bytes-v1/tokenizer.json",
        "split_mapping": {"train": "train", "dev": "val", "test": "test"},
        "corpus_candidate_manifest_sha256": sha256_file(args.corpus / "manifest.json"),
        "splits": splits, "code": code_revision(),
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
