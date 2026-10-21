"""Encode fixed document splits without leaking the sealed test into the trainer."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--shard-tokens", type=int, default=25_000_000)
    args = parser.parse_args()

    if args.shard_tokens <= 0:
        parser.error("--shard-tokens must be positive")
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError("Token output directory must be empty to avoid stale training shards")
    tok = Tokenizer.from_file(str(args.tokenizer))
    eos = tok.token_to_id("<|endoftext|>")
    if eos is None:
        raise ValueError("tokenizer must define <|endoftext|>")
    dtype = np.uint16 if tok.get_vocab_size() <= 65535 else np.uint32
    dtype_name = "uint16" if dtype is np.uint16 else "uint32"
    args.out.mkdir(parents=True, exist_ok=True)
    split_rows = {}
    for split in ("train", "dev", "test"):
        split_name = "val" if split == "dev" else split
        shard_meta = []
        buffers: list[np.ndarray] = []
        buffered = 0
        docs = tokens = shard_index = 0

        def flush() -> None:
            nonlocal buffers, buffered, shard_index
            if not buffers:
                return
            array = np.concatenate(buffers).astype(dtype, copy=False)
            name = f"{split_name}_{shard_index:05d}.bin"
            target = args.out / name
            array.tofile(target)
            shard_meta.append({"file": name, "tokens": int(array.size), "sha256": sha256(target)})
            buffers, buffered = [], 0
            shard_index += 1

        for language in ("en", "fr"):
            source = args.split_dir / f"{language}-{split}.jsonl"
            with source.open(encoding="utf-8") as stream:
                for line in stream:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    ids = tok.encode(row["text"]).ids
                    piece = np.asarray([*ids, eos], dtype=np.uint32)
                    buffers.append(piece)
                    buffered += len(piece)
                    docs += 1
                    tokens += len(piece)
                    if buffered >= args.shard_tokens:
                        flush()
        flush()
        split_rows[split_name] = {"tokens": tokens, "docs": docs, "shards": shard_meta}

    tokenizer_manifest = args.tokenizer.parent / "manifest.json"
    manifest = {
        "kind": "slm.tokens",
        "dtype": dtype_name,
        "eos_id": eos,
        "vocab_size": tok.get_vocab_size(),
        "tokenizer_sha256": sha256(args.tokenizer),
        "inputs": [{"path": str(args.split_manifest), "sha256": sha256(args.split_manifest)}],
        "tokenizer_manifest_sha256": sha256(tokenizer_manifest) if tokenizer_manifest.is_file() else None,
        "splits": split_rows,
        "sealed_test_scored": False,
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({name: {"tokens": row["tokens"], "docs": row["docs"]}
                      for name, row in split_rows.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
