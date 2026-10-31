"""Encode processed documents into token shards and read them back for training.

Documents are split into train and validation deterministically by a hash of
their id, so the split does not depend on file order or worker count. Each
document is followed by the end-of-text token. Shards are flat binary arrays
(uint16 when the vocabulary fits, else uint32) described by ``manifest.json``.
"""

from __future__ import annotations

import hashlib
import gzip
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from slm import tokenizer as slm_tokenizer
from slm.runlog import code_revision, read_json, sha256_file, write_json


def split_of(doc_id: str, val_fraction: float, salt: str = "slm-split-v1") -> str:
    h = int.from_bytes(hashlib.sha256(f"{salt}:{doc_id}".encode()).digest()[:8], "big")
    return "val" if h / 2**64 < val_fraction else "train"


def encode(
    tokenizer_path: str | Path,
    documents: list[str],
    out_dir: str | Path,
    val_fraction: float = 0.01,
    shard_tokens: int = 100_000_000,
    batch_size: int = 1024,
) -> dict[str, Any]:
    tok = slm_tokenizer.load(tokenizer_path)
    eos = tok.token_to_id(slm_tokenizer.EOS)
    dtype = np.uint16 if tok.get_vocab_size() <= 65535 else np.uint32
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    writers: dict[str, dict[str, Any]] = {
        s: {"buf": [], "buffered": 0, "shards": [], "tokens": 0, "docs": 0} for s in ("train", "val")
    }

    def flush(split: str) -> None:
        w = writers[split]
        if not w["buf"]:
            return
        arr = np.concatenate(w["buf"]).astype(dtype)
        name = f"{split}_{len(w['shards']):05d}.bin"
        arr.tofile(out_dir / name)
        w["shards"].append({"file": name, "tokens": int(arr.size), "sha256": sha256_file(out_dir / name)})
        w["buf"], w["buffered"] = [], 0

    def process(batch: list[dict[str, Any]]) -> None:
        for rec, enc in zip(batch, tok.encode_batch([r["text"] for r in batch])):
            split = split_of(rec["id"], val_fraction)
            ids = np.asarray(enc.ids + [eos], dtype=np.uint32)
            w = writers[split]
            w["buf"].append(ids)
            w["buffered"] += ids.size
            w["tokens"] += ids.size
            w["docs"] += 1
            if w["buffered"] >= shard_tokens:
                flush(split)

    for path in documents:
        batch: list[dict[str, Any]] = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    batch.append(json.loads(line))
                if len(batch) >= batch_size:
                    process(batch)
                    batch = []
        if batch:
            process(batch)
    for split in writers:
        flush(split)

    tok_file = Path(tokenizer_path)
    tok_file = tok_file / "tokenizer.json" if tok_file.is_dir() else tok_file
    manifest = {
        "kind": "slm.tokens",
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dtype": np.dtype(dtype).name,
        "eos_id": eos,
        "vocab_size": tok.get_vocab_size(),
        "tokenizer_sha256": sha256_file(tok_file),
        "val_fraction": val_fraction,
        "inputs": [{"path": p, "sha256": sha256_file(p)} for p in documents],
        "splits": {s: {"tokens": w["tokens"], "docs": w["docs"], "shards": w["shards"]} for s, w in writers.items()},
        "code": code_revision(),
    }
    write_json(out_dir / "manifest.json", manifest)
    return manifest


def encode_fixed_splits(
    tokenizer_path: str | Path,
    split_inputs: dict[str, list[str | Path]],
    out_dir: str | Path,
    shard_tokens: int = 100_000_000,
) -> dict[str, Any]:
    """Encode preassigned corpus partitions without re-splitting documents.

    ``train`` is used for optimization, ``dev`` maps to the trainer's ``val``
    partition, and optional ``test`` remains a held-out corpus split. Gzipped
    JSONL input is streamed so large corpora need not be decompressed to disk.
    """
    tok = slm_tokenizer.load(tokenizer_path)
    eos = tok.token_to_id(slm_tokenizer.EOS)
    dtype = np.uint16 if tok.get_vocab_size() <= 65535 else np.uint32
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    input_map = {"train": "train", "dev": "val", "test": "test"}
    unknown = set(split_inputs) - set(input_map)
    if unknown:
        raise ValueError(f"unsupported fixed corpus splits: {sorted(unknown)}")
    writers: dict[str, dict[str, Any]] = {
        output_split: {"buf": [], "buffered": 0, "shards": [], "tokens": 0, "docs": 0}
        for source_split, output_split in input_map.items() if source_split in split_inputs
    }

    def flush(split: str) -> None:
        writer = writers[split]
        if not writer["buf"]:
            return
        arr = np.concatenate(writer["buf"]).astype(dtype)
        name = f"{split}_{len(writer['shards']):05d}.bin"
        arr.tofile(out_dir / name)
        writer["shards"].append({"file": name, "tokens": int(arr.size), "sha256": sha256_file(out_dir / name)})
        writer["buf"], writer["buffered"] = [], 0

    for source_split, paths in split_inputs.items():
        output_split = input_map[source_split]
        for input_path in paths:
            path = Path(input_path)
            opener = gzip.open if path.suffix == ".gz" else open
            batch: list[dict[str, Any]] = []
            with opener(path, "rt", encoding="utf-8") as stream:
                for line in stream:
                    if line.strip():
                        batch.append(json.loads(line))
                    if len(batch) < 1024:
                        continue
                    _encode_fixed_batch(batch, tok, eos, writers[output_split], dtype, shard_tokens, out_dir, output_split)
                    batch = []
            if batch:
                _encode_fixed_batch(batch, tok, eos, writers[output_split], dtype, shard_tokens, out_dir, output_split)
        flush(output_split)

    tok_file = Path(tokenizer_path)
    tok_file = tok_file / "tokenizer.json" if tok_file.is_dir() else tok_file
    manifest = {
        "kind": "slm.tokens.fixed_splits",
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dtype": np.dtype(dtype).name,
        "eos_id": eos,
        "vocab_size": tok.get_vocab_size(),
        "tokenizer_sha256": sha256_file(tok_file),
        "split_mapping": input_map,
        "shard_tokens_target": shard_tokens,
        "inputs": {
            source_split: [{"path": str(path), "sha256": sha256_file(path)} for path in paths]
            for source_split, paths in split_inputs.items()
        },
        "splits": {split: {"tokens": writer["tokens"], "docs": writer["docs"], "shards": writer["shards"]}
                   for split, writer in writers.items()},
        "code": code_revision(),
    }
    write_json(out_dir / "manifest.json", manifest)
    return manifest


def _encode_fixed_batch(batch, tok, eos, writer, dtype, shard_tokens, out_dir, split):
    for rec, enc in zip(batch, tok.encode_batch([r["text"] for r in batch])):
        ids = np.asarray(enc.ids + [eos], dtype=np.uint32)
        writer["buf"].append(ids)
        writer["buffered"] += ids.size
        writer["tokens"] += ids.size
        writer["docs"] += 1
        if writer["buffered"] >= shard_tokens:
            array = np.concatenate(writer["buf"]).astype(dtype)
            name = f"{split}_{len(writer['shards']):05d}.bin"
            array.tofile(out_dir / name)
            writer["shards"].append({"file": name, "tokens": int(array.size), "sha256": sha256_file(out_dir / name)})
            writer["buf"], writer["buffered"] = [], 0


class TokenShards:
    """Memory-mapped view over the shards of one split, with deterministic window sampling."""

    def __init__(self, data_dir: str | Path, split: str = "train") -> None:
        self.dir = Path(data_dir)
        self.manifest = read_json(self.dir / "manifest.json")
        info = self.manifest["splits"][split]
        dtype = np.dtype(self.manifest["dtype"])
        self.arrays = [np.memmap(self.dir / s["file"], dtype=dtype, mode="r") for s in info["shards"]]
        self.sizes = np.array([a.size for a in self.arrays], dtype=np.int64)
        self.total = int(self.sizes.sum())
        self.vocab_size = int(self.manifest["vocab_size"])
        if self.total == 0:
            raise ValueError(f"split {split!r} in {data_dir} has no tokens")

    def sample(self, rng: np.random.Generator, batch: int, seq_len: int) -> tuple[np.ndarray, np.ndarray]:
        """Random windows of seq_len + 1 tokens; shards are chosen in proportion to size."""
        row_length = int(self.manifest.get("row_length", 0))
        if row_length:
            # Fixed-length records (e.g. one game per row): windows start at a record boundary.
            if seq_len + 1 != row_length:
                raise ValueError(f"row-aligned data needs seq_len + 1 == row_length ({row_length})")
            rows = self.sizes // row_length
            shard_ids = rng.choice(len(self.arrays), size=batch, p=rows / rows.sum())
            out = np.empty((batch, row_length), dtype=np.int64)
            for row, sid in enumerate(shard_ids):
                start = int(rng.integers(0, rows[sid])) * row_length
                out[row] = self.arrays[sid][start : start + row_length]
            return out[:, :-1], out[:, 1:]
        eligible = np.maximum(self.sizes - seq_len - 1, 0)
        if eligible.sum() == 0:
            raise ValueError(f"no shard is longer than seq_len + 1 = {seq_len + 1} tokens")
        shard_ids = rng.choice(len(self.arrays), size=batch, p=eligible / eligible.sum())
        out = np.empty((batch, seq_len + 1), dtype=np.int64)
        for row, sid in enumerate(shard_ids):
            start = int(rng.integers(0, eligible[sid] + 1))
            out[row] = self.arrays[sid][start : start + seq_len + 1]
        return out[:, :-1], out[:, 1:]
