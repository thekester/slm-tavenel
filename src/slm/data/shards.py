"""Encode processed documents into token shards and read them back for training.

Documents are split into train and validation deterministically by a hash of
their id, so the split does not depend on file order or worker count. Each
document is followed by the end-of-text token. Shards are flat binary arrays
(uint16 when the vocabulary fits, else uint32) described by ``manifest.json``.
"""

from __future__ import annotations

import hashlib
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
        eligible = np.maximum(self.sizes - seq_len - 1, 0)
        if eligible.sum() == 0:
            raise ValueError(f"no shard is longer than seq_len + 1 = {seq_len + 1} tokens")
        shard_ids = rng.choice(len(self.arrays), size=batch, p=eligible / eligible.sum())
        out = np.empty((batch, seq_len + 1), dtype=np.int64)
        for row, sid in enumerate(shard_ids):
            start = int(rng.integers(0, eligible[sid] + 1))
            out[row] = self.arrays[sid][start : start + seq_len + 1]
        return out[:, :-1], out[:, 1:]
