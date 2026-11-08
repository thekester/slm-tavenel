#!/usr/bin/env python3
"""Tokenized English/French translation pairs for EXP-0043 (good-view objective and its control).

Source: Tatoeba English-French pairs as packaged by `sentence-transformers/parallel-sentences-tatoeba`
(revision cec1343a, config en-fr, train split); Tatoeba sentences are CC-BY 2.0 FR (tatoeba.org). Steps:
exact dedup of (en, fr) pairs; removal of pairs whose English or French sentence (lower-cased, >= 5 words)
occurs inside any Belebele passage (the evaluation set, FLORES-200 passages); tokenization with the pinned
SmolLM2 tokenizer, truncation to 64 tokens, right-padding with the end-of-text id; a seeded 2,000-pair
dev split. Writes train.npz, dev.npz and manifest.json (counts, hashes, licence, limitations).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from slm.runlog import sha256_file  # noqa: E402

SOURCE = ("sentence-transformers/parallel-sentences-tatoeba", "cec1343a", "en-fr/train-00000-of-00001.parquet")
TOKENIZER = ROOT / "data/tokenizers/smollm2-f8027fd0/tokenizer.json"
MAX_TOKENS = 64


def belebele_text() -> str:
    parts = []
    for language in ("en", "fr"):
        for line in (ROOT / f"data/benchmarks/belebele_{language}.jsonl").read_text(encoding="utf-8").splitlines():
            parts.append(json.loads(line)["context"].lower())
    return "\n".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "data/parallel/tatoeba-en-fr-v1")
    args = parser.parse_args()
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer

    path = Path(hf_hub_download(SOURCE[0], SOURCE[2], repo_type="dataset", revision=SOURCE[1],
                                local_dir=ROOT / "data/raw/tatoeba-en-fr"))
    table = pq.read_table(path)
    pairs = list(zip(table.column("english").to_pylist(), table.column("non_english").to_pylist()))
    counts = {"input": len(pairs)}
    seen, unique = set(), []
    for en, fr in pairs:
        key = (en.strip(), fr.strip())
        if key not in seen and key[0] and key[1]:
            seen.add(key)
            unique.append(key)
    counts["after_dedup"] = len(unique)
    evaluation = belebele_text()
    kept = [(en, fr) for en, fr in unique
            if not any(len(s.split()) >= 5 and s.lower() in evaluation for s in (en, fr))]
    counts["removed_belebele_overlap"] = len(unique) - len(kept)
    random.Random(20261108).shuffle(kept)
    tokenizer = Tokenizer.from_file(str(TOKENIZER))
    eos = tokenizer.token_to_id("<|endoftext|>")
    if eos is None:
        raise SystemExit("tokenizer has no <|endoftext|> token")

    def encode(texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
        ids = [enc.ids[:MAX_TOKENS] for enc in tokenizer.encode_batch(texts)]
        arr = np.full((len(ids), MAX_TOKENS), eos, dtype=np.int32)
        for i, row in enumerate(ids):
            arr[i, : len(row)] = row
        return arr, np.array([max(1, len(r)) for r in ids], dtype=np.int32)

    args.out.mkdir(parents=True, exist_ok=True)
    splits = {"dev": kept[:2000], "train": kept[2000:]}
    files = {}
    for name, rows in splits.items():
        en, en_len = encode([r[0] for r in rows])
        fr, fr_len = encode([r[1] for r in rows])
        np.savez(args.out / f"{name}.npz", en=en, fr=fr, en_len=en_len, fr_len=fr_len)
        files[name] = {"pairs": len(rows), "sha256": sha256_file(args.out / f"{name}.npz"),
                       "mean_tokens_en": round(float(en_len.mean()), 2), "mean_tokens_fr": round(float(fr_len.mean()), 2),
                       "truncated_pairs": int(((en_len >= MAX_TOKENS) | (fr_len >= MAX_TOKENS)).sum())}
    manifest = {
        "schema": "slm-tavenel.parallel_pairs.v1",
        "source": {"dataset": SOURCE[0], "revision": SOURCE[1], "file": SOURCE[2], "file_sha256": sha256_file(path),
                   "declared_license": "Tatoeba sentences: CC-BY 2.0 FR (tatoeba.org); packaging card declares none",
                   "attribution": "Tatoeba contributors, https://tatoeba.org"},
        "tokenizer_sha256": hashlib.sha256(TOKENIZER.read_bytes()).hexdigest(),
        "pad_id": int(eos), "max_tokens": MAX_TOKENS, "counts": counts, "files": files,
        "limitations": ["Short everyday sentences (about 7 words), unlike the long evaluation passages.",
                        "Overlap removal is exact substring matching on sentences of >= 5 words only.",
                        "Per-sentence authorship and licence are those declared by Tatoeba; not independently verified."],
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"counts": counts, "files": files}, indent=2))


if __name__ == "__main__":
    main()
