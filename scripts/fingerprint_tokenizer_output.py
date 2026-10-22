"""Fingerprint token IDs for a small private text sample without printing source text."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import tokenizers

from slm.tokenizer import load


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("documents", type=Path)
    parser.add_argument("tokenizer", type=Path)
    parser.add_argument("--limit", type=int, default=128)
    args = parser.parse_args()
    if args.limit <= 0:
        parser.error("--limit must be positive")
    tok = load(args.tokenizer)
    texts = []
    with args.documents.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                texts.append(json.loads(line)["text"])
                if len(texts) >= args.limit:
                    break
    encodings = tok.encode_batch(texts)
    ids = np.asarray([token for enc in encodings for token in enc.ids], dtype=np.uint32)
    print(json.dumps({"tokenizers_version": tokenizers.__version__, "documents": len(texts),
                      "tokens": int(ids.size), "ids_sha256": hashlib.sha256(ids.tobytes()).hexdigest()}, indent=2))


if __name__ == "__main__":
    main()
