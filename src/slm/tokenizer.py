"""Byte-level BPE tokenizer training with a provenance manifest.

Design choices (see docs/ARCHITECTURE.md): byte-level so no input is ever
out-of-vocabulary, NFC normalisation, digits split individually for arithmetic,
and reserved special tokens for chat, reasoning and tool calls so that post
training does not require resizing the embeddings.
"""

from __future__ import annotations

import json
import random
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from tokenizers import Regex, Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers

from slm.runlog import code_revision, environment_info, sha256_file, write_json

EOS = "<|endoftext|>"
SPECIAL_TOKENS = [
    EOS,
    "<|pad|>",
    "<|im_start|>",
    "<|im_end|>",
    "<think>",
    "</think>",
    "<tool_call>",
    "</tool_call>",
    "<tool_response>",
    "</tool_response>",
    *[f"<|reserved_{i}|>" for i in range(22)],
]


def build_empty() -> Tokenizer:
    tok = Tokenizer(models.BPE())
    tok.normalizer = normalizers.NFC()
    tok.pre_tokenizer = pre_tokenizers.Sequence(
        [
            pre_tokenizers.Split(Regex(r"\p{N}"), behavior="isolated"),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True),
        ]
    )
    tok.decoder = decoders.ByteLevel()
    return tok


def _lines(paths: list[str]) -> Iterator[str]:
    for p in paths:
        with open(p, encoding="utf-8") as f:
            yield from (line for line in f if line.strip())


def iter_texts(paths: list[str], max_docs: int | None, seed: int) -> Iterator[str]:
    """Deterministic sample of ``max_docs`` documents chosen with ``seed``, streamed in two passes
    so the corpus never has to fit in memory."""
    if not max_docs:
        yield from (json.loads(line)["text"] for line in _lines(paths))
        return
    total = sum(1 for _ in _lines(paths))
    chosen = set(random.Random(seed).sample(range(total), min(max_docs, total)))
    yield from (json.loads(line)["text"] for i, line in enumerate(_lines(paths)) if i in chosen)


def train(
    inputs: list[str],
    out_dir: str | Path,
    vocab_size: int = 49152,
    min_frequency: int = 2,
    max_docs: int | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tok = build_empty()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    tok.train_from_iterator(iter_texts(inputs, max_docs, seed), trainer=trainer)
    path = out_dir / "tokenizer.json"
    tok.save(str(path))
    manifest = {
        "kind": "slm.tokenizer",
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "params": {"vocab_size": vocab_size, "min_frequency": min_frequency, "max_docs": max_docs, "seed": seed},
        "actual_vocab_size": tok.get_vocab_size(),
        "special_tokens": SPECIAL_TOKENS,
        "inputs": [{"path": p, "sha256": sha256_file(p)} for p in inputs],
        "tokenizer_sha256": sha256_file(path),
        "code": code_revision(),
        "environment": environment_info(),
    }
    write_json(out_dir / "manifest.json", manifest)
    return manifest


def load(path: str | Path) -> Tokenizer:
    path = Path(path)
    if path.is_dir():
        path = path / "tokenizer.json"
    return Tokenizer.from_file(str(path))


def fertility(tok: Tokenizer, texts: list[str]) -> dict[str, float]:
    """Tokens per word and bytes per token: compare tokenizers across languages."""
    n_tok = sum(len(e.ids) for e in tok.encode_batch(texts))
    n_words = sum(len(t.split()) for t in texts)
    n_bytes = sum(len(t.encode("utf-8")) for t in texts)
    return {"tokens_per_word": n_tok / max(1, n_words), "bytes_per_token": n_bytes / max(1, n_tok)}
