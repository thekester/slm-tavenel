"""Exact and approximate (MinHash LSH) deduplication.

This in-memory implementation is meant for research-scale corpora and for testing
the pipeline. For trillion-token runs the same parameters should be applied with a
distributed implementation such as DataTrove; the parameters are recorded in the
manifest either way.
"""

from __future__ import annotations

import hashlib
import re
import zlib

import numpy as np

_MERSENNE = np.uint64((1 << 61) - 1)
_MAX_HASH = np.uint64((1 << 32) - 1)
_TOKEN = re.compile(r"\w+", re.UNICODE)


def normalized_hash(text: str) -> str:
    """Hash after lowercasing and collapsing whitespace, so trivial variants collide."""
    norm = " ".join(text.lower().split())
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()


class ExactDedup:
    def __init__(self) -> None:
        self.seen: dict[str, str] = {}

    def check(self, doc_id: str, text: str) -> str | None:
        """Return the id of an earlier identical document, or None (and remember this one)."""
        h = normalized_hash(text)
        if h in self.seen:
            return self.seen[h]
        self.seen[h] = doc_id
        return None


class MinHashLSH:
    """MinHash signatures over word n-gram shingles with banded LSH.

    With ``bands`` x ``rows`` = ``num_perm``, two documents with Jaccard similarity s
    become candidates with probability 1 - (1 - s**rows)**bands. Candidates are then
    confirmed by comparing estimated similarity against ``threshold``.
    """

    def __init__(self, num_perm: int = 128, bands: int = 16, ngram: int = 5, threshold: float = 0.8, seed: int = 0) -> None:
        if num_perm % bands:
            raise ValueError("num_perm must be divisible by bands")
        self.num_perm, self.bands, self.rows = num_perm, bands, num_perm // bands
        self.ngram, self.threshold = ngram, threshold
        rng = np.random.default_rng(seed)
        # a, b < 2**31 and shingle hashes < 2**32 keep a * x + b below 2**64: no overflow.
        self.a = rng.integers(1, 1 << 31, size=num_perm, dtype=np.uint64)
        self.b = rng.integers(0, 1 << 31, size=num_perm, dtype=np.uint64)
        self.buckets: list[dict[bytes, str]] = [{} for _ in range(bands)]
        self.signatures: dict[str, np.ndarray] = {}

    def shingles(self, text: str) -> np.ndarray:
        toks = _TOKEN.findall(text.lower())
        n = self.ngram
        grams = {" ".join(toks[i : i + n]) for i in range(max(1, len(toks) - n + 1))}
        return np.fromiter((zlib.crc32(g.encode("utf-8")) for g in grams), dtype=np.uint64, count=len(grams))

    def signature(self, text: str) -> np.ndarray:
        x = self.shingles(text)
        if x.size == 0:
            return np.full(self.num_perm, _MAX_HASH, dtype=np.uint64)
        h = (self.a[:, None] * x[None, :] + self.b[:, None]) % _MERSENNE
        return (h & _MAX_HASH).min(axis=1)

    @staticmethod
    def similarity(s1: np.ndarray, s2: np.ndarray) -> float:
        return float((s1 == s2).mean())

    def check(self, doc_id: str, text: str) -> tuple[str, float] | None:
        """Return (earlier_id, estimated_jaccard) for a near-duplicate, or None and index this doc."""
        sig = self.signature(text)
        keys = [sig[i * self.rows : (i + 1) * self.rows].tobytes() for i in range(self.bands)]
        best: tuple[str, float] | None = None
        for band, key in enumerate(keys):
            other = self.buckets[band].get(key)
            if other is not None:
                sim = self.similarity(sig, self.signatures[other])
                if sim >= self.threshold and (best is None or sim > best[1]):
                    best = (other, sim)
        if best is not None:
            return best
        for band, key in enumerate(keys):
            self.buckets[band].setdefault(key, doc_id)
        self.signatures[doc_id] = sig
        return None
