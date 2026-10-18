"""Document-level quality filters and text transforms.

Each filter returns ``None`` to keep a document or a short reason string to drop it;
reasons are written to the decision log so that every removal can be audited.
Thresholds follow the heuristics published with Gopher (Rae et al., 2021) and
FineWeb (Penedo et al., 2024) and are configuration, not constants: changing them
is an experiment and must be logged as one.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from collections.abc import Callable
from typing import Any

Doc = dict[str, Any]
Filter = Callable[[Doc], str | None]
Transform = Callable[[Doc], Doc]

_WORD = re.compile(r"\w+", re.UNICODE)
_BULLET = re.compile(r"^\s*([-*•‣◦▪]|\d+[.)])\s")


def words(text: str) -> list[str]:
    return _WORD.findall(text)


# ---- transforms -------------------------------------------------------------------------


def normalize(doc: Doc) -> Doc:
    """Unicode NFC, unified newlines, trailing whitespace stripped, no runs of >2 blank lines."""
    text = unicodedata.normalize("NFC", doc["text"]).replace("\r\n", "\n").replace("\r", "\n")
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return {**doc, "text": text}


# ---- filters ----------------------------------------------------------------------------


def gopher_quality(
    min_words: int = 50,
    max_words: int = 100_000,
    min_mean_word_len: float = 3.0,
    max_mean_word_len: float = 10.0,
    max_symbol_ratio: float = 0.1,
    max_bullet_lines: float = 0.9,
    max_ellipsis_lines: float = 0.3,
    min_alpha_words: float = 0.8,
) -> Filter:
    def f(doc: Doc) -> str | None:
        text = doc["text"]
        ws = text.split()
        n = len(ws)
        if n < min_words:
            return "too_few_words"
        if n > max_words:
            return "too_many_words"
        mean_len = sum(len(w) for w in ws) / n
        if not min_mean_word_len <= mean_len <= max_mean_word_len:
            return "mean_word_length"
        symbols = text.count("#") + text.count("...") + text.count("…")
        if symbols / n > max_symbol_ratio:
            return "symbol_ratio"
        lines = [line for line in text.split("\n") if line.strip()]
        if lines:
            if sum(bool(_BULLET.match(line)) for line in lines) / len(lines) > max_bullet_lines:
                return "bullet_lines"
            ellipsis = sum(line.rstrip().endswith(("...", "…")) for line in lines)
            if ellipsis / len(lines) > max_ellipsis_lines:
                return "ellipsis_lines"
        if sum(any(c.isalpha() for c in w) for w in ws) / n < min_alpha_words:
            return "alpha_words"
        return None

    return f


def repetition(
    max_dup_line_frac: float = 0.3,
    max_dup_line_char_frac: float = 0.2,
    top_ngram: tuple[int, float] = (2, 0.2),
    dup_ngram: tuple[int, float] = (5, 0.15),
) -> Filter:
    """Gopher repetition removal: duplicated lines and over-represented n-grams."""

    def f(doc: Doc) -> str | None:
        text = doc["text"]
        lines = [line for line in text.split("\n") if line.strip()]
        if lines:
            counts = Counter(lines)
            dup_lines = sum(c for c in counts.values() if c > 1)
            if dup_lines / len(lines) > max_dup_line_frac:
                return "dup_line_frac"
            dup_chars = sum(len(line) * c for line, c in counts.items() if c > 1)
            if dup_chars / max(1, sum(len(line) for line in lines)) > max_dup_line_char_frac:
                return "dup_line_char_frac"
        ws = words(text.lower())
        total_chars = sum(len(w) for w in ws) or 1
        n, limit = top_ngram
        grams = Counter(tuple(ws[i : i + n]) for i in range(len(ws) - n + 1))
        if grams:
            gram, count = grams.most_common(1)[0]
            if count > 1 and count * sum(len(w) for w in gram) / total_chars > limit:
                return f"top_{n}gram"
        n, limit = dup_ngram
        grams = Counter(tuple(ws[i : i + n]) for i in range(len(ws) - n + 1))
        covered = [False] * len(ws)
        for i in range(len(ws) - n + 1):
            if grams[tuple(ws[i : i + n])] > 1:
                for j in range(i, i + n):
                    covered[j] = True
        if sum(len(w) for w, c in zip(ws, covered) if c) / total_chars > limit:
            return f"dup_{n}gram"
        return None

    return f


def blocklist(patterns: list[str]) -> Filter:
    """Drop documents matching any case-insensitive regular expression."""
    compiled = [re.compile(p, re.IGNORECASE) for p in patterns]

    def f(doc: Doc) -> str | None:
        for p in compiled:
            if p.search(doc["text"]):
                return f"blocklist:{p.pattern}"
        return None

    return f


def min_length(min_chars: int = 200) -> Filter:
    def f(doc: Doc) -> str | None:
        return "too_short" if len(doc["text"]) < min_chars else None

    return f
