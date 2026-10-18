"""Benchmark contamination checks by word n-gram overlap.

Evaluation texts are indexed once; a training document is flagged when it shares
at least ``min_matches`` n-grams with any evaluation item. The default of 13-grams
follows the GPT-3 decontamination procedure. The index records which benchmark
items matched so that contamination can be reported per benchmark.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

_TOKEN = re.compile(r"\w+", re.UNICODE)


def ngrams(text: str, n: int) -> set[tuple[str, ...]]:
    toks = _TOKEN.findall(text.lower())
    return {tuple(toks[i : i + n]) for i in range(len(toks) - n + 1)}


class ContaminationIndex:
    def __init__(self, n: int = 13, min_matches: int = 1) -> None:
        self.n = n
        self.min_matches = min_matches
        self.index: dict[tuple[str, ...], set[str]] = defaultdict(set)
        self.items = 0

    def add(self, item_id: str, text: str) -> None:
        for g in ngrams(text, self.n):
            self.index[g].add(item_id)
        self.items += 1

    def add_jsonl(self, path: str | Path, fields: Iterable[str] = ("text",), id_field: str = "id") -> None:
        name = Path(path).stem
        with open(path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if not line.strip():
                    continue
                rec = json.loads(line)
                text = "\n".join(str(rec[k]) for k in fields if k in rec)
                self.add(f"{name}:{rec.get(id_field, i)}", text)

    def matches(self, text: str) -> dict[str, int]:
        """Evaluation items sharing n-grams with ``text`` and the number of shared n-grams."""
        hits: dict[str, int] = defaultdict(int)
        for g in ngrams(text, self.n):
            for item in self.index.get(g, ()):
                hits[item] += 1
        return {k: v for k, v in hits.items() if v >= self.min_matches}
