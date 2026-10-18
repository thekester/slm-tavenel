"""Deterministic synthetic data with verifiable outcomes.

Two generators, both seeded and dependency-free:

- ``prose``: bilingual (English/French) template text used to exercise the
  pipeline and to smoke-test training end to end;
- ``arithmetic``: tasks whose answers are checked by ``verify_arithmetic``.

Every record carries the generator name, version, seed and parameters, as the
README requires for synthetic data. Teacher-model generators will follow the same
record format with the teacher identity and generation settings added.
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any

GENERATOR_VERSION = "1"

_EN_SUBJECTS = ["The river", "A small model", "The library", "Every experiment", "The old bridge",
                "This dataset", "The research team", "A careful reviewer", "The compiler", "The garden"]
_EN_VERBS = ["describes", "measures", "connects", "records", "explains", "compares", "protects", "reveals"]
_EN_OBJECTS = ["the flow of information", "the cost of each answer", "the origin of every document",
               "a simple rule", "the weather in spring", "the history of the town", "an unexpected result",
               "the structure of the network"]
_FR_SUBJECTS = ["La rivière", "Un petit modèle", "La bibliothèque", "Chaque expérience", "Le vieux pont",
                "Ce jeu de données", "L'équipe de recherche", "Une relectrice attentive", "Le compilateur"]
_FR_VERBS = ["décrit", "mesure", "relie", "enregistre", "explique", "compare", "protège", "révèle"]
_FR_OBJECTS = ["le flux d'information", "le coût de chaque réponse", "l'origine de chaque document",
               "une règle simple", "le temps au printemps", "l'histoire de la ville", "un résultat inattendu"]


def _sentence(rng: random.Random, lang: str) -> str:
    if lang == "fr":
        s, v, o = rng.choice(_FR_SUBJECTS), rng.choice(_FR_VERBS), rng.choice(_FR_OBJECTS)
    else:
        s, v, o = rng.choice(_EN_SUBJECTS), rng.choice(_EN_VERBS), rng.choice(_EN_OBJECTS)
    n = rng.randint(2, 999)
    tail = rng.choice([f" in {n} steps", f" after {n} days", "", f" with {n} examples"]) if lang == "en" else \
        rng.choice([f" en {n} étapes", f" après {n} jours", "", f" avec {n} exemples"])
    return f"{s} {v} {o}{tail}."


def prose(n_docs: int, seed: int = 0, min_sentences: int = 12, max_sentences: int = 40) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    meta = {"generator": "prose", "version": GENERATOR_VERSION, "seed": seed}
    docs = []
    for i in range(n_docs):
        lang = rng.choice(["en", "fr"])
        paragraphs = []
        for _ in range(rng.randint(2, 4)):
            k = rng.randint(min_sentences // 3, max_sentences // 3)
            paragraphs.append(" ".join(_sentence(rng, lang) for _ in range(max(1, k))))
        docs.append({"id": f"prose-{i:07d}", "lang": lang, "text": "\n\n".join(paragraphs), "synthetic": meta})
    return docs


def arithmetic(n_tasks: int, seed: int = 0, max_operand: int = 999) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    meta = {"generator": "arithmetic", "version": GENERATOR_VERSION, "seed": seed, "max_operand": max_operand}
    tasks = []
    for i in range(n_tasks):
        a, b = rng.randint(0, max_operand), rng.randint(0, max_operand)
        op = rng.choice(["+", "-", "*"])
        answer = {"+": a + b, "-": a - b, "*": a * b}[op]
        tasks.append({
            "id": f"arith-{i:07d}",
            "prompt": f"Compute {a} {op} {b}.",
            "answer": str(answer),
            "text": f"Question: Compute {a} {op} {b}.\nAnswer: {answer}",
            "verifier": "exact_integer",
            "synthetic": meta,
        })
    return tasks


def verify_arithmetic(task: dict[str, Any], completion: str) -> bool:
    """True when the first integer in the completion equals the reference answer."""
    m = re.search(r"-?\d+", completion)
    return m is not None and m.group(0) == task["answer"]


def write_jsonl(records: list[dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
