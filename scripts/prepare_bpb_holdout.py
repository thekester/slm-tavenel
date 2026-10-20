"""Create a deterministic, language-stratified BPB set from wiki-v1's held-out docs."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

from slm.data.shards import split_of
from slm.runlog import sha256_file, write_json

SOURCE = Path("data/processed/wiki-v1/documents.jsonl")
OUT = Path("data/processed/wiki-v1")
SEED = "slm-tavenel-wiki-v1-bpb-2026-10"
DOCS_PER_LANGUAGE = 500


def main() -> None:
    by_language: dict[str, list[dict[str, str]]] = defaultdict(list)
    with SOURCE.open(encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if split_of(rec["id"], 0.01) != "val":
                continue
            language = "en" if rec["source"] == "wikipedia-en" else "fr" if rec["source"] == "wikipedia-fr" else None
            if language:
                by_language[language].append(rec)

    files = {}
    for language in ("en", "fr"):
        docs = sorted(by_language[language], key=lambda x: hashlib.sha256(f"{SEED}:{x['id']}".encode()).digest())
        if len(docs) < DOCS_PER_LANGUAGE:
            raise RuntimeError(f"only {len(docs)} validation documents for {language}; need {DOCS_PER_LANGUAGE}")
        selected = docs[:DOCS_PER_LANGUAGE]
        path = OUT / f"bpb-val-{language}.jsonl"
        with path.open("w", encoding="utf-8", newline="\n") as f:
            for rec in selected:
                f.write(json.dumps({"id": rec["id"], "text": rec["text"]}, ensure_ascii=False) + "\n")
        files[language] = {"path": str(path), "items": len(selected), "sha256": sha256_file(path),
                          "source_ids_sha256": hashlib.sha256("\n".join(x["id"] for x in selected).encode()).hexdigest()}

    write_json(OUT / "bpb-holdout-manifest.json", {
        "kind": "slm.bpb-holdout",
        "source": str(SOURCE),
        "source_sha256": sha256_file(SOURCE),
        "split": "val from split_of(id, 0.01), same rule as data/tokens/wiki-v1",
        "selection": f"first {DOCS_PER_LANGUAGE} ids by SHA-256({SEED}:id), independently per language",
        "seed": SEED,
        "files": files,
    })
    print(json.dumps(files, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
