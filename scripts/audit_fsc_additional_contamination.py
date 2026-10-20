"""Audit exact word-13gram overlap between added tasks and the frozen FSC train split."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from slm.benchmarks import load_items
from slm.data.contamination import ContaminationIndex
from slm.runlog import sha256_file, write_json


BENCHMARKS = ["data/benchmarks/arc_challenge.jsonl", "data/benchmarks/winogrande.jsonl"]
TRAIN = ["data/processed/fsc-pilot-v1-token-balanced/en-train.jsonl",
         "data/processed/fsc-pilot-v1-token-balanced/fr-train.jsonl"]


def main() -> int:
    index = ContaminationIndex(n=13, min_matches=1)
    item_counts = {}
    for path in BENCHMARKS:
        items = load_items(path)
        item_counts[Path(path).stem] = len(items)
        for item in items:
            text = item["context"] + "\n" + "\n".join(item["choices"])
            index.add(f"{Path(path).stem}:{item['id']}", text)

    matched_docs: dict[str, set[str]] = defaultdict(set)
    matched_items: dict[str, set[str]] = defaultdict(set)
    matched_ngrams: dict[str, int] = defaultdict(int)
    documents_scanned = 0
    for path in TRAIN:
        with open(path, encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                documents_scanned += 1
                hits = index.matches(str(row.get("text", "")))
                if hits:
                    doc_id = f"{Path(path).stem}:{row.get('id', documents_scanned)}"
                    for item_id, count in hits.items():
                        benchmark, _, _ = item_id.partition(":")
                        matched_docs[benchmark].add(doc_id)
                        matched_items[benchmark].add(item_id)
                        matched_ngrams[benchmark] += count

    report = {
        "kind": "slm.exact-benchmark-contamination-audit",
        "documents_scanned": documents_scanned,
        "method": "exact lowercased Unicode word 13-gram overlap between each benchmark context+choices and FSC train documents; >=1 shared n-gram is a hit",
        "scope": "fsc-pilot-v1-token-balanced train split only; does not test validation text, paraphrases, translations, or external-model corpora",
        "benchmarks": [{"path": path, "sha256": sha256_file(path), "items": item_counts[Path(path).stem],
                        "matched_train_documents": len(matched_docs[Path(path).stem]),
                        "matched_evaluation_items": len(matched_items[Path(path).stem]),
                        "shared_13gram_counts": matched_ngrams[Path(path).stem]}
                       for path in BENCHMARKS],
        "train_files": [{"path": path, "sha256": sha256_file(path)} for path in TRAIN],
    }
    out = Path("runs/fsc-additional/contamination-audit.json")
    write_json(out, report)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
