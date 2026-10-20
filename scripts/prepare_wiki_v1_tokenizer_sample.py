"""Select a deterministic 120k-document tokenizer sample from wiki-v1's train split only."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from slm.data.shards import split_of
from slm.runlog import sha256_file, write_json

SOURCE = Path("data/processed/wiki-v1/documents.jsonl")
SAMPLE = Path("data/processed/wiki-v1/tokenizer-train-120k.jsonl")
MANIFEST = Path("data/processed/wiki-v1/tokenizer-train-sample-manifest.json")
COUNT = 120_000
SEED = "slm-tavenel-wiki-v1-tokenizer-2026-10"
VAL_FRACTION = 0.01


def main() -> None:
    ranked_ids = []
    with SOURCE.open(encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if split_of(rec["id"], VAL_FRACTION) == "train":
                rank = hashlib.sha256(f"{SEED}:{rec['id']}".encode()).digest()
                ranked_ids.append((rank, rec["id"]))
    if len(ranked_ids) < COUNT:
        raise RuntimeError(f"only {len(ranked_ids)} training documents available")
    selected_ids = {doc_id for _, doc_id in sorted(ranked_ids)[:COUNT]}
    selected_order, per_language = [], {"en": 0, "fr": 0}
    with SOURCE.open(encoding="utf-8") as src, SAMPLE.open("w", encoding="utf-8", newline="\n") as out:
        for line in src:
            rec = json.loads(line)
            if rec["id"] not in selected_ids:
                continue
            language = "en" if rec["source"] == "wikipedia-en" else "fr"
            per_language[language] += 1
            selected_order.append(rec["id"])
            out.write(json.dumps({"id": rec["id"], "text": rec["text"]}, ensure_ascii=False) + "\n")
    write_json(MANIFEST, {
        "kind": "slm.tokenizer-training-sample",
        "source": str(SOURCE),
        "source_sha256": sha256_file(SOURCE),
        "selection": f"lowest SHA-256({SEED}:id) ranks among docs with split_of(id, {VAL_FRACTION}) == train",
        "seed": SEED,
        "split_val_fraction": VAL_FRACTION,
        "documents": len(selected_order),
        "documents_by_language": per_language,
        "selected_ids_sha256": hashlib.sha256("\n".join(selected_order).encode()).hexdigest(),
        "sample": str(SAMPLE),
        "sample_sha256": sha256_file(SAMPLE),
    })
    print(json.dumps({"documents": len(selected_order), "documents_by_language": per_language,
                      "sample_sha256": sha256_file(SAMPLE)}, indent=2))


if __name__ == "__main__":
    main()
