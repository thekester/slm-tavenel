"""Summarize the exact overlap decisions recorded by the wiki-v1 pipeline."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from slm.runlog import sha256_file, write_json

ROOT = Path("data/processed/wiki-v1")


def count_lines(path: str) -> int:
    with open(path, encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def main() -> None:
    manifest = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
    by_benchmark_docs: dict[str, set[str]] = defaultdict(set)
    by_benchmark_items: dict[str, set[str]] = defaultdict(set)
    by_source: dict[str, set[str]] = defaultdict(set)
    detail_truncated_candidates = []
    with (ROOT / "decisions.jsonl").open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row.get("step") != "decontaminate":
                continue
            source = row["id"].split("/", 1)[0]
            by_source[source].add(row["id"])
            items = row.get("detail", {}).get("items", {})
            if len(items) == 5:
                detail_truncated_candidates.append(row["id"])
            for item_id in items:
                benchmark, _, example_id = item_id.partition(":")
                by_benchmark_docs[benchmark].add(row["id"])
                by_benchmark_items[benchmark].add(example_id)

    benchmarks = {}
    for spec in manifest["eval_files"]:
        name = Path(spec["path"]).stem
        benchmarks[name] = {
            "evaluation_items": count_lines(spec["path"]),
            "removed_training_documents": len(by_benchmark_docs[name]),
            "matched_evaluation_items_from_logged_details": len(by_benchmark_items[name]),
        }
    report = {
        "kind": "slm.decontamination-audit",
        "method": "exact word 13-gram overlap against context and choices; training-side document removal",
        "pipeline_manifest_sha256": sha256_file(ROOT / "manifest.json"),
        "decisions_sha256": sha256_file(ROOT / "decisions.jsonl"),
        "filtered_documents": sum(manifest["stats"]["per_source"][s]["out"] for s in manifest["stats"]["per_source"]),
        "removed_training_documents": sum(manifest["stats"]["dropped"]["decontaminate"].values()),
        "removed_documents_by_source": {k: len(v) for k, v in sorted(by_source.items())},
        "benchmarks": benchmarks,
        "details_at_five_item_cap": detail_truncated_candidates,
        "interpretation": "A zero or small n-gram count does not rule out paraphrase, translation, or semantic exposure.",
    }
    out = ROOT / "decontamination-audit.json"
    write_json(out, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
