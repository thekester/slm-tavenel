"""Hash code, configs and data manifests used by the clean 10M replication."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from slm.runlog import code_revision, sha256_file, write_json

ROOT = Path(".")
OUT = Path("runs/wiki-v1-clean/source-snapshot.json")


def main() -> None:
    files = set((ROOT / "src/slm").rglob("*.py"))
    files.update((ROOT / "configs/model").rglob("*.yaml"))
    files.update((ROOT / "configs/train").glob("slm-10m-wiki*.yaml"))
    files.update(ROOT / path for path in (
        "scripts/compare_mc.py",
        "scripts/audit_wiki_v1_decontamination.py",
        "scripts/prepare_bpb_holdout.py",
        "scripts/prepare_wiki_v1_tokenizer_sample.py",
        "scripts/build_wiki_v1_clean.ps1",
        "scripts/start_wiki_v1_clean.ps1",
        "scripts/queue_wiki_v1_clean.ps1",
        "scripts/eval_wiki_v1_clean.ps1",
        "scripts/snapshot_clean_run_sources.py",
        "scripts/analyze_sinks.py",
        "experiments/EXP-0008-wiki-v1-clean-replication.md",
        "data/processed/wiki-v1/manifest.json",
        "data/processed/wiki-v1/stats.json",
        "data/processed/wiki-v1/bpb-holdout-manifest.json",
        "data/processed/wiki-v1/tokenizer-train-sample-manifest.json",
        "data/processed/wiki-v1/tokenizer-train-120k.jsonl",
        "data/processed/wiki-v1/decontamination-audit.json",
        "data/tokens/wiki-v1-clean/manifest.json",
        "data/tokenizer/wiki-v1-clean/manifest.json",
        "data/benchmarks/arc_easy.jsonl",
        "data/benchmarks/hellaswag.jsonl",
        "data/benchmarks/belebele_en.jsonl",
        "data/benchmarks/belebele_fr.jsonl",
    ))
    entries = [{"path": str(path).replace("\\", "/"), "sha256": sha256_file(path)}
               for path in sorted(files, key=lambda p: str(p).lower()) if path.is_file()]
    digest = hashlib.sha256(json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    pipeline = json.loads((ROOT / "data/processed/wiki-v1/manifest.json").read_text(encoding="utf-8"))
    code = code_revision(ROOT)
    write_json(OUT, {
        "kind": "slm.source-snapshot",
        "base_pipeline_code_commit": pipeline["code"]["commit"],
        "code_commit": code["commit"],
        "working_tree_dirty": code["dirty"],
        "files_sha256": entries,
        "snapshot_sha256": digest,
    })
    print(digest)


if __name__ == "__main__":
    main()
