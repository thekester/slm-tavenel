"""Record the complete local provenance chain for the EXP-0009 token shards."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    data_dir = Path("data/tokens/fsc-pilot-v1-scaling")
    refs = {
        "source_registry": Path("configs/data/sources.yaml"),
        "metadata_audit": Path("data/raw/french-science-commons-pilot/document-metadata-audit-v2.json"),
        "initial_metadata_audit": Path("data/raw/french-science-commons-pilot/document-metadata-audit.json"),
        "pipeline_manifest": Path("data/processed/fsc-pilot-v1/manifest.json"),
        "whitespace_balance_manifest": Path("data/processed/fsc-pilot-v1-balanced/balance_manifest.json"),
        "document_split_manifest": Path("data/processed/fsc-pilot-v1-balanced/splits/split_manifest.json"),
        "token_balance_manifest": Path("data/processed/fsc-pilot-v1-token-balanced/final_token_balance_manifest.json"),
        "tokenizer_manifest": Path("data/tokenizer/fsc-pilot-v1-train-only/manifest.json"),
        "token_shard_manifest": data_dir / "manifest.json",
    }
    missing = [str(path) for path in refs.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing provenance inputs: {missing}")
    pipeline = json.loads(refs["pipeline_manifest"].read_text(encoding="utf-8"))
    audit = json.loads(refs["metadata_audit"].read_text(encoding="utf-8"))
    initial_audit = json.loads(refs["initial_metadata_audit"].read_text(encoding="utf-8"))
    token_balance = json.loads(refs["token_balance_manifest"].read_text(encoding="utf-8"))
    token_shards = json.loads(refs["token_shard_manifest"].read_text(encoding="utf-8"))
    manifest = {
        "kind": "slm.dataset_provenance",
        "artifacts": {name: {"path": str(path), "sha256": sha256_file(path)} for name, path in refs.items()},
        "name": "French-Science-Commons FSC pilot v1, token-balanced scaling sample",
        "dataset": "PleIAs/French-Science-Commons",
        "subset": "french_open_science",
        "revision": "bfa114c09d7c28dcd0a8a47fe445c64b4a3c6156",
        "status": "candidate; exploratory use only; not independently rights-cleared; no public weight release approval",
        "source_ids": ["fsc-en-cc0", "fsc-fr-cc0", "fsc-en-ccby4", "fsc-fr-ccby4", "fsc-en-lo2", "fsc-fr-lo2"],
        "license_classes": ["CC0-1.0", "CC-BY-4.0 (attribution required)", "Licence-Ouverte-2.0 (attribution required)"],
        "training_policy": {"allow_unapproved": pipeline.get("allow_unapproved"),
                            "source_registry_status": "all six entries remain candidate",
                            "public_weights_approved": False},
        "metadata_audit": {"path": str(refs["metadata_audit"]), "sha256": sha256_file(refs["metadata_audit"]),
                           "documents_scanned": audit["documents_scanned"], "documents_excluded": audit["documents_excluded"],
                           "excluded_ids_match_initial_audit": set(audit["excluded_document_ids"]) == set(initial_audit["excluded_document_ids"])},
        "processing": {"manifest": str(refs["pipeline_manifest"]), "sha256": sha256_file(refs["pipeline_manifest"]),
                       "documents_in": pipeline["stats"]["docs_in"], "documents_out": pipeline["stats"]["docs_out"],
                       "benchmark_filter_ngram": 13, "benchmark_documents_removed": pipeline["stats"]["dropped"].get("decontaminate", {}).get("benchmark_overlap", 0),
                       "repetition_filter": "omitted after academic-text false-positive review"},
        "token_balance": {"manifest": str(refs["token_balance_manifest"]), "sha256": sha256_file(refs["token_balance_manifest"]),
                          "language_counts": token_balance["language_token_balance"]},
        "tokenizer": {"manifest": str(refs["tokenizer_manifest"]), "sha256": sha256_file(refs["tokenizer_manifest"]),
                      "tokenizer_sha256": token_shards["tokenizer_sha256"],
                      "training_input_policy": "train-only inputs required by the FSC preparation protocol; not re-audited by this manifest writer"},
        "token_shards": {"manifest": str(refs["token_shard_manifest"]), "sha256": sha256_file(refs["token_shard_manifest"]),
                         "splits": {name: {"tokens": info["tokens"], "documents": info["docs"]}
                                    for name, info in token_shards["splits"].items()}},
        "review_gaps": ["verify each included per-record license against its originating publisher/repository",
                        "finish privacy review beyond pattern-based PII redaction",
                        "prepare attribution handling for CC-BY and Licence Ouverte before any public release"],
    }
    out = data_dir / "provenance.json"
    out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"path": str(out), "sha256": sha256_file(out), "status": manifest["status"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
