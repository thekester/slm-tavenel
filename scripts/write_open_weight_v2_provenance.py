"""Create the hash-linked data provenance record for the privacy-risk-reduced pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline", type=Path, default=Path("data/processed/fsc-cc0-open-weights-v2/manifest.json"))
    parser.add_argument("--split", type=Path, default=Path("data/processed/fsc-cc0-open-weights-v2/splits/split_manifest.json"))
    parser.add_argument("--tokenizer", type=Path, default=Path("data/tokenizer/fsc-cc0-open-weights-v2/manifest.json"))
    parser.add_argument("--tokens", type=Path, default=Path("data/tokens/fsc-cc0-open-weights-v2/training/manifest.json"))
    parser.add_argument("--out", type=Path, default=Path("data/tokens/fsc-cc0-open-weights-v2/training/provenance.json"))
    args = parser.parse_args()
    pipeline = json.loads(args.pipeline.read_text(encoding="utf-8"))
    token_info = json.loads(args.tokens.read_text(encoding="utf-8"))
    split_info = json.loads(args.split.read_text(encoding="utf-8"))
    row = {
        "schema": "slm.open_weight_data_provenance.v1",
        "candidate_id": "slm-tavenel-1m-cc0-v2-pilot-s20261022",
        "source": {
            "dataset": "PleIAs/French-Science-Commons",
            "revision": "bfa114c09d7c28dcd0a8a47fe445c64b4a3c6156",
            "subset": "french_open_science",
            "record_license_policy": "CC0 label matched against every HAL API version; unresolved records excluded",
            "source_review_status": "candidate; maintainer source sign-off pending",
            "source_review_signed": False,
        },
        "privacy": {
            "metadata_screening": "excluded records tagged with human-subject-sensitive disciplines and/or title risk terms",
            "manual_review_status": "pending",
            "manual_review_signed": False,
            "pii_redactions": pipeline.get("stats", {}).get("pii_redactions", {}),
            "training_text_distributed": False,
        },
        "exclusions": pipeline.get("stats", {}).get("dropped", {}),
        "documents": {"input": pipeline.get("stats", {}).get("docs_in"),
                      "retained": pipeline.get("stats", {}).get("docs_out")},
        "splits": {"manifest_sha256": sha256(args.split), "salt": split_info["split_salt"],
                   "sealed_test_scored": False},
        "tokenizer": {"manifest_sha256": sha256(args.tokenizer),
                      "tokenizer_sha256": token_info["tokenizer_sha256"]},
        "tokens": {"manifest_sha256": sha256(args.tokens),
                    "train": token_info["splits"]["train"]["tokens"],
                    "dev": token_info["splits"]["val"]["tokens"],
                    "sealed_test": token_info["splits"]["test"]["tokens"]},
        "manifests": {"pipeline_sha256": sha256(args.pipeline)},
        "pipeline_config_sha256": pipeline.get("config_sha256"),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(row, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(args.out), "retained_documents": row["documents"]["retained"],
                      "tokens": row["tokens"], "source_review_signed": False,
                      "manual_review_signed": False}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
