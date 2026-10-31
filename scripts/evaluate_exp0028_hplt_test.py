#!/usr/bin/env python3
"""Score the sealed EXP-0028 HPLT candidate test split by language and source."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from scripts.evaluate_exp0023_50m import score_documents_batched  # noqa: E402
from slm.runlog import assert_tokenizer_compatible, sha256_file, write_json  # noqa: E402
from slm.tokenizer import load as load_tokenizer  # noqa: E402
from slm.train import load_model  # noqa: E402


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, default=ROOT / "data/tokenizer/fsc-pilot-v1-train-only/tokenizer.json")
    parser.add_argument("--corpus", type=Path, default=ROOT / "data/processed/slm-tavenel-growth-candidate-v2")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--experiment-id", default="EXP-0028")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--document-batch", type=int, default=16,
                        help="documents held in CPU memory per scoring batch")
    parser.add_argument("--forward-batch", type=int, default=2,
                        help="sequences per model forward pass; keep small for 8 GiB GPUs")
    parser.add_argument("--every", type=int, default=1,
                        help="score every N-th test document (systematic subsample); 1 scores all")
    args = parser.parse_args()

    checkpoint = args.checkpoint.resolve()
    tokenizer_sha = sha256_file(args.tokenizer)
    tokenizer_lineage = assert_tokenizer_compatible(checkpoint, tokenizer_sha)
    test_path = args.corpus / "test.jsonl.gz"
    manifest_path = args.corpus / "manifest.json"
    corpus_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if corpus_manifest.get("status") != "candidate_not_legal_clearance":
        raise ValueError("Unexpected corpus review status")
    expected_test_sha = corpus_manifest["files"]["test"]["sha256"]
    if digest(test_path) != expected_test_sha:
        raise ValueError("Test split hash does not match pinned corpus manifest")

    device = torch.device(args.device)
    model = load_model(checkpoint, device).eval()
    tok = load_tokenizer(args.tokenizer)
    groups: dict[str, dict[str, float]] = defaultdict(lambda: {"nll": 0.0, "bytes": 0, "tokens": 0, "documents": 0})
    per_document = []
    pending = []
    seen = 0

    def score_pending() -> None:
        nonlocal seen
        if not pending:
            return
        texts = [row["text"] for row in pending]
        scores = score_documents_batched(model, tok, texts, device, batch_size=args.forward_batch)
        for metadata, scored in zip(pending, scores):
            if metadata.get("split") != "test":
                raise ValueError("Non-test document appeared in the sealed test file")
            bpb = scored["nll"] / (scored["bytes"] * math.log(2))
            record = {
                "id": metadata.get("id"),
                "text_sha256": metadata.get("text_sha256"),
                "language": metadata.get("language", "unknown"),
                "source_dataset": metadata.get("source_dataset", "unknown"),
                "tokens": scored["tokens"],
                "bytes": scored["bytes"],
                "nll": scored["nll"],
                "bits_per_byte": bpb,
            }
            per_document.append(record)
            keys = ["all", f"language:{record['language']}", f"source_dataset:{record['source_dataset']}",
                    f"language_source:{record['language']}|{record['source_dataset']}"]
            for key in keys:
                agg = groups[key]
                agg["nll"] += scored["nll"]
                agg["bytes"] += scored["bytes"]
                agg["tokens"] += scored["tokens"]
                agg["documents"] += 1
            seen += 1
        pending.clear()

    index = -1
    with gzip.open(test_path, "rt", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row.get("text"), str) or not row["text"]:
                continue
            index += 1
            if index % args.every:
                continue  # systematic subsample keeps the file's language/source proportions
            pending.append(row)
            if len(pending) >= args.document_batch:
                score_pending()
                if seen and seen % 512 == 0:
                    print(f"Scored {seen} held-out documents", flush=True)
    score_pending()

    summaries = {}
    for key, agg in groups.items():
        summaries[key] = {
            "documents": int(agg["documents"]),
            "tokens": int(agg["tokens"]),
            "bytes": int(agg["bytes"]),
            "nll": agg["nll"],
            "nll_per_token": agg["nll"] / max(1, agg["tokens"]),
            "bits_per_byte": agg["nll"] / (max(1, agg["bytes"]) * math.log(2)),
        }
    result = {
        "kind": "slm.exp0028_hplt_candidate_test_bpb",
        "experiment_id": args.experiment_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "tokenizer": str(args.tokenizer.resolve()),
        "tokenizer_sha256": sha256_file(args.tokenizer),
        "tokenizer_lineage": tokenizer_lineage,
        "device": str(device),
        "corpus_manifest_sha256": sha256_file(manifest_path),
        "test_path": str(test_path),
        "test_sha256": expected_test_sha,
        "evaluation_script_sha256": sha256_file(Path(__file__).resolve()),
        "documents_scored": seen,
        "systematic_subsample_every": args.every,
        "groups": summaries,
        "per_document": per_document,
        "limitations": [
            "single-seed candidate-corpus evaluation; source rights and human privacy review are unresolved",
            "candidate test split is domain-held-out but is not an external benchmark",
            "bits per byte is tokenizer-independent but remains conditional on this corpus and filter pipeline",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.out, result)
    print(f"Wrote {args.out}; scored {seen} documents", flush=True)


if __name__ == "__main__":
    main()
