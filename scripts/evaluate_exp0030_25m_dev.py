"""Score fixed FSC and HPLT development documents for EXP-0030."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.evaluate_exp0023_50m import score_documents_batched  # noqa: E402
from slm.runlog import sha256_file, write_json  # noqa: E402
from slm.tokenizer import load as load_tokenizer  # noqa: E402
from slm.train import load_model  # noqa: E402


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def repo_relative(path: Path) -> str:
    """Return a repository-relative path for either relative or absolute CLI input."""
    resolved = path if path.is_absolute() else ROOT / path
    return resolved.resolve().relative_to(ROOT.resolve()).as_posix()


def read_jsonl(path: Path, language: str) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row.get("text"):
                rows.append({"id": row.get("source_doc_id", row.get("id")),
                             "text": row["text"], "language": language})
    return rows


def read_hplt_split(path: Path, expected_split: str) -> list[dict]:
    rows = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row.get("split") != expected_split:
                raise ValueError(f"Non-{expected_split} HPLT record found in the evaluation file")
            if row.get("text") and row.get("language") in ("en", "fr"):
                rows.append({"id": row.get("id"), "text": row["text"],
                             "language": row["language"],
                             "source_dataset": row.get("source_dataset", "unknown")})
    return rows


def stable_dev_sample(rows: list[dict], max_per_language: int = 1500) -> list[dict]:
    selected = []
    for lang in ("en", "fr"):
        group = [r for r in rows if r["language"] == lang]
        group.sort(key=lambda r: hashlib.sha256(str(r["id"]).encode("utf-8")).hexdigest())
        selected.extend(group[:max_per_language])
    return selected


def score_group(model, tok, rows: list[dict], device) -> dict:
    scored = score_documents_batched(model, tok, [r["text"] for r in rows], device, batch_size=2)
    per_doc = []
    groups: dict[str, list[dict]] = {"en": [], "fr": []}
    for metadata, result in zip(rows, scored):
        doc = {
            "id": metadata.get("id"),
            "text_sha256": result["text_sha256"],
            "language": metadata["language"],
            "source_dataset": metadata.get("source_dataset"),
            "nll": result["nll"],
            "bytes": result["bytes"],
            "tokens": result["tokens"],
        }
        doc["bits_per_byte"] = doc["nll"] / (doc["bytes"] * math.log(2))
        per_doc.append(doc)
        groups[doc["language"]].append(doc)
    summary = {}
    for lang, docs in groups.items():
        if not docs:
            continue
        summary[lang] = {
            "documents": len(docs),
            "tokens": sum(r["tokens"] for r in docs),
            "bytes": sum(r["bytes"] for r in docs),
            "bits_per_byte": sum(r["nll"] for r in docs) /
                             (sum(r["bytes"] for r in docs) * math.log(2)),
        }
    return {"summary_by_language": summary, "per_document": per_doc}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, default=ROOT / "data/tokenizer/fsc-pilot-v1-train-only/tokenizer.json")
    parser.add_argument("--fsc-dir", type=Path, default=ROOT / "data/processed/fsc-pilot-v1-token-balanced")
    parser.add_argument("--hplt-path", "--hplt-dev", dest="hplt_path", type=Path,
                        default=ROOT / "data/processed/slm-tavenel-growth-candidate-v2/dev.jsonl.gz")
    parser.add_argument("--hplt-split", choices=("dev", "test"), default="dev")
    parser.add_argument("--hplt-only", action="store_true",
                        help="score only the selected HPLT split; useful for the sealed test evaluation")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    checkpoint = args.checkpoint.resolve()
    model = load_model(checkpoint, args.device).eval()
    tok = load_tokenizer(args.tokenizer)
    fsc_paths = {"en": args.fsc_dir / "en-val.jsonl", "fr": args.fsc_dir / "fr-val.jsonl"}
    fsc_rows = [] if args.hplt_only else [row for lang, path in fsc_paths.items()
                                          for row in read_jsonl(path, lang)]
    hplt_all_rows = read_hplt_split(args.hplt_path, args.hplt_split)
    if args.hplt_split == "dev":
        hplt_rows = stable_dev_sample(hplt_all_rows)
        hplt_sampling = "lowest SHA-256(document id) within language, up to 1,500 per language"
    else:
        hplt_rows = hplt_all_rows
        hplt_sampling = "all pinned test documents; no subsampling"
    device = torch.device(args.device)

    hplt_result_name = ("hplt_candidate_v2_dev_stable_sample" if args.hplt_split == "dev"
                        else "hplt_candidate_v2_test_all_documents")
    results = {hplt_result_name: score_group(model, tok, hplt_rows, device)}
    if not args.hplt_only:
        results = {"fsc_v1_validation": score_group(model, tok, fsc_rows, device), **results}
    output = {
        "kind": "slm.exp0030_25m_dev_likelihood",
        "checkpoint": str(checkpoint.relative_to(ROOT)),
        "checkpoint_sha256": sha256_file(checkpoint),
        "parameters": model.num_params(),
        "tokenizer": repo_relative(args.tokenizer),
        "tokenizer_sha256": sha256_file(args.tokenizer),
        "fsc_inputs": ({} if args.hplt_only else {
            lang: {"path": str(path.relative_to(ROOT)), "sha256": file_sha(path)}
            for lang, path in fsc_paths.items()
        }),
        f"hplt_{args.hplt_split}_input": {"path": repo_relative(args.hplt_path),
                                           "sha256": file_sha(args.hplt_path),
                                           "sampling": hplt_sampling},
        "evaluator": "scripts/evaluate_exp0030_25m_dev.py; document-level NLL / UTF-8 bytes",
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.out, output)
    for dataset, payload in results.items():
        print(json.dumps({"dataset": dataset, "summary_by_language": payload["summary_by_language"]}), flush=True)
    print(f"Wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
