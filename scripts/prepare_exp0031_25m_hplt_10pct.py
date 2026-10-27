"""Build a deterministic 90% FSC / 10% HPLT-rich candidate stream for EXP-0031.

The HPLT-rich input remains a local research candidate; this script does not grant or
infer rights to the source pages.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from slm.runlog import code_revision, environment_info, sha256_file, write_json  # noqa: E402

FSC = ROOT / "data/tokens/fsc-pilot-v1-scaling"
CORPUS = ROOT / "data/processed/slm-tavenel-growth-candidate-v2"
TOKENIZER = ROOT / "data/tokenizer/fsc-pilot-v1-train-only/tokenizer.json"
TOKENIZER_MANIFEST = ROOT / "data/tokenizer/fsc-pilot-v1-train-only/manifest.json"
OUT = ROOT / "data/tokens/exp0031-25m-hplt-10pct-fsc-tokenizer"
SALT = "slm-tavenel-exp0031-hplt-candidate-v2-10pct-v1"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def iter_train(path: Path):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                row = json.loads(line)
                if row.get("split") == "train" and row.get("language") in ("en", "fr") and row.get("text"):
                    yield row


def fsc_validation_keys() -> tuple[set[str], set[str], dict[str, str]]:
    """Collect exact text hashes and stable source IDs for the held-out FSC documents."""
    text_hashes: set[str] = set()
    source_ids: set[str] = set()
    files = {}
    for lang in ("en", "fr"):
        path = ROOT / f"data/processed/fsc-pilot-v1-token-balanced/{lang}-val.jsonl"
        files[lang] = sha256_file(path)
        with path.open("r", encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                text_hashes.add(hashlib.sha256(str(row["text"]).encode("utf-8")).hexdigest())
                for key in ("id", "source_doc_id"):
                    if row.get(key) is not None:
                        source_ids.add(str(row[key]))
    return text_hashes, source_ids, files


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--fraction", type=float, default=0.10)
    args = parser.parse_args()
    out = args.out if args.out.is_absolute() else ROOT / args.out
    if not 0 < args.fraction < 0.5:
        raise ValueError("fraction must be between 0 and 0.5")
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Output must be empty: {out}")

    corpus_manifest_path = CORPUS / "manifest.json"
    corpus_manifest = read_json(corpus_manifest_path)
    if corpus_manifest.get("status") != "candidate_not_legal_clearance":
        raise ValueError("Unexpected corpus review status")
    corpus_path = CORPUS / "train.jsonl.gz"
    if sha256_file(corpus_path) != corpus_manifest["files"]["train"]["sha256"]:
        raise ValueError("HPLT-rich candidate train file hash mismatch")
    fsc_manifest_path = FSC / "manifest.json"
    fsc = read_json(fsc_manifest_path)
    tokenizer_manifest = read_json(TOKENIZER_MANIFEST)
    tokenizer_sha = sha256_file(TOKENIZER)
    if tokenizer_sha != tokenizer_manifest["tokenizer_sha256"] or tokenizer_sha != fsc["tokenizer_sha256"]:
        raise ValueError("FSC and EXP-0024 tokenizer lineage must match exactly")
    tok = Tokenizer.from_file(str(TOKENIZER))
    eos = tok.token_to_id("<|endoftext|>")
    if eos is None:
        raise ValueError("Expected <|endoftext|> token")
    dtype = np.uint16 if tok.get_vocab_size() <= 65535 else np.uint32
    dtype_name = "uint16" if dtype is np.uint16 else "uint32"

    fsc_tokens = int(fsc["splits"]["train"]["tokens"])
    hplt_target = int(round(fsc_tokens * args.fraction / (1 - args.fraction)))
    targets = {"en": hplt_target // 2, "fr": hplt_target - hplt_target // 2}
    fsc_val_text_hashes, fsc_val_source_ids, fsc_val_hashes = fsc_validation_keys()
    excluded = {"fsc_source_records": 0, "exact_fsc_validation_text": 0,
                "fsc_validation_source_id": 0}

    def keep_candidate(row: dict) -> bool:
        source_dataset = str(row.get("source_dataset", "")).casefold()
        if "french-science-commons" in source_dataset or source_dataset in {"fsc", "pleias/fsc"}:
            excluded["fsc_source_records"] += 1
            return False
        text_hash = hashlib.sha256(str(row["text"]).encode("utf-8")).hexdigest()
        if text_hash in fsc_val_text_hashes:
            excluded["exact_fsc_validation_text"] += 1
            return False
        source_ids = {str(row.get(k)) for k in ("id", "source_id", "source_doc_id") if row.get(k) is not None}
        if source_ids & fsc_val_source_ids:
            excluded["fsc_validation_source_id"] += 1
            return False
        return True

    # Rank documents by a salted stable hash, so file order does not determine membership.
    ranked: dict[str, list[tuple[str, str, int]]] = {"en": [], "fr": []}
    pending: list[dict] = []
    def record_lengths(rows: list[dict]) -> None:
        if not rows:
            return
        encodings = tok.encode_batch([r["text"] for r in rows])
        for row, enc in zip(rows, encodings):
            lang = row["language"]
            rank = hashlib.sha256(f"{SALT}|{row['id']}".encode("utf-8")).hexdigest()
            ranked[lang].append((rank, str(row["id"]), len(enc.ids) + 1))
    for row in iter_train(corpus_path):
        if not keep_candidate(row):
            continue
        pending.append(row)
        if len(pending) >= 256:
            record_lengths(pending)
            pending.clear()
    record_lengths(pending)

    selected: dict[str, dict[str, int]] = {"en": {}, "fr": {}}
    selected_hashes = {}
    for lang in ("en", "fr"):
        used = 0
        h = hashlib.sha256()
        for rank, doc_id, token_count in sorted(ranked[lang]):
            if used >= targets[lang]:
                break
            take = min(token_count, targets[lang] - used)
            if take < 2:
                break
            selected[lang][doc_id] = take
            used += take
            h.update((doc_id + "\n").encode("utf-8"))
        if used < int(targets[lang] * 0.999):
            raise ValueError(f"Insufficient HPLT-rich tokens for {lang}: {used}/{targets[lang]}")
        selected_hashes[lang] = h.hexdigest()

    out.mkdir(parents=True, exist_ok=True)
    hplt_shards = []
    token_written = {"en": 0, "fr": 0}
    docs_written = {"en": 0, "fr": 0}
    paths = {lang: out / f"train_hplt_{lang}.bin" for lang in ("en", "fr")}
    streams = {lang: paths[lang].open("wb") for lang in ("en", "fr")}
    pending = []
    def write_selected(rows: list[dict]) -> None:
        if not rows:
            return
        encodings = tok.encode_batch([r["text"] for r in rows])
        for row, enc in zip(rows, encodings):
            lang = row["language"]
            take = selected[lang].get(str(row["id"]))
            if take is None:
                continue
            ids = enc.ids[:max(0, take - 1)]
            np.asarray([*ids, int(eos)], dtype=dtype).tofile(streams[lang])
            token_written[lang] += len(ids) + 1
            docs_written[lang] += 1
    for row in iter_train(corpus_path):
        if str(row["id"]) in selected[row["language"]]:
            pending.append(row)
        if len(pending) >= 256:
            write_selected(pending)
            pending.clear()
    write_selected(pending)
    for stream in streams.values():
        stream.close()

    for lang in ("en", "fr"):
        hplt_shards.append({"file": paths[lang].name, "tokens": token_written[lang],
                            "sha256": sha256_file(paths[lang]), "component": f"hplt-candidate-v2-{lang}"})

    def copy_shards(split: str, prefix: str, component: str) -> list[dict]:
        result = []
        for i, shard in enumerate(fsc["splits"][split]["shards"]):
            src = FSC / shard["file"]
            if sha256_file(src) != shard["sha256"]:
                raise ValueError(f"FSC shard hash mismatch: {src}")
            name = f"{prefix}_{i:05d}.bin"
            dst = out / name
            shutil.copyfile(src, dst)
            result.append({"file": name, "tokens": int(shard["tokens"]), "sha256": sha256_file(dst),
                           "component": component})
        return result

    fsc_train_shards = copy_shards("train", "train_fsc", "fsc-pilot-v1-balanced-science")
    fsc_val_shards = copy_shards("val", "val_fsc", "fsc-pilot-v1-heldout")
    train_shards = fsc_train_shards + hplt_shards
    total_train = sum(x["tokens"] for x in train_shards)
    actual_fraction = sum(token_written.values()) / total_train
    manifest = {
        "kind": "slm.tokens", "status": "candidate_not_legal_clearance",
        "experiment": "EXP-0031", "created": "2026-10-27",
        "dtype": dtype_name, "eos_id": int(eos), "vocab_size": tok.get_vocab_size(),
        "tokenizer_sha256": tokenizer_sha,
        "tokenizer_manifest_sha256": sha256_file(TOKENIZER_MANIFEST),
        "mix": {
            "target_hplt_candidate_fraction": args.fraction,
            "actual_hplt_candidate_fraction": actual_fraction,
            "fsc_unique_train_tokens": fsc_tokens,
            "hplt_candidate_tokens_by_language": token_written,
            "hplt_target_tokens_by_language": targets,
            "hplt_documents_by_language": docs_written,
            "selected_document_id_sha256_by_language": selected_hashes,
            "selection": "lowest salted SHA-256(document id) ranks per language; one deterministic final document may be truncated and terminated with EOS",
            "selection_salt": SALT,
            "fsc_validation_decontamination": {
                "exact_text_hash_and_source_id_exclusion_counts": excluded,
                "validation_inputs": {lang: {"path": f"data/processed/fsc-pilot-v1-token-balanced/{lang}-val.jsonl",
                                             "sha256": fsc_val_hashes[lang]} for lang in ("en", "fr")},
                "rule": "exclude candidate source rows from French-Science-Commons and any exact text-hash or stable source-ID match to FSC validation documents",
            },
        },
        "inputs": [
            {"path": "data/tokens/fsc-pilot-v1-scaling/manifest.json", "sha256": sha256_file(fsc_manifest_path)},
            {"path": "data/processed/slm-tavenel-growth-candidate-v2/manifest.json", "sha256": sha256_file(corpus_manifest_path)},
            {"path": "data/processed/slm-tavenel-growth-candidate-v2/train.jsonl.gz", "sha256": sha256_file(corpus_path)},
        ],
        "splits": {
            "train": {"tokens": total_train, "shards": train_shards},
            "val": {"tokens": sum(x["tokens"] for x in fsc_val_shards),
                    "docs": fsc["splits"]["val"].get("docs"), "shards": fsc_val_shards},
        },
        "source_rights": "HPLT-derived candidate; upstream labels are recorded, not independently verified; page-level rights and privacy remain unresolved; local exploratory training only",
        "benchmarks": "six pinned benchmark files used only by the upstream candidate-v2 exact-overlap filter and as evaluation data; never training text",
        "code": code_revision(), "environment": environment_info(),
    }
    if abs(actual_fraction - args.fraction) > 0.001:
        raise ValueError(f"Actual mixture fraction differs from target: {actual_fraction}")
    write_json(out / "manifest.json", manifest)
    print(json.dumps({"out": str(out), "target_fraction": args.fraction,
                      "actual_fraction": actual_fraction, "hplt_tokens": token_written,
                      "hplt_docs": docs_written, "tokenizer_sha256": tokenizer_sha}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
