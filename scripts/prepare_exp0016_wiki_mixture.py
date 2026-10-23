"""Prepare a fixed-token, benchmark-filtered wiki/FSC mixture for EXP-0016.

The wiki source remains a local comparison-only candidate. This script selects only the original
wiki-v1 train split, removes exact 13-gram overlap with all six reported benchmark files, retokenizes
with the existing 25M tokenizer, and keeps the existing wiki BPB holdout out of training.
"""

from __future__ import annotations

import hashlib
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from slm.data.contamination import ContaminationIndex  # noqa: E402
from slm.data.shards import split_of  # noqa: E402
from slm.runlog import code_revision, environment_info, sha256_file, write_json  # noqa: E402


SOURCE_DOCS = ROOT / "data/processed/wiki-v1/documents.jsonl"
SOURCE_MANIFEST = ROOT / "data/processed/wiki-v1/manifest.json"
BPB_HOLDOUT = ROOT / "data/processed/wiki-v1"
TOKENIZER_PATH = ROOT / "data/tokenizer/fsc-pilot-v1-train-only/tokenizer.json"
FSC_TOKENS = ROOT / "data/tokens/fsc-pilot-v1-scaling"
OUT = ROOT / "data/processed/exp0016-wiki"
BENCHMARKS = [
    ("arc_easy", ["context", "choices"]),
    ("arc_challenge", ["context", "choices"]),
    ("hellaswag", ["context", "choices"]),
    ("belebele_en", ["context", "choices"]),
    ("belebele_fr", ["context", "choices"]),
    ("winogrande", ["context", "choices"]),
]
SAMPLE_FRACTION = 0.30
DEFAULT_SELECTION_SALT = "slm-tavenel-exp0016-wiki-v1-sample-20261023"


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wiki-fraction", type=float, default=0.20,
                        help="fraction of total FSC+wiki training stream assigned to wiki")
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--selection-salt", default=DEFAULT_SELECTION_SALT)
    args = parser.parse_args()
    if not 0.0 < args.wiki_fraction < 1.0:
        raise ValueError("--wiki-fraction must be strictly between 0 and 1")
    out_dir = args.out if args.out.is_absolute() else ROOT / args.out
    selection_salt = args.selection_salt
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"Output must be empty to avoid overwriting a prior preparation: {out_dir}")
    for path in (SOURCE_DOCS, SOURCE_MANIFEST, TOKENIZER_PATH, FSC_TOKENS / "manifest.json"):
        if not path.is_file():
            raise FileNotFoundError(path)

    source = read_json(SOURCE_MANIFEST)
    bpb_manifest = read_json(BPB_HOLDOUT / "bpb-holdout-manifest.json")
    source_sha = sha256_file(SOURCE_DOCS)
    if source_sha != bpb_manifest["source_sha256"]:
        raise ValueError("wiki-v1 document hash does not match the pre-existing BPB holdout manifest")
    bpb_ids: dict[str, set[str]] = {"en": set(), "fr": set()}
    holdout_paths = {}
    for lang in ("en", "fr"):
        holdout_paths[lang] = BPB_HOLDOUT / f"bpb-val-{lang}.jsonl"
        expected = bpb_manifest["files"][lang]["sha256"]
        if sha256_file(holdout_paths[lang]) != expected:
            raise ValueError(f"wiki-v1 {lang} BPB holdout hash mismatch")
        with holdout_paths[lang].open(encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    bpb_ids[lang].add(str(json.loads(line)["id"]))

    index = ContaminationIndex(n=13, min_matches=1)
    benchmark_paths = []
    for name, fields in BENCHMARKS:
        path = ROOT / f"data/benchmarks/{name}.jsonl"
        index.add_jsonl(path, fields=fields)
        benchmark_paths.append({"path": path.relative_to(ROOT).as_posix(),
                                "sha256": sha256_file(path), "fields": fields})

    tok = Tokenizer.from_file(str(TOKENIZER_PATH))
    eos = tok.token_to_id("<|endoftext|>")
    if eos is None:
        raise ValueError("The 25M tokenizer has no <|endoftext|> token")
    dtype = np.uint16 if tok.get_vocab_size() <= 65535 else np.uint32
    dtype_name = "uint16" if dtype is np.uint16 else "uint32"
    fsc_manifest = read_json(FSC_TOKENS / "manifest.json")
    if fsc_manifest["tokenizer_sha256"] != sha256_file(TOKENIZER_PATH):
        raise ValueError("FSC and 25M tokenizer hashes differ")

    # Match the 80/20 data-stream ratio used in EXP-0014. English/French wiki tokens
    # are balanced so the treatment does not change language exposure as an extra factor.
    fsc_tokens = int(fsc_manifest["splits"]["train"]["tokens"])
    wiki_target_total = int(round(fsc_tokens * args.wiki_fraction / (1.0 - args.wiki_fraction)))
    targets = {"en": wiki_target_total // 2, "fr": wiki_target_total - wiki_target_total // 2}
    out_dir.mkdir(parents=True)
    paths = {lang: out_dir / f"train_wiki_{lang}.bin" for lang in ("en", "fr")}
    streams = {lang: path.open("wb") for lang, path in paths.items()}
    tokens_written = {"en": 0, "fr": 0}
    docs_written = {"en": 0, "fr": 0}
    contaminated: dict[str, int] = {}
    candidates = {"en": 0, "fr": 0}
    selected_ids_hash = {"en": hashlib.sha256(), "fr": hashlib.sha256()}
    threshold = int(SAMPLE_FRACTION * (1 << 64))

    def flush(language: str, rows: list[dict]) -> None:
        if not rows:
            return
        encodings = tok.encode_batch([row["text"] for row in rows])
        for row, enc in zip(rows, encodings):
            remaining = targets[language] - tokens_written[language]
            if remaining <= 0:
                break
            ids = enc.ids
            # End on EOS and keep every document boundary. The final document can be
            # truncated to the exact planned token budget; all other documents are whole.
            if len(ids) + 1 > remaining:
                if remaining < 2:
                    break
                ids = ids[: remaining - 1]
            array = np.asarray([*ids, int(eos)], dtype=dtype)
            array.tofile(streams[language])
            tokens_written[language] += int(array.size)
            docs_written[language] += 1
            selected_ids_hash[language].update((str(row["id"]) + "\n").encode("utf-8"))

    pending = {"en": [], "fr": []}
    with SOURCE_DOCS.open(encoding="utf-8") as source_stream:
        for line in source_stream:
            if not line.strip():
                continue
            row = json.loads(line)
            lang = "en" if str(row.get("source", "")).endswith("-en") else "fr"
            if tokens_written[lang] >= targets[lang]:
                continue
            doc_id = str(row["id"])
            if doc_id in bpb_ids[lang] or split_of(doc_id, 0.01, "slm-split-v1") != "train":
                continue
            digest = hashlib.sha256(f"{selection_salt}:{doc_id}".encode("utf-8")).digest()
            if int.from_bytes(digest[:8], "big") >= threshold:
                continue
            candidates[lang] += 1
            hits = index.matches(str(row["text"]))
            if hits:
                for name in hits:
                    benchmark = name.split(":", 1)[0]
                    contaminated[benchmark] = contaminated.get(benchmark, 0) + 1
                continue
            pending[lang].append(row)
            if len(pending[lang]) >= 256:
                flush(lang, pending[lang])
                pending[lang] = []
    for lang in ("en", "fr"):
        flush(lang, pending[lang])
        streams[lang].close()
        if tokens_written[lang] < int(targets[lang] * 0.98):
            raise ValueError(f"Insufficient decontaminated wiki tokens for {lang}: "
                             f"{tokens_written[lang]} / {targets[lang]}")

    # Keep a fresh, independently filtered text likelihood set for each language.
    holdouts = {}
    for lang in ("en", "fr"):
        source_path = holdout_paths[lang]
        output_path = out_dir / f"wiki_bpb_{lang}.jsonl"
        kept = 0
        removed: dict[str, int] = {}
        with source_path.open(encoding="utf-8") as src, output_path.open("w", encoding="utf-8") as dst:
            for line in src:
                row = json.loads(line)
                hits = index.matches(str(row["text"]))
                if hits:
                    for name in hits:
                        benchmark = name.split(":", 1)[0]
                        removed[benchmark] = removed.get(benchmark, 0) + 1
                    continue
                dst.write(json.dumps(row, ensure_ascii=False) + "\n")
                kept += 1
        holdouts[lang] = {"path": output_path.relative_to(ROOT).as_posix(),
                          "sha256": sha256_file(output_path), "documents_kept": kept,
                          "documents_removed_by_benchmark": removed,
                          "source_sha256": sha256_file(source_path)}
        if kept < 400:
            raise ValueError(f"Too few clean wiki BPB documents for {lang}: {kept}")

    manifest = {
        "kind": "slm.wiki_mixture_tokens",
        "source": {"path": SOURCE_DOCS.relative_to(ROOT).as_posix(),
                   "sha256": source_sha,
                   "manifest_sha256": sha256_file(SOURCE_MANIFEST),
                   "revision": "b04c8d1ceb2f5cd4588862100d08de323dccfbaa",
                   "declared_licenses": ["CC-BY-SA-3.0", "GFDL"],
                   "review_status": "candidate; rights and privacy review incomplete; local exploratory only"},
        "tokenizer": {"path": TOKENIZER_PATH.relative_to(ROOT).as_posix(),
                      "sha256": sha256_file(TOKENIZER_PATH), "vocab_size": tok.get_vocab_size(),
                      "eos_id": int(eos)},
        "dtype": dtype_name,
        "sampling": {"selection_salt": selection_salt, "hash_threshold_fraction": SAMPLE_FRACTION,
                     "original_wiki_train_split": "split_of(id, 0.01, slm-split-v1)",
                     "target_wiki_fraction": args.wiki_fraction,
                     "existing_bpb_holdout_ids_excluded": {k: len(v) for k, v in bpb_ids.items()},
                     "target_tokens_by_language": targets, "tokens_written_by_language": tokens_written,
                     "documents_written_by_language": docs_written, "candidate_docs_by_language": candidates,
                     "benchmark_overlaps_removed_by_benchmark": contaminated,
                     "benchmark_decontamination": "exact word 13-gram overlap; excludes matching train docs"},
        "benchmarks": benchmark_paths,
        "wiki_bpb_holdouts": holdouts,
        "training_shards": {lang: {"path": path.relative_to(ROOT).as_posix(),
                                     "tokens": tokens_written[lang], "sha256": sha256_file(path)}
                            for lang, path in paths.items()},
        "fsc_base_train_tokens": fsc_tokens,
        "actual_wiki_fraction_of_available_stream": sum(tokens_written.values()) /
            (fsc_tokens + sum(tokens_written.values())),
        "code": code_revision(), "environment": environment_info(),
    }
    write_json(out_dir / "manifest.json", manifest)
    print(json.dumps({"out": out_dir.relative_to(ROOT).as_posix(), "tokens_written": tokens_written,
                      "docs_written": docs_written,
                      "actual_wiki_fraction": manifest["actual_wiki_fraction_of_available_stream"],
                      "contaminated_training_docs": contaminated,
                      "holdouts": holdouts}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
