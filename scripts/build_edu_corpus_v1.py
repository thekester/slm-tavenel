#!/usr/bin/env python3
"""Build the EN/FR educational-web corpus candidate for the classic 362M from-scratch run (EXP-0034).

Sources (pinned revisions, ODC-By 1.0 as declared by their dataset cards):
- English: HuggingFaceFW/fineweb-edu, `sample/10BT` shards (educational-quality classifier score >= 3).
- French: epfml/FineWeb2-HQ, `fra_Latn` shards (model-filtered high-quality subset of FineWeb-2).

Processing: length and character filters, PII redaction with ``slm.data.pii`` placeholders, exact
normalized-text dedup, exact 13-token overlap removal against the six pinned benchmark files, and a
registrable-domain hash split (98% train / 1% dev / 1% test) so held-out pages never share a site with
training pages. Every decision is counted; the manifest records shard hashes, licences and limitations.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pyarrow.parquet as pq
import tldextract
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
from build_hplt_growth_candidate import benchmark_ngrams, hash_file, overlaps_benchmark, site_key, write_jsonl_gz  # noqa: E402
from slm.data.dedup import normalized_hash  # noqa: E402
from slm.data.pii import redact  # noqa: E402

SOURCES = {
    "en": {
        "dataset": "HuggingFaceFW/fineweb-edu",
        "revision": "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9",
        "files": ["sample/10BT/000_00000.parquet", "sample/10BT/001_00000.parquet"],
        "declared_license": "ODC-By 1.0 (dataset card); also subject to Common Crawl terms of use",
        "selection": "FineWeb-Edu: FineWeb pages with Llama-3-70B-annotated educational classifier score >= 3",
    },
    "fr": {
        "dataset": "epfml/FineWeb2-HQ",
        "revision": "c0c06e94fd3a44ae9e802b2b0fc533817601eb5e",
        "files": ["fra_Latn/000_00000.parquet"],
        "declared_license": "ODC-By 1.0 (dataset card), inherited from FineWeb-2; Common Crawl terms of use",
        "selection": "FineWeb2-HQ: top-quality FineWeb-2 French pages selected by an embedding-based classifier",
    },
}
RAW = ROOT / "data/raw/edu-corpus-v1-2026-10-29"
OUT = ROOT / "data/processed/slm-tavenel-edu-corpus-v1"
MIN_CHARS = 200
MAX_CHARS = 100_000
MIN_ALNUM_RATIO = 0.6
SPLIT_SALT = "slm-tavenel-edu-domain-split-v1:"

_GRAMS: set = set()
_EXTRACTOR = None


def _init_worker(grams: set) -> None:
    global _GRAMS, _EXTRACTOR
    _GRAMS = grams
    _EXTRACTOR = tldextract.TLDExtract(suffix_list_urls=())


def split_for_site(site: str) -> str:
    value = int(hashlib.sha256((SPLIT_SALT + site).encode()).hexdigest()[:16], 16) / 2**64
    return "test" if value >= 0.99 else "dev" if value >= 0.98 else "train"


def process_batch(language: str, source: str, rows: list[dict]) -> tuple[list[tuple[str, str, dict]], Counter]:
    """Filter one batch; returns (split, dedup_key, record) triples and decision counts."""
    kept, counts = [], Counter()
    for row in rows:
        text = (row.get("text") or "").strip()
        counts["input"] += 1
        if not MIN_CHARS <= len(text) <= MAX_CHARS:
            counts["excluded_length"] += 1
            continue
        alnum = sum(ch.isalnum() or ch.isspace() for ch in text) / len(text)
        if alnum < MIN_ALNUM_RATIO:
            counts["excluded_low_alphanumeric_ratio"] += 1
            continue
        if "�" in text:
            counts["excluded_replacement_character"] += 1
            continue
        if overlaps_benchmark(text, _GRAMS):
            counts["excluded_benchmark_13gram_overlap"] += 1
            continue
        text, found = redact(text)
        for kind, n in found.items():
            counts[f"redacted_{kind}"] += n
        if any(found.values()):
            counts["documents_with_redactions"] += 1
        url = row.get("url") or ""
        site = site_key(url, _EXTRACTOR) if url else f"no-url:{row.get('id', '')}"
        split = split_for_site(site)
        record = {"id": str(row.get("id", "")), "text": text, "language": language, "source_dataset": source,
                  "url": url, "split": split, "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()}
        kept.append((split, normalized_hash(text), record))
    return kept, counts


def iter_batches(path: Path, batch_rows: int = 2000):
    columns = [c for c in ("text", "id", "url") if c in pq.read_schema(path).names]
    for batch in pq.ParquetFile(path).iter_batches(batch_size=batch_rows, columns=columns):
        yield batch.to_pylist()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--fr-files", type=int, default=1,
                        help="number of FineWeb2-HQ fra_Latn shards (v1 used 1; v2 uses 5 for ~25%% French)")
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--schema-version", default="v1")
    args = parser.parse_args()
    out_dir = args.out.resolve()
    SOURCES["fr"]["files"] = [f"fra_Latn/000_{i:05d}.parquet" for i in range(args.fr_files)]
    shards = []
    for language, src in SOURCES.items():
        card = hf_hub_download(repo_id=src["dataset"], filename="README.md", repo_type="dataset",
                               revision=src["revision"], local_dir=str(RAW / language))
        src["dataset_card_sha256"] = hash_file(Path(card))
        for name in src["files"]:
            path = Path(hf_hub_download(repo_id=src["dataset"], filename=name, repo_type="dataset",
                                        revision=src["revision"], local_dir=str(RAW / language)))
            shards.append((language, src["dataset"], path))
            src.setdefault("file_sha256", {})[name] = hash_file(path)
        print(f"downloaded {language}: {src['files']}", flush=True)

    grams, benchmark_inputs = benchmark_ngrams()
    out_dir.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    totals: dict[str, Counter] = {lang: Counter() for lang in SOURCES}
    buffers: dict[str, list[dict]] = {"train": [], "dev": [], "test": []}
    per_split = Counter()

    def kept_rows():
        with ProcessPoolExecutor(args.workers, initializer=_init_worker, initargs=(grams,)) as pool:
            for language, source, path in shards:
                futures = []
                for rows in iter_batches(path):
                    futures.append(pool.submit(process_batch, language, source, rows))
                    if len(futures) >= args.workers * 4:
                        yield language, futures.pop(0).result()
                for future in futures:
                    yield language, future.result()
                print(f"processed {path.name}: {dict(totals[language])}", flush=True)

    writers = {split: (out_dir /f"{split}.jsonl").open("w", encoding="utf-8") for split in buffers}
    try:
        for language, (kept, counts) in kept_rows():
            totals[language].update(counts)
            for split, key, record in kept:
                if key in seen:
                    totals[language]["excluded_exact_duplicate"] += 1
                    continue
                seen.add(key)
                totals[language][f"retained_{split}"] += 1
                per_split[f"{split}_{language}"] += 1
                writers[split].write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    finally:
        for stream in writers.values():
            stream.close()

    files = {}
    for split in buffers:
        plain = out_dir / f"{split}.jsonl"

        def rows(p=plain):
            with p.open("r", encoding="utf-8") as stream:
                for line in stream:
                    yield json.loads(line)
        files[split] = write_jsonl_gz(out_dir /f"{split}.jsonl.gz", rows())
        plain.unlink()

    manifest = {
        "schema": f"slm-tavenel.edu_corpus_candidate.{args.schema_version}",
        "status": "candidate_not_legal_clearance",
        "created": time.strftime("%Y-%m-%d"),
        "purpose": "EXP-0034 classic 362M from-scratch pretraining (EN/FR educational web)",
        "builder_arguments": {"fr_files": args.fr_files, "out": out_dir.relative_to(ROOT).as_posix()},
        "sources": SOURCES,
        "filters": {
            "characters": [MIN_CHARS, MAX_CHARS],
            "minimum_alphanumeric_or_space_ratio": MIN_ALNUM_RATIO,
            "pii": "slm.data.pii.redact placeholders (email, phone, public IPv4, IBAN); documents kept",
            "deduplication": "exact normalized-text SHA-256 across all selected shards; upstream MinHash dedup per dataset card",
            "benchmark_decontamination": {"exact_normalized_13_token_overlap": True, "inputs": benchmark_inputs},
            "split": "registrable-domain hash: 98% train, 1% dev, 1% held-out test",
        },
        "counts": {lang: dict(c) for lang, c in totals.items()},
        "records_by_split_language": dict(per_split),
        "files": files,
        "limitations": [
            "Web-crawl text; source-page rights are not verified record by record. ODC-By applies to the dataset compilation.",
            "Educational/quality classifiers were trained with LLM annotations (FineWeb-Edu: Llama-3-70B); their biases carry over.",
            "Regex PII redaction is not a complete privacy review.",
            "Exact 13-gram decontamination does not detect paraphrased or translated benchmark items.",
            "Candidate corpus for exploratory training; not a legal clearance or release authorization.",
        ],
    }
    (out_dir /"manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"counts": manifest["counts"], "records": manifest["records_by_split_language"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
