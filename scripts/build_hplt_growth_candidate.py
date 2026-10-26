#!/usr/bin/env python3
"""Extend the local bilingual corpus candidate with a pinned HPLT growth slice.

Only HPLT's published TRAIN shards are consumed. Benchmark files are read only
to exclude exact 13-token overlaps; they are never copied into corpus outputs.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path
from urllib.parse import urlsplit

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer
import tldextract

ROOT = Path(__file__).resolve().parents[1]
DATASET = "ashtok897/european-hplt-v1"
REVISION = "51edb798468aa71b1000e950d3ab43bb435f72cd"
SHARD_SHA256 = {
    0: "657fc30a96b4450828ef3f96cdad28befacc4c5abbb3bcb6f3ee5791223a2556",
    1: "81850215c6fdc710e51788ef3fd50d57e5f1bc639f0856103bf166215c83280d",
    2: "cd8fff349850c616f59f50e63c9db148300dfe2c21fde24c2fa900dd87445278",
    3: "1a8833c0f4cc9a35894059ce10172813dcbcb4a6c6c95ff7325a71d02d360975",
    4: "b6b92f3d3b3021084a6b264305a037f348a31c29957d4023a0c80712264ae9f2",
    5: "92f6e44899319183fea60cef94e207a96777665a471bf661a4125d3d4541c322",
    6: "9b0d2be630496fbe1258c5b1e780ccb962c5417c1e2e4143c7af19448da34ed9",
    7: "f5d2ff53744aa13369f47aa434e7c0ec5bcde83e7df92e266dfdacdd34689ed6",
    8: "4bb2d71f1a2a149a2a071a009255baffbce8a19fbfaf12737284550efa57bf1c",
    9: "d71668b2ece44431e3722f867a22389b65db62f623781ec7f20556c5b8e9f26e",
}
RAW = ROOT / "data/raw/european-hplt-v1-2026-10-25"
PARENT = ROOT / "data/processed/slm-tavenel-reliable-corpus-candidate-v1"
OUT = ROOT / "data/processed/slm-tavenel-growth-candidate-v2"
TOKENIZER_PATH = ROOT / "exports/slm-tavenel-1m-exp0010-release-candidate-v4/tokenizer.json"
TOKENIZER_SHA = "93c7b9364de2c42315571c851353862e63830bfbeb220dc14dde80e6ca82cfcb"
MIN_LANG_PROB = 0.90
MIN_CHARS = 300
MAX_CHARS = 100_000
MIN_ALNUM_RATIO = 0.48
MAX_REG_CHARS = frozenset({"MT", "LY"})
REGISTER_CLASSES = ("MT", "LY", "SP", "ID", "NA", "HI", "IN", "OP", "IP")
MIME_ALLOWLIST = frozenset({
    "text/html", "text/plain", "application/xml", "application/xhtml+xml",
    "application/atom+xml", "application/rss+xml", "application/epub+zip",
})
TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)
EMAIL_RE = re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b")
PHONE_RE = re.compile(r"(?<!\w)(?:\+?\d{1,3}[ .-]?)?(?:\(\d{2,4}\)[ .-]?|\d{3}[ .-])\d{3}[ .-]\d{4}(?!\w)")
IP_RE = re.compile(r"(?i)(?:\bIP(?:v4)? address\b|\bIPv4\b|\bhost address\b|\bserver address\b)\D{0,25}(?:\d{1,3}\.){3}\d{1,3}")
SSN_RE = re.compile(r"(?i)(?:\bSSN\b|\bSocial Security (?:number|No\.?)\b)\D{0,20}\d{3}-\d{2}-\d{4}")
SENSITIVE_RE = re.compile(r"(?i)\b(patient|participant|clinical trial|case report|human subject|medical record|dossier m[eé]dical|essai clinique|cas clinique|sujet humain|diagnosis|diagnostic|biometric|facial recognition|surveillance)\b")
BENCHMARKS = [
    "arc_easy.jsonl", "arc_challenge.jsonl", "hellaswag.jsonl",
    "belebele_en.jsonl", "belebele_fr.jsonl", "winogrande.jsonl",
]


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def benchmark_ngrams(n: int = 13) -> tuple[set[tuple[str, ...]], list[dict]]:
    grams: set[tuple[str, ...]] = set()
    manifest = []
    for name in BENCHMARKS:
        path = ROOT / "data/benchmarks" / name
        sidecar = path.with_suffix(".manifest.json")
        if not path.exists() or not sidecar.exists():
            raise RuntimeError(f"Pinned benchmark input or manifest missing: {name}")
        digest = hash_file(path)
        meta = json.loads(sidecar.read_text(encoding="utf-8"))
        if meta.get("sha256") != digest:
            raise RuntimeError(f"Benchmark file hash mismatch: {name}")
        manifest.append({"path": path.relative_to(ROOT).as_posix(), "sha256": digest,
                         "dataset": meta.get("repo"), "revision": meta.get("revision"),
                         "items": meta.get("items"), "role": "decontamination_only_not_training_data"})
        with path.open("r", encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                for chunk in [row.get("context", ""), *row.get("choices", [])]:
                    tokens = TOKEN_RE.findall(chunk.casefold())
                    grams.update(tuple(tokens[i:i+n]) for i in range(max(0, len(tokens)-n+1)))
    return grams, manifest


def overlaps_benchmark(text: str, grams: set[tuple[str, ...]], n: int = 13) -> bool:
    tokens = TOKEN_RE.findall(text.casefold())
    return any(tuple(tokens[i:i+n]) in grams for i in range(max(0, len(tokens)-n+1)))


def site_key(url: str, extractor: tldextract.TLDExtract) -> str:
    host = (urlsplit(url).hostname or "").casefold().strip(".")
    result = extractor(host) if host else None
    if result and result.domain and result.suffix:
        return f"{result.domain}.{result.suffix}"
    return host or "unknown-host"


def split_for_host(host: str) -> str:
    value = int(hashlib.sha256(("slm-tavenel-hplt-domain-split-v1:" + host).encode()).hexdigest()[:16], 16) / 2**64
    return "test" if value >= 0.95 else "dev" if value >= 0.90 else "train"


def write_jsonl_gz(path: Path, rows) -> dict:
    count = 0
    with path.open("wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as gz:
        for row in rows:
            gz.write((json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8"))
            count += 1
    return {"path": path.relative_to(ROOT).as_posix(), "records": count,
            "bytes": path.stat().st_size, "sha256": hash_file(path)}


def main() -> int:
    sys.path.insert(0, str(ROOT / "scripts"))
    for index, expected in SHARD_SHA256.items():
        name = f"data/train/train-{index:05d}-of-00181.parquet"
        path = RAW / name
        if not path.exists():
            hf_hub_download(repo_id=DATASET, filename=name, repo_type="dataset",
                            revision=REVISION, local_dir=str(RAW))
        actual = hash_file(path)
        if actual != expected:
            raise RuntimeError(f"HPLT shard hash mismatch for {name}: {actual}")

    card = RAW / "README.md"
    if not card.exists():
        hf_hub_download(repo_id=DATASET, filename="README.md", repo_type="dataset",
                        revision=REVISION, local_dir=str(RAW))
    card_sha = hash_file(card)
    parent_manifest = PARENT / "manifest.json"
    if not parent_manifest.exists():
        raise RuntimeError("Build corpus candidate v1 first: scripts/build_reliable_corpus_candidate.py")
    if not TOKENIZER_PATH.exists() or hash_file(TOKENIZER_PATH) != TOKENIZER_SHA:
        raise RuntimeError("Pinned SLM-Tavenel tokenizer is missing or has changed")

    grams, benchmark_inputs = benchmark_ngrams()
    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
    extractor = tldextract.TLDExtract(suffix_list_urls=())
    OUT.mkdir(parents=True, exist_ok=True)

    seen_hashes: set[str] = set()
    seen_ids: set[str] = set()
    parent_counts = Counter()
    for split in ("train", "dev", "test"):
        path = PARENT / f"{split}.jsonl.gz"
        if not path.exists():
            raise RuntimeError(f"Missing parent corpus split: {path}")
        with gzip.open(path, "rt", encoding="utf-8") as source, (OUT / f"{split}.jsonl.gz").open("wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as target:
            for line in source:
                record = json.loads(line)
                if record["id"] in seen_ids or record["text_sha256"] in seen_hashes:
                    raise RuntimeError("Parent corpus already contains duplicate IDs or texts")
                seen_ids.add(record["id"])
                seen_hashes.add(record["text_sha256"])
                target.write((json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8"))
                parent_counts[split] += 1

    counters = Counter()
    languages = Counter()
    registers = Counter()
    splits = Counter()
    token_counts = Counter()
    source_files = []
    decision_path = OUT / "decisions.jsonl.gz"
    decisions_raw = decision_path.open("wb")
    decisions_gz = gzip.GzipFile(fileobj=decisions_raw, mode="wb", mtime=0)
    output_raw = {split: (OUT / f"{split}.jsonl.gz").open("ab") for split in ("train", "dev", "test")}
    output_gz = {split: gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) for split, raw in output_raw.items()}
    pending: dict[str, list[dict]] = {split: [] for split in ("train", "dev", "test")}

    def flush(split: str) -> None:
        rows = pending[split]
        if not rows:
            return
        encoded = tokenizer.encode_batch([row["text"] for row in rows], add_special_tokens=False)
        for row, tokenized in zip(rows, encoded):
            row["token_count"] = len(tokenized.ids)
            token_counts[(split, "en" if row["language"] == "en" else "fr")] += row["token_count"]
            output_gz[split].write((json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8"))
            splits[split] += 1
        pending[split] = []

    try:
        for index in SHARD_SHA256:
            rel = f"data/train/train-{index:05d}-of-00181.parquet"
            path = RAW / rel
            pf = pq.ParquetFile(path)
            if pf.metadata.num_rows != 250_000:
                raise RuntimeError(f"Unexpected row count in {rel}: {pf.metadata.num_rows}")
            source_files.append({"path": rel, "records": pf.metadata.num_rows,
                                 "bytes": path.stat().st_size, "sha256": SHARD_SHA256[index],
                                 "upstream_split": "train"})
            print(f"Processing HPLT train shard {index + 1}/{len(SHARD_SHA256)}: {rel}", flush=True)
            for batch in pf.iter_batches(batch_size=2048, columns=["lang", "prob", "score", "doc_id", "url", "web-register", "text", "word_count", "char_count", "collection"]):
                for row in batch.to_pylist():
                    counters["input"] += 1
                    lang = row.get("lang") or ""
                    text = row.get("text") or ""
                    reason = None
                    if lang not in ("en", "fr"):
                        reason = "language_not_selected"
                    elif float(row.get("score") or 0) < 10:
                        reason = "hplt_quality_score_below_10"
                    else:
                        prob = row.get("prob") or []
                        if not prob or float(prob[0]) < MIN_LANG_PROB:
                            reason = "language_confidence_below_0_90"
                        else:
                            labels = row.get("web-register") or {}
                            label_probs = {key: float(labels.get(key) or 0) for key in REGISTER_CLASSES}
                            register = max(label_probs, key=label_probs.get) if label_probs else "unknown"
                            registers[(lang, register)] += 1
                            mime = row.get("collection") or ""
                            if register in MAX_REG_CHARS:
                                reason = "machine_translated_or_lyrical_register"
                            elif mime not in MIME_ALLOWLIST:
                                reason = "non_textual_or_unrecognized_mime"
                            elif len(text) < MIN_CHARS or len(text) > MAX_CHARS:
                                reason = "outside_character_length_limits"
                            elif "\ufffd" in text:
                                reason = "replacement_character"
                            elif len(re.findall(r"[\w]", text, re.UNICODE)) / max(1, len(text)) < MIN_ALNUM_RATIO:
                                reason = "low_alphanumeric_ratio"
                            elif EMAIL_RE.search(text) or PHONE_RE.search(text) or IP_RE.search(text) or SSN_RE.search(text):
                                reason = "direct_identifier_pattern"
                            elif SENSITIVE_RE.search(text):
                                reason = "sensitive_topic_signal"
                            elif overlaps_benchmark(text, grams):
                                reason = "benchmark_13gram_overlap"
                    digest = sha256(text.encode("utf-8")) if text else ""
                    if reason is None and digest in seen_hashes:
                        reason = "exact_duplicate_with_existing_or_hplt_data"
                    source_id = row.get("doc_id") or f"row-{counters['input']}"
                    if reason:
                        counters["excluded_" + reason] += 1
                        decision = {"source_dataset": DATASET, "source_revision": REVISION,
                                    "source_file": rel, "source_id": source_id,
                                    "decision": "excluded", "reason": reason}
                        decisions_gz.write((json.dumps(decision, sort_keys=True) + "\n").encode())
                        continue

                    host = site_key(row.get("url") or "", extractor)
                    split = split_for_host(host)
                    record_id = f"hplt-eu:{source_id}:{digest[:16]}"
                    if record_id in seen_ids:
                        counters["excluded_duplicate_id"] += 1
                        continue
                    seen_ids.add(record_id)
                    seen_hashes.add(digest)
                    language = "en" if lang == "en" else "fr"
                    record = {
                        "id": record_id, "text": text, "language": language,
                        "source_dataset": DATASET, "source_revision": REVISION,
                        "source_file": rel, "source_id": source_id,
                        "source_url": row.get("url"), "source_collection": row.get("collection"),
                        "source_type": register, "license": "CC0-1.0",
                        "license_status": "dataset_level_CC0_declared_as_inherited_from_HPLT_v3; original_page_rights_not_independently_verified",
                        "hplt_quality_score": row.get("score"),
                        "hplt_language_confidence": float((row.get("prob") or [0])[0]),
                        "word_count_upstream": row.get("word_count"),
                        "text_sha256": digest, "split": split,
                        "attribution": f"European HPLT v1, derived from HPLT v3 web crawl; source URL: {row.get('url')}; dataset card declares inherited CC0-1.0. Original page rights not independently checked.",
                        "modifications": ["kept English/French train records with language confidence >=0.90 and HPLT WDS score >=10", "excluded predicted machine-translated and lyrical registers and unsupported MIME types", "screened for length, text quality, direct identifiers, sensitive-topic signals, and exact 13-token overlap with pinned evaluation benchmarks", "exact duplicate removal across the parent corpus and selected HPLT shards", "assigned split by registrable source domain using the bundled Public Suffix List"],
                    }
                    pending[split].append(record)
                    languages[language] += 1
                    counters["retained"] += 1
                    decision = {"source_dataset": DATASET, "source_revision": REVISION,
                                "source_file": rel, "source_id": source_id,
                                "decision": "retained", "reason": "passed_source_and_content_screens",
                                "language": language, "split": split, "text_sha256": digest}
                    decisions_gz.write((json.dumps(decision, sort_keys=True) + "\n").encode())
                    if len(pending[split]) >= 2048:
                        flush(split)
            print(f"Finished shard {index + 1}/{len(SHARD_SHA256)}; retained so far: {counters['retained']:,}", flush=True)
        for split in pending:
            flush(split)
    finally:
        for gz in output_gz.values():
            gz.close()
        for raw in output_raw.values():
            raw.close()
        decisions_gz.close()
        decisions_raw.close()

    file_info = {}
    for split in ("train", "dev", "test"):
        path = OUT / f"{split}.jsonl.gz"
        file_info[split] = {"path": path.relative_to(ROOT).as_posix(),
                            "records": parent_counts[split] + splits[split],
                            "bytes": path.stat().st_size, "sha256": hash_file(path),
                            "parent_candidate_records": parent_counts[split],
                            "hplt_records": splits[split]}
    decision_meta = {"path": decision_path.relative_to(ROOT).as_posix(),
                     "bytes": decision_path.stat().st_size, "sha256": hash_file(decision_path)}
    manifest = {
        "schema": "slm-tavenel.growth_corpus_candidate.v2",
        "status": "candidate_not_legal_clearance",
        "created": "2026-10-25",
        "parent_candidate": {"path": PARENT.relative_to(ROOT).as_posix(),
                             "manifest_sha256": hash_file(parent_manifest)},
        "sources": {
            "hplt": {"dataset": DATASET, "revision": REVISION,
                     "dataset_card": f"https://huggingface.co/datasets/{DATASET}/tree/{REVISION}",
                     "pinned_dataset_card_sha256": card_sha,
                     "upstream_project": "HPLT Monolingual v3, derived from Internet Archive and Common Crawl crawls",
                     "declared_license": "CC0-1.0, inherited from HPLT v3 per the pinned dataset card",
                     "source_rights_note": "HPLT says it does not own original crawled pages and leaves legal compliance to downstream users; CC0 declaration is recorded, not treated as independent proof of each page's rights.",
                     "selected_train_shards": source_files,
                     "input_records": counters["input"], "retained_records": counters["retained"],
                     "retained_by_language": dict(languages), "retained_by_split": dict(splits),
                     "register_counts_before_later_filters": {f"{lang}|{reg}": n for (lang, reg), n in sorted(registers.items())}},
        },
        "filters": {"languages": ["en", "fr"], "minimum_language_confidence": MIN_LANG_PROB,
                    "minimum_hplt_wds_score": 10, "excluded_web_register_top_labels": sorted(MAX_REG_CHARS),
                    "mime_allowlist": sorted(MIME_ALLOWLIST), "minimum_characters": MIN_CHARS,
                    "maximum_characters": MAX_CHARS, "minimum_alphanumeric_ratio": MIN_ALNUM_RATIO,
                    "direct_identifier_regexes": ["email", "phone", "contextual IPv4", "contextual US SSN"],
                    "sensitive_topic_blocklist": "configured English/French patient, clinical, medical-record, biometric and surveillance terms",
                    "benchmark_decontamination": {"exact_normalized_13_token_overlap": True,
                                                   "benchmarks_are_decontamination_only_not_training_data": True,
                                                   "inputs": benchmark_inputs},
                    "deduplication": "exact UTF-8 text SHA-256 against parent corpus and all selected HPLT shards; no local MinHash stage",
                    "split": "registrable source domain hashed deterministically: 90% train, 5% dev, 5% corpus holdout; these are corpus splits, not the project's evaluation benchmarks"},
        "counts": {"parent_by_split": dict(parent_counts), "hplt_by_split": dict(splits),
                   "combined_by_split": {split: file_info[split]["records"] for split in file_info},
                   "hplt_by_language": dict(languages), "hplt_decisions": dict(counters)},
        "token_counts": {"tokenizer": TOKENIZER_PATH.relative_to(ROOT).as_posix(),
                         "tokenizer_sha256": TOKENIZER_SHA,
                         "hplt_by_split_language": {f"{split}_{lang}": count for (split, lang), count in sorted(token_counts.items())},
                         "hplt_totals_by_split": {split: sum(n for (s, _), n in token_counts.items() if s == split) for split in ("train", "dev", "test")}},
        "files": {**file_info, "decisions": decision_meta},
        "limitations": ["HPLT data are web-crawl text; source-page rights are not verified record by record.",
                        "HPLT's declared CC0 status does not itself establish that every original page is in the public domain or licensed by its rightsholder.",
                        "Automated identifier and topic screens do not provide complete PII removal or human privacy review.",
                        "No local MinHash near-duplicate pass was applied; exact duplicates and the upstream HPLT dedup claim are recorded.",
                        "The output is a candidate data-growth corpus, not a legal clearance, release authorization, or model validation.",
                        "No training or benchmark evaluation was run by this builder."],
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": OUT.relative_to(ROOT).as_posix(), "counts": manifest["counts"],
                      "hplt_tokens": manifest["token_counts"],
                      "files": {key: value["sha256"] for key, value in file_info.items()}}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
