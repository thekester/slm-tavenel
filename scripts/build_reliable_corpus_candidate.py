#!/usr/bin/env python3
"""Build a pinned, auditable bilingual SLM-Tavenel corpus candidate.

Sources: the existing FSC CC0 v4 selection and the CC-BY-only OpenBallast T2
partition. OpenBallast book license files are fetched from their exact pinned
Git commits and checked before any passage is retained.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import gzip
import hashlib
import json
import re
import sys
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
REVISION = "e53b6dd0c2fb6a16909e30f6fdb2c5d787878cdf"
DATASET = "OpenBallast/ballast-t2"
RAW = ROOT / "data/raw/openballast-t2-screen-2026-10-24"
OUT = ROOT / "data/processed/slm-tavenel-reliable-corpus-candidate-v1"
FSCDIR = ROOT / "data/processed/fsc-cc0-open-weights-v4"
FSCREV = "bfa114c09d7c28dcd0a8a47fe445c64b4a3c6156"
PARQUET = "passages_tier=T2_lang=en_lic=by_rank_bucket=7_part-0.parquet"
EXPECTED_PARQUET_SHA = "4f963d7052352fe576850c03990a553f6988eb007be85e8b260ff1a41427b8b5"
COMMON_DATASET = "PleIAs/common_corpus"
COMMON_REVISION = "307910e4c5d040d6f318e6edf2a2b97849155771"
COMMON_FILE = "common_corpus_1/subset_100_1.parquet"
COMMON_RAW = ROOT / "data/raw/pleias-common-corpus-2026-10-24/subset_100_1.parquet"
COMMON_SHA = "4ee719a130b7f86978b08c20cc9f490309e2bae2a5c0df4f04fd427365530f07"
COMMON_ALLOWLIST = {
    ("OpenAlex", "CC-By"),
    ("Eurlex", "CC-By"),
    ("French-Science-Pile", "Various open science"),
    ("govinfo", "Public Domain"),
    ("dotgov", "Public Domain"),
}
EXPECTED_FILES = {
    "README.md": None,
    "manifest.json": None,
    "licenses.parquet": None,
    "sources.parquet": None,
    PARQUET: EXPECTED_PARQUET_SHA,
}

BENCHMARKS = [
    "arc_easy.jsonl", "arc_challenge.jsonl", "hellaswag.jsonl",
    "belebele_en.jsonl", "belebele_fr.jsonl", "winogrande.jsonl",
]

TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)
EMAIL_RE = re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b")
PHONE_RE = re.compile(r"(?<!\w)(?:\+?\d{1,3}[ .-]?)?(?:\(\d{2,4}\)[ .-]?|\d{3}[ .-])\d{3}[ .-]\d{4}(?!\w)")
IP_CONTEXT_RE = re.compile(r"(?i)(?:\bIP(?:v4)? address\b|\bIPv4\b|\bhost address\b|\bserver address\b)\D{0,25}(?:\d{1,3}\.){3}\d{1,3}")
SSN_CONTEXT_RE = re.compile(r"(?i)(?:\bSSN\b|\bSocial Security (?:number|No\.?))\D{0,20}\d{3}-\d{2}-\d{4}")
COMMON_RISK_RE = re.compile(r"(?i)\b(patient|participant|clinical trial|case report|human subject|medical record|dossier m[eé]dical|essai clinique|cas clinique|sujet humain|diagnosis|diagnostic|biometric|facial recognition|surveillance)\b")
TOKENIZER_PATH = ROOT / "exports/slm-tavenel-1m-exp0010-release-candidate-v4/tokenizer.json"
TOKENIZER_SHA = "93c7b9364de2c42315571c851353862e63830bfbeb220dc14dde80e6ca82cfcb"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def download(url: str, destination: Path, expected_sha: str | None = None) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        digest = sha256(destination.read_bytes())
        if expected_sha is None or digest == expected_sha:
            return
        destination.unlink()
    request = urllib.request.Request(url, headers={"User-Agent": "slm-tavenel-data-audit/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response, destination.open("wb") as out:
        while chunk := response.read(1024 * 1024):
            out.write(chunk)
    digest = sha256(destination.read_bytes())
    if expected_sha and digest != expected_sha:
        destination.unlink(missing_ok=True)
        raise RuntimeError(f"SHA-256 mismatch for {url}: expected {expected_sha}, got {digest}")


def normalize(text: str) -> str:
    return " ".join(TOKEN_RE.findall(text.casefold()))


def load_eval_ngrams(n: int = 13) -> set[tuple[str, ...]]:
    grams: set[tuple[str, ...]] = set()
    for name in BENCHMARKS:
        path = ROOT / "data/benchmarks" / name
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                chunks = [row.get("context", ""), *row.get("choices", [])]
                for chunk in chunks:
                    tokens = TOKEN_RE.findall(chunk.casefold())
                    grams.update(tuple(tokens[i:i+n]) for i in range(max(0, len(tokens)-n+1)))
    return grams


def has_benchmark_overlap(text: str, benchmark_ngrams: set[tuple[str, ...]], n: int = 13) -> bool:
    tokens = TOKEN_RE.findall(text.casefold())
    return any(tuple(tokens[i:i+n]) in benchmark_ngrams for i in range(max(0, len(tokens)-n+1)))


def benchmark_input_manifest() -> list[dict]:
    inputs = []
    for name in BENCHMARKS:
        path = ROOT / "data/benchmarks" / name
        manifest_path = path.with_suffix(".manifest.json")
        if not path.exists() or not manifest_path.exists():
            raise RuntimeError(f"Pinned benchmark input or manifest missing: {name}")
        actual = sha256(path.read_bytes())
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected = manifest.get("sha256")
        if expected != actual:
            raise RuntimeError(f"Benchmark file hash does not match its manifest: {name}")
        inputs.append({"path": path.relative_to(ROOT).as_posix(), "manifest_path": manifest_path.relative_to(ROOT).as_posix(), "sha256": actual, "dataset": manifest.get("repo"), "revision": manifest.get("revision"), "items": manifest.get("items")})
    return inputs


def verify_book_license(source: dict) -> tuple[str, str, str]:
    access_path = source["access_path"]
    match = re.fullmatch(r"git:(https://github\.com/[^@]+)@([0-9a-f]{7,40})", access_path)
    if not match:
        raise ValueError(f"Unpinned or unrecognized OpenStax source path: {access_path}")
    repo, short_commit = match.groups()
    commit = source["snapshot_id"]
    if not re.fullmatch(r"[0-9a-f]{40}", commit) or not commit.startswith(short_commit):
        raise ValueError(f"Pinned source commit does not match snapshot id: {access_path}")
    github_path = repo.removeprefix("https://github.com/").removesuffix(".git")
    url = f"https://raw.githubusercontent.com/{github_path}/{commit}/LICENSE"
    request = urllib.request.Request(url, headers={"User-Agent": "slm-tavenel-data-audit/1.0"})
    with urllib.request.urlopen(request, timeout=30) as response:
        content = response.read()
    text = content.decode("utf-8", errors="replace")
    if "Attribution 4.0 International" not in text:
        raise ValueError(f"Pinned LICENSE does not identify CC BY 4.0: {url}")
    if re.search(r"NonCommercial|Attribution-NonCommercial|BY-NC", text, re.I):
        raise ValueError(f"Pinned LICENSE appears to contain NC terms: {url}")
    return url, sha256(content), text


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-license-fetch", action="store_true", help="Use previously saved verified LICENSE files")
    args = parser.parse_args()

    for filename, expected in EXPECTED_FILES.items():
        path = RAW / filename
        if not path.exists():
            url = f"https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/" + filename.replace("_", "/", 1) if filename != PARQUET else f"https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/passages/tier=T2/lang=en/lic=by/rank_bucket=7/part-0.parquet"
            download(url, path, expected)
    passages_path = RAW / PARQUET
    if sha256(passages_path.read_bytes()) != EXPECTED_PARQUET_SHA:
        raise RuntimeError("OpenBallast CC-BY partition hash mismatch")
    if not COMMON_RAW.exists():
        download(f"https://huggingface.co/datasets/{COMMON_DATASET}/resolve/{COMMON_REVISION}/{COMMON_FILE}", COMMON_RAW, COMMON_SHA)
    if sha256(COMMON_RAW.read_bytes()) != COMMON_SHA:
        raise RuntimeError("Pinned Common Corpus shard hash mismatch")

    source_table = pq.read_table(RAW / "sources.parquet").to_pylist()
    license_table = {row["license_id"]: row for row in pq.read_table(RAW / "licenses.parquet").to_pylist()}
    allowed_license = license_table.get("CC-BY-4.0")
    if not allowed_license or allowed_license["share_alike"] or not allowed_license["commercial_ok"]:
        raise RuntimeError("OpenBallast license registry does not match the CC-BY-4.0 allowlist")
    by_sources = {row["source_id"]: row for row in source_table if row["default_license_id"] == "CC-BY-4.0"}
    if len(by_sources) != 61:
        raise RuntimeError(f"Expected 61 CC-BY source books, found {len(by_sources)}")

    license_dir = RAW / "verified-book-licenses"
    license_dir.mkdir(parents=True, exist_ok=True)
    verified_licenses: dict[str, dict] = {}
    def audit_one(item):
        source_id, source = item
        path = license_dir / f"{source_id}.LICENSE"
        if args.skip_license_fetch and path.exists():
            content = path.read_bytes()
            text = content.decode("utf-8", errors="replace")
            github_path = source["access_path"].split("@")[0].removeprefix("git:https://github.com/").removesuffix(".git")
            url = f"https://raw.githubusercontent.com/{github_path}/{source['snapshot_id']}/LICENSE"
            if "Attribution 4.0 International" not in text or re.search(r"NonCommercial|Attribution-NonCommercial|BY-NC", text, re.I):
                raise ValueError(f"Saved license does not pass CC-BY check: {source_id}")
            return source_id, {"url": url, "sha256": sha256(content), "file": path.relative_to(ROOT).as_posix()}
        url, digest, text = verify_book_license(source)
        path.write_text(text, encoding="utf-8")
        return source_id, {"url": url, "sha256": digest, "file": path.relative_to(ROOT).as_posix()}
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        for source_id, record in pool.map(audit_one, by_sources.items()):
            verified_licenses[source_id] = record

    rows = pq.read_table(passages_path).to_pylist()
    if len(rows) != 90445:
        raise RuntimeError(f"Unexpected OpenBallast row count: {len(rows)}")
    common_rows = pq.read_table(COMMON_RAW).to_pylist()
    if len(common_rows) != 69907:
        raise RuntimeError(f"Unexpected Common Corpus shard row count: {len(common_rows)}")
    split_data = json.loads((FSCDIR / "splits/split_manifest.json").read_text(encoding="utf-8"))
    fsc_split: dict[str, str] = {}
    for split_record in split_data["splits"]:
        path = ROOT / split_record["path"]
        with path.open("r", encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                fsc_split[row["id"]] = split_record["split"]

    benchmark_ngrams = load_eval_ngrams()
    benchmark_inputs = benchmark_input_manifest()
    group_sizes = Counter(
        by_sources[row["source_id"]]["snapshot_id"]
        for row in rows
        if row["source_id"] in by_sources and row["license_id"] == "CC-BY-4.0"
    )
    targets = {"train": 0.8 * sum(group_sizes.values()), "dev": 0.1 * sum(group_sizes.values()), "test": 0.1 * sum(group_sizes.values())}
    assigned = Counter()
    split_by_group = {}
    for group, size in sorted(group_sizes.items(), key=lambda pair: (-pair[1], hashlib.sha256(("slm-tavenel-openballast-book-split-v1:" + pair[0]).encode()).hexdigest())):
        split = min(targets, key=lambda name: ((assigned[name] + size) / targets[name], (0 if name == "train" else 1)))
        split_by_group[group] = split
        assigned[split] += size
    rejection = Counter()
    output_rows: dict[str, list[dict]] = defaultdict(list)
    decisions: list[dict] = []
    seen_text: set[str] = set()
    source_counts: Counter = Counter()
    split_counts: Counter = Counter()
    for row in rows:
        source = by_sources.get(row["source_id"])
        reason = None
        if source is None or row["license_id"] != "CC-BY-4.0":
            reason = "license_or_source_not_allowlisted"
        elif row["license_evidence"] != f"per-record-metadata:LICENSE@{source['snapshot_id'][:8]}":
            reason = "record_license_evidence_mismatch"
        elif row["revision"] != source["snapshot_id"]:
            reason = "source_revision_mismatch"
        elif not row["source_url"].startswith(source["access_path"].split("@")[0].removeprefix("git:").removesuffix(".git") + "/blob/" + source["snapshot_id"] + "/"):
            reason = "source_url_revision_mismatch"
        text = row["text"] or ""
        if reason is None:
            if "\ufffd" in text:
                reason = "replacement_character"
            elif len(text) < 120 or len(TOKEN_RE.findall(text)) < 20:
                reason = "too_short_or_low_content"
            elif len(text) > 20000:
                reason = "overlong_chunk"
            elif EMAIL_RE.search(text) or PHONE_RE.search(text) or IP_CONTEXT_RE.search(text) or SSN_CONTEXT_RE.search(text):
                reason = "direct_identifier_pattern"
            elif len(re.findall(r"[\w]", text, re.UNICODE)) / max(1, len(text)) < 0.48:
                reason = "low_alphanumeric_ratio"
            elif has_benchmark_overlap(text, benchmark_ngrams):
                reason = "benchmark_13gram_overlap"
            elif row["text_sha256"] != sha256(text.encode("utf-8")):
                reason = "upstream_text_hash_mismatch"
            elif row["text_sha256"] in seen_text:
                reason = "exact_duplicate"
        if reason:
            rejection[reason] += 1
            decisions.append({"source_dataset": DATASET, "record_id": row["chunk_id"], "source_id": row["source_id"], "decision": "excluded", "reason": reason})
            continue
        seen_text.add(row["text_sha256"])
        book_group = source["snapshot_id"]
        split = split_by_group[book_group]
        source_info = source
        record = {
            "id": f"openballast-t2:{row['chunk_id']}",
            "text": text,
            "language": "en",
            "source_dataset": DATASET,
            "source_revision": REVISION,
            "source_id": row["source_id"],
            "source_name": source_info["name"],
            "source_url": row["source_url"],
            "source_commit": source_info["snapshot_id"],
            "book_homepage": source_info["homepage"],
            "license": row["license_id"],
            "license_url": allowed_license["license_url"],
            "license_evidence": row["license_evidence"],
            "license_file_url": verified_licenses[row["source_id"]]["url"],
            "license_file_sha256": verified_licenses[row["source_id"]]["sha256"],
            "attribution": f"OpenStax, {source_info['name']}; {source_info['homepage']}; CC BY 4.0. Text extracted and chunked; privacy and benchmark-overlap filters applied.",
            "text_sha256": row["text_sha256"],
            "split": split,
            "split_group": book_group,
            "modifications": ["extracted from pinned official OpenStax source", "chunked upstream by OpenBallast", "screened for direct identifier patterns", "screened against pinned project benchmark texts", "exact duplicates removed"],
        }
        output_rows[split].append(record)
        source_counts[row["source_id"]] += 1
        split_counts[split] += 1
        decisions.append({"source_dataset": DATASET, "record_id": row["chunk_id"], "source_id": row["source_id"], "decision": "retained", "reason": "all source, license, quality, privacy-pattern, dedup, and benchmark-overlap rules passed", "split": split})

    # Include the existing bilingual FSC v4 records, preserving their recorded split and provenance.
    fsc_seen = set(seen_text)
    fsc_counts = Counter()
    with (FSCDIR / "documents.jsonl").open("r", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            split = fsc_split.get(row["id"])
            if split is None:
                raise RuntimeError(f"FSC document lacks predeclared split: {row['id']}")
            if row["license"] != "CC0-1.0" or row["source_revision"] != FSCREV:
                raise RuntimeError(f"FSC source/license mismatch: {row['id']}")
            text = row["text"]
            if has_benchmark_overlap(text, benchmark_ngrams):
                rejection["fsc_benchmark_13gram_overlap"] += 1
                decisions.append({"source_dataset": "PleIAs/French-Science-Commons", "record_id": row["id"], "decision": "excluded", "reason": "benchmark_13gram_overlap"})
                continue
            digest = sha256(text.encode("utf-8"))
            if digest in fsc_seen:
                rejection["fsc_exact_duplicate"] += 1
                decisions.append({"source_dataset": "PleIAs/French-Science-Commons", "record_id": row["id"], "decision": "excluded", "reason": "exact_duplicate"})
                continue
            fsc_seen.add(digest)
            record = {
                "id": row["id"], "text": text,
                "language": "fr" if row["source"].endswith("-fr") else "en",
                "source_dataset": "PleIAs/French-Science-Commons",
                "source_revision": FSCREV, "source_id": row["source"],
                "source_document_id": row["source_doc_id"], "license": row["license"],
                "source_metadata": row.get("source_metadata", ""),
                "text_sha256": digest, "split": split,
                "attribution": "Source metadata retained; CC0-1.0 declared in the pinned dataset and matched against HAL versions by the existing project audit. Rights-holder authority remains unverified.",
                "modifications": ["selected and filtered by FSC CC0 v4 pipeline", "document-level split preserved", "screened against pinned project benchmark texts", "exact duplicates removed"],
            }
            output_rows[split].append(record)
            fsc_counts[(record["language"], split)] += 1
            decisions.append({"source_dataset": "PleIAs/French-Science-Commons", "record_id": row["id"], "decision": "retained", "reason": "existing FSC CC0 v4 selection passed additional benchmark decontamination", "split": split})

    # Add a traceable slice from Common Corpus instead of treating the whole
    # aggregate as one approved/blocked dataset. Preserve every upstream label.
    common_counts = Counter()
    common_source_counts = Counter()
    common_rows_by_license = Counter()
    for row in common_rows:
        collection = row.get("collection") or ""
        license_label = row.get("license") or ""
        language = row.get("language") or ""
        identifier = row.get("identifier") or ""
        text = row.get("text") or ""
        reason = None
        if (collection, license_label) not in COMMON_ALLOWLIST:
            reason = "source_or_declared_license_not_allowlisted"
        elif language not in ("English", "French") or row.get("language_type") != "Written":
            reason = "language_or_modality_not_selected"
        elif not identifier.strip():
            reason = "missing_stable_source_identifier"
        elif "\ufffd" in text:
            reason = "replacement_character"
        elif len(text) < 300 or len(TOKEN_RE.findall(text)) < 50:
            reason = "too_short_or_low_content"
        elif len(text) > 200000:
            reason = "overlong_document"
        elif EMAIL_RE.search(text) or PHONE_RE.search(text) or IP_CONTEXT_RE.search(text) or SSN_CONTEXT_RE.search(text):
            reason = "direct_identifier_pattern"
        elif COMMON_RISK_RE.search(text):
            reason = "sensitive_or_human_subject_content_signal"
        elif len(re.findall(r"[\w]", text, re.UNICODE)) / max(1, len(text)) < 0.48:
            reason = "low_alphanumeric_ratio"
        elif has_benchmark_overlap(text, benchmark_ngrams):
            reason = "benchmark_13gram_overlap"
        digest = sha256(text.encode("utf-8")) if text else ""
        if reason is None and digest in fsc_seen:
            reason = "exact_duplicate_across_sources"
        if reason:
            rejection[f"common_{reason}"] += 1
            decisions.append({"source_dataset": COMMON_DATASET, "record_id": identifier, "source_id": collection, "declared_license_label": license_label, "decision": "excluded", "reason": reason})
            continue
        split_number = int(hashlib.sha256(("slm-tavenel-common-corpus-split-v1:" + collection + ":" + identifier).encode()).hexdigest()[:16], 16) / 2**64
        split = "test" if split_number >= 0.9 else "dev" if split_number >= 0.8 else "train"
        fsc_seen.add(digest)
        stable_id = f"common-corpus:{collection}:{identifier}:{digest[:16]}"
        record = {
            "id": stable_id,
            "text": text,
            "language": "en" if language == "English" else "fr",
            "source_dataset": COMMON_DATASET,
            "source_revision": COMMON_REVISION,
            "source_file": COMMON_FILE,
            "source_id": identifier,
            "source_collection": collection,
            "source_type": row.get("open_type"),
            "source_curator": row.get("curator"),
            "title": row.get("title"),
            "creator": row.get("creator"),
            "date": row.get("date"),
            "license": license_label,
            "license_status": "upstream_declared_label; exact source terms not independently checked by this filter",
            "word_count_upstream": row.get("word_count"),
            "token_count_upstream": row.get("token_count"),
            "text_sha256": digest,
            "split": split,
            "attribution": f"{row.get('creator') or 'Creator not supplied'}, {row.get('title') or 'title not supplied'}; source {identifier}; PleIAs Common Corpus collection {collection}. Upstream license label: {license_label}. Source text screened and split by document; verify any record-specific attribution before redistribution.",
            "modifications": ["selected from pinned Common Corpus shard using collection, language, modality, and upstream declared license label", "screened for configured quality, direct-identifier, content-risk, and benchmark-overlap signals", "exact duplicates removed across included sources", "document-level split assigned deterministically"],
        }
        output_rows[split].append(record)
        common_counts[split] += 1
        common_source_counts[(collection, license_label)] += 1
        common_rows_by_license[license_label] += 1
        decisions.append({"source_dataset": COMMON_DATASET, "record_id": identifier, "source_id": collection, "declared_license_label": license_label, "decision": "retained", "reason": "allowlisted source and upstream license label passed language, quality, privacy-pattern, duplicate, and benchmark-overlap screens", "split": split})

    all_candidate_ids = [record["id"] for records in output_rows.values() for record in records]
    if len(all_candidate_ids) != len(set(all_candidate_ids)):
        raise RuntimeError("Output document IDs are not unique")
    OUT.mkdir(parents=True, exist_ok=True)
    file_info = {}
    tokenizer_sha = sha256(TOKENIZER_PATH.read_bytes())
    if tokenizer_sha != TOKENIZER_SHA:
        raise RuntimeError("Pinned SLM-Tavenel tokenizer hash mismatch")
    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
    tokens_by_split_language = Counter()
    tokens_by_split_source = Counter()
    for split, records in output_rows.items():
        path = OUT / f"{split}.jsonl.gz"
        records.sort(key=lambda r: r["id"])
        with path.open("wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as gz:
            for record in records:
                token_count = len(tokenizer.encode(record["text"]).ids)
                record["token_count"] = token_count
                tokens_by_split_language[(split, record["language"])] += token_count
                tokens_by_split_source[(split, record["source_dataset"])] += token_count
                gz.write((json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8"))
        file_info[split] = {"path": path.relative_to(ROOT).as_posix(), "records": len(records), "bytes": path.stat().st_size, "sha256": sha256(path.read_bytes()), "openballast_records": split_counts[split], "fsc_records": len([r for r in records if r["source_dataset"] == "PleIAs/French-Science-Commons"]), "common_corpus_records": common_counts[split]}
    decision_path = OUT / "decisions.jsonl.gz"
    with decision_path.open("wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as gz:
        for row in decisions:
            gz.write((json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8"))

    source_ledger = []
    for source_id in sorted(source_counts):
        source = by_sources[source_id]
        source_ledger.append({**source, "license": "CC-BY-4.0", "license_url": allowed_license["license_url"], **verified_licenses[source_id], "retained_chunks": source_counts[source_id], "attribution": f"OpenStax, {source['name']}; {source['homepage']}; CC BY 4.0. Modifications: extraction, chunking, filtering and document-level splitting."})
    (OUT / "openballast-source-attribution.json").write_text(json.dumps(source_ledger, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    common_ledger = []
    for split, records in output_rows.items():
        for record in records:
            if record["source_dataset"] == COMMON_DATASET:
                common_ledger.append({key: record.get(key) for key in ["id", "source_dataset", "source_revision", "source_file", "source_id", "source_collection", "source_type", "source_curator", "title", "creator", "date", "language", "license", "license_status", "split", "text_sha256", "attribution"]})
    common_ledger.sort(key=lambda row: row["id"])
    common_ledger_path = OUT / "common-corpus-attribution.json"
    common_ledger_path.write_text(json.dumps(common_ledger, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    manifest = {
        "schema": "slm-tavenel.corpus_candidate.v1",
        "status": "candidate_not_legal_clearance",
        "created": "2026-10-24",
        "benchmark_decontamination": {"rule": "exclude whole document if any exact normalized 13-token sequence overlaps any prompt or choice in pinned local benchmark files", "benchmark_inputs": benchmark_inputs, "benchmark_ngrams": len(benchmark_ngrams)},
        "sources": {
            "openballast": {"dataset": DATASET, "revision": REVISION, "partition": "passages/tier=T2/lang=en/lic=by/rank_bucket=7/part-0.parquet", "source_partition_sha256": EXPECTED_PARQUET_SHA, "input_records": len(rows), "eligible_source_books": len(by_sources), "verified_pinned_license_files": len(verified_licenses), "license": "CC-BY-4.0", "license_verification": "official GitHub LICENSE at each source's pinned commit contains Creative Commons Attribution 4.0 International and no NC marker", "filtered_source_books": len(source_counts)},
            "fsc": {"dataset": "PleIAs/French-Science-Commons", "revision": FSCREV, "selection": "existing local fsc-cc0-open-weights-v4 documents.jsonl", "license": "CC0-1.0 declared; existing HAL metadata audit", "input_records": sum(fsc_counts.values()), "source_rights_review_signed": False, "manual_privacy_review_signed": False},
            "common_corpus": {"dataset": COMMON_DATASET, "revision": COMMON_REVISION, "file": COMMON_FILE, "file_sha256": COMMON_SHA, "input_records": len(common_rows), "allowlisted_collection_and_license_pairs": [{"collection": collection, "upstream_license_label": license_label, "retained_records": count} for (collection, license_label), count in sorted(common_source_counts.items())], "retained_by_split": dict(common_counts), "license_note": "upstream labels are preserved as supplied; CC-By version is unspecified and Various open science is a category, not a record-level license. This screening does not independently verify source rights."},
        },
        "rules": {"openballast_allowlist": "exact CC-BY-4.0 row and source license, source URL and revision match, pinned official LICENSE fetched and checked", "common_corpus_allowlist": sorted([{"collection": collection, "upstream_license_label": license_label} for collection, license_label in COMMON_ALLOWLIST], key=lambda r: (r["collection"], r["upstream_license_label"])), "common_corpus_scope": "only one pinned shard; Written English/French; preserve upstream license strings; metadata labels are treated as traceability signals, not legal proof", "excluded": ["CC-BY-NC-SA", "unknown license", "unmapped source", "license or revision mismatch", "short or low-content text", "overlong chunks", "replacement characters", "direct email/phone/IP/SSN patterns", "configured sensitive-content signals", "low alphanumeric ratio", "13-token benchmark overlap", "exact duplicates"], "split": "OpenBallast groups all books sharing a pinned upstream repository commit; deterministic greedy allocation by group size and commit hash targets 80/10/10 document proportions; FSC v4 document splits preserved; Common Corpus records split deterministically by collection and identifier"},
        "counts": {"retained": {split: info["records"] for split, info in file_info.items()}, "openballast_by_split": dict(split_counts), "openballast_source_books": len(source_counts), "fsc_by_language_split": {f"{lang}_{split}": n for (lang, split), n in fsc_counts.items()}, "common_corpus_by_split": dict(common_counts), "common_corpus_by_source_and_license": {f"{collection}|{license_label}": n for (collection, license_label), n in common_source_counts.items()}, "excluded": dict(rejection), "decisions": len(decisions)},
        "token_counts": {"tokenizer_path": TOKENIZER_PATH.relative_to(ROOT).as_posix(), "tokenizer_sha256": tokenizer_sha, "by_split_language": {f"{split}_{language}": n for (split, language), n in tokens_by_split_language.items()}, "by_split_source": {f"{split}_{source}": n for (split, source), n in tokens_by_split_source.items()}, "totals_by_split": {split: sum(n for (s, _), n in tokens_by_split_language.items() if s == split) for split in output_rows}},
        "files": {**file_info, "decisions": {"path": decision_path.relative_to(ROOT).as_posix(), "bytes": decision_path.stat().st_size, "sha256": sha256(decision_path.read_bytes())}, "openballast_attribution": {"path": (OUT / "openballast-source-attribution.json").relative_to(ROOT).as_posix(), "sha256": sha256((OUT / "openballast-source-attribution.json").read_bytes())}, "common_corpus_attribution": {"path": common_ledger_path.relative_to(ROOT).as_posix(), "sha256": sha256(common_ledger_path.read_bytes())}},
        "limitations": ["This is a screening and corpus-construction artifact, not independent legal advice or a rights-holder authority audit.", "The existing FSC rights and manual privacy review fields remain unsigned.", "Common Corpus contribution is only one pinned shard; source labels such as CC-By and Various open science are not exact, independently verified licenses.", "English OpenBallast content is textbooks only and is not domain-balanced general web text.", "Automated identifier and benchmark filters do not establish anonymization or zero contamination.", "Attribution ledgers are included for OpenBallast and Common Corpus; upstream metadata for some Common Corpus records are incomplete.", "No model training was started by this script."],
    }
    manifest_path = OUT / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUT), "manifest": manifest, "verified_openballast_license_files": len(verified_licenses)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (urllib.error.URLError, TimeoutError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
