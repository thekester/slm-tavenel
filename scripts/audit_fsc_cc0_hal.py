"""Compare the selected FSC CC0 documents with the upstream HAL records API.

The output contains identifiers, metadata and hashes, not document text. It verifies that
the currently prepared French-Science-Commons CC0-only candidate agrees with HAL's current
recorded CC0 license. It does not independently establish copyright ownership or clear privacy.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen


API = "https://api.archives-ouvertes.fr/search/"
CC0_URL = "https://creativecommons.org/publicdomain/zero/1.0/"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_documents(paths: list[Path]) -> list[dict]:
    documents: list[dict] = []
    seen: set[str] = set()
    for path in paths:
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                row = json.loads(line)
                identifier = str(row.get("id", ""))
                metadata = row.get("source_metadata") or {}
                if not identifier or identifier in seen:
                    raise ValueError(f"Missing or duplicate document ID in {path}:{line_number}: {identifier!r}")
                if str(metadata.get("license", "")).strip().casefold() not in {"cc0", "cc0-1.0"}:
                    raise ValueError(f"Non-CC0 row in CC0 input {path}:{line_number}: {identifier}")
                if not str(metadata.get("source", "")).strip().casefold() in {"hal", "halshs", "tel", "dumas", "hprints", "pasteur", "inserm", "cea", "cel", "lirmm", "inria", "irsn", "obspm", "insu", "mnhn", "ird", "anses", "cirad"}:
                    raise ValueError(f"Unrecognized repository label for {identifier}: {metadata.get('source')!r}")
                seen.add(identifier)
                documents.append({
                    "id": identifier,
                    "language": row.get("language") or path.parent.name,
                    "source_label": metadata.get("source"),
                    "dataset_license": metadata.get("license"),
                    "text_sha256": hashlib.sha256(str(row.get("text", "")).encode("utf-8")).hexdigest(),
                })
    return documents


def chunks(values: list[dict], size: int):
    for start in range(0, len(values), size):
        yield values[start:start + size]


def query_hal(batch: list[dict], timeout: int, retries: int) -> list[dict]:
    ids = [row["id"] for row in batch]
    query = "halId_s:(" + " OR ".join(ids) + ")"
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            records = []
            total = None
            while total is None or len(records) < total:
                params = urlencode({"q": query, "fl": "halId_s,licence_s,uri_s,docType_s",
                                    "wt": "json", "rows": str(len(ids)), "start": str(len(records))})
                request = Request(f"{API}?{params}", headers={"User-Agent": "SLM-Tavenel corpus-rights audit/1.0"})
                with urlopen(request, timeout=timeout) as response:
                    payload = json.load(response)["response"]
                found = int(payload["numFound"])
                if total is not None and found != total:
                    raise ValueError("HAL result count changed during pagination")
                total = found
                page = payload["docs"]
                if len(records) + len(page) > total or (not page and len(records) < total):
                    raise ValueError("Incomplete or inconsistent HAL response")
                records.extend(page)
            return records
        except Exception as exc:  # network/API transient errors are retried and then surfaced
            last_error = exc
            if attempt + 1 < retries:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"HAL API request failed after {retries} attempts: {last_error}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--filtered-output-dir", type=Path,
                        help="write a local CC0-only corpus after excluding unresolved or mismatched HAL IDs")
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--timeout", type=int, default=40)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--request-pause", type=float, default=0.2)
    args = parser.parse_args()
    if not 1 <= args.batch_size <= 100 or args.request_pause < 0:
        parser.error("--batch-size must be in [1,100] and --request-pause must be nonnegative")

    documents = load_documents(args.inputs)
    api_records: dict[str, list[dict]] = {}
    batches = list(chunks(documents, args.batch_size))
    for index, batch in enumerate(batches):
        for record in query_hal(batch, args.timeout, args.retries):
            identifier = record.get("halId_s")
            api_records.setdefault(identifier, []).append(record)
        if index + 1 < len(batches):
            time.sleep(args.request_pause)

    audited: list[dict] = []
    for row in documents:
        records = api_records.get(row["id"], [])
        if not records:
            outcome = "not_found"
            api_licenses = []
            record_uris = []
        elif all(record.get("licence_s") == CC0_URL for record in records):
            outcome = "cc0_all_versions"
            api_licenses = sorted({record.get("licence_s") for record in records})
            record_uris = sorted({record.get("uri_s") for record in records if record.get("uri_s")})
        else:
            outcome = "license_conflict" if any(record.get("licence_s") == CC0_URL for record in records) else "license_mismatch"
            api_licenses = sorted({record.get("licence_s") or "(missing)" for record in records})
            record_uris = sorted({record.get("uri_s") for record in records if record.get("uri_s")})
        audited.append({**row, "hal_uris": record_uris, "hal_licenses": api_licenses,
                        "hal_versions": len(records), "outcome": outcome})

    counts = {name: sum(row["outcome"] == name for row in audited)
              for name in ("cc0_all_versions", "license_conflict", "not_found", "license_mismatch")}
    accepted_ids = {row["id"] for row in audited if row["outcome"] == "cc0_all_versions"}
    filtered_files: list[dict] = []
    if args.filtered_output_dir:
        args.filtered_output_dir.mkdir(parents=True, exist_ok=True)
        for source_path in args.inputs:
            language = source_path.parent.name
            target_path = args.filtered_output_dir / f"{language}.jsonl.gz"
            row_count = 0
            with gzip.open(source_path, "rt", encoding="utf-8") as source, target_path.open("wb") as raw_target:
                with gzip.GzipFile(fileobj=raw_target, mode="wb", filename="", mtime=0) as compressed:
                    with io.TextIOWrapper(compressed, encoding="utf-8", newline="\n") as target:
                        for line in source:
                            row = json.loads(line)
                            if row["id"] in accepted_ids:
                                target.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                                row_count += 1
            filtered_files.append({"path": str(target_path), "sha256": sha256_file(target_path),
                                   "bytes": target_path.stat().st_size, "documents": row_count})
    report = {
        "kind": "slm.fsc_cc0_hal_license_audit",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "PleIAs/French-Science-Commons",
        "dataset_revision": "bfa114c09d7c28dcd0a8a47fe445c64b4a3c6156",
        "subset": "french_open_science",
        "source_api": API,
        "source_api_query_fields": ["halId_s", "licence_s", "uri_s", "docType_s"],
        "expected_license": CC0_URL,
        "inputs": [{"path": str(path), "sha256": sha256_file(path)} for path in args.inputs],
        "counts": {"documents": len(documents), "by_language": {
            lang: sum(row["language"] == lang for row in documents) for lang in sorted({row["language"] for row in documents})}, **counts},
        "result": "PASS" if counts["cc0_all_versions"] == len(documents) else "BLOCKED",
        "filtered_output": filtered_files,
        "limitations": [
            "Only IDs for which every version returned by HAL reports CC0 pass this check; a version-specific conflict is excluded.",
            "The live HAL metadata API is independent of the dataset card but still a rights declaration, not a judicial finding of ownership.",
            "A CC0 metadata match does not complete privacy, source-content, benchmark-contamination or model-weight review.",
            "No author identity is inferred when absent; this audit records stable repository IDs and source links.",
        ],
        "records": audited,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "counts": report["counts"], "result": report["result"]}, indent=2))
    return 0 if report["result"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
