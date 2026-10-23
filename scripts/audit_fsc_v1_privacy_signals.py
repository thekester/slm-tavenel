"""Scan the processed FSC-v1 training corpus for residual structured PII and review signals.

The output contains counts, record IDs and hashes only; it never copies text excerpts.
This is triage for human review, not a privacy attestation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import yaml

from slm.data.pii import redact


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("documents", type=Path)
    parser.add_argument("pipeline_config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = yaml.safe_load(args.pipeline_config.read_text(encoding="utf-8"))
    risk_step = next(step["publication_risk_metadata"] for step in config["steps"] if "publication_risk_metadata" in step)
    title_patterns = [(pattern, re.compile(pattern, re.IGNORECASE)) for pattern in risk_step["title_patterns"]]
    excluded_disciplines = {str(value).casefold() for value in risk_step["excluded_disciplines"]}
    sensitive_body_patterns = [
        re.compile(pattern, re.IGNORECASE)
        for pattern in (
            r"\b(patient|participant|interview|consent|clinical trial|case report|human subject|medical record|diagnosis|facial recognition|face recognition|biometric(s)?|fingerprint(s)?|surveillance|vocal fold(s)?|laryngeal)\b",
            r"\b(patient|participant|entretien|consentement|essai clinique|cas clinique|sujet humain|dossier médical|diagnostic)\b",
        )
    ]

    counts: Counter = Counter()
    flagged: dict[str, dict] = {}
    source_counts: Counter = Counter()
    language_counts: Counter = Counter()
    chars = 0
    with args.documents.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            doc = json.loads(line)
            text = str(doc.get("text", ""))
            metadata = doc.get("source_metadata") or {}
            doc_id = str(doc.get("id", ""))
            reasons: set[str] = set()
            chars += len(text)
            counts["documents"] += 1
            source_counts[str(doc.get("source", "unknown"))] += 1
            language_counts[str(metadata.get("language", "unknown"))] += 1

            _, pii_hits = redact(text)
            for kind, n in pii_hits.items():
                counts[f"residual_{kind}_matches"] += n
                if n:
                    reasons.add(f"residual_{kind}")
            if "\ufffd" in text:
                counts["documents_with_replacement_character"] += 1
                reasons.add("replacement_character")

            title = str(metadata.get("title") or "")
            for label, pattern in title_patterns:
                if pattern.search(title):
                    # Preserve hit categories as stable keys without copying matched text.
                    counts[f"title_pattern:{label}"] += 1
                    reasons.add("title_risk_signal")
            discipline = metadata.get("discipline")
            if not isinstance(discipline, str) or not discipline.strip():
                counts["documents_missing_discipline"] += 1
                reasons.add("missing_discipline")
            else:
                disciplines = {value.strip().casefold() for value in discipline.split(";") if value.strip()}
                if disciplines & excluded_disciplines:
                    counts["documents_excluded_discipline"] += 1
                    reasons.add("excluded_discipline")

            body_hits = 0
            for pattern in sensitive_body_patterns:
                body_hits += len(pattern.findall(text))
            if body_hits:
                counts["documents_with_sensitive_body_term"] += 1
                counts["sensitive_body_term_matches"] += body_hits
                reasons.add("sensitive_body_term")

            if reasons:
                counts["documents_flagged_for_review"] += 1
                flagged[doc_id] = {
                    "source": doc.get("source"),
                    "language": metadata.get("language"),
                    "processed_text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                    "review_flags": sorted(reasons),
                }

    report = {
        "kind": "slm.fsc_v1_privacy_signal_scan",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "input_documents": str(args.documents),
        "input_sha256": sha256(args.documents),
        "privacy_rules": str(args.pipeline_config),
        "privacy_rules_sha256": sha256(args.pipeline_config),
        "scan_scope": "processed FSC-v1 corpus used for EXP-0009 10M and grown-25M training; scans text and selected source metadata, not learned weights",
        "counts": {
            "documents": counts["documents"],
            "characters": chars,
            "by_source": dict(source_counts),
            "by_language": dict(language_counts),
            "residual_structured_identifier_matches": {
                key.removeprefix("residual_"): counts[key]
                for key in ("residual_email_matches", "residual_iban_matches", "residual_ip_matches", "residual_phone_matches")
            },
            "documents_with_replacement_character": counts["documents_with_replacement_character"],
            "title_pattern_hits_by_rule": {key.removeprefix("title_pattern:"): value for key, value in counts.items() if key.startswith("title_pattern:")},
            "documents_missing_discipline": counts["documents_missing_discipline"],
            "documents_excluded_discipline": counts["documents_excluded_discipline"],
            "documents_with_sensitive_body_term": counts["documents_with_sensitive_body_term"],
            "sensitive_body_term_matches": counts["sensitive_body_term_matches"],
            "documents_flagged_for_review": counts["documents_flagged_for_review"],
        },
        "limitations": [
            "Pattern scans miss names, indirect identifiers, free-text sensitive details, tables and rare combinations of facts.",
            "Body-term flags are broad triage signals and may be false positives in scientific writing.",
            "No exhaustive human review was performed; this report does not establish privacy clearance or anonymity.",
            "This scan does not measure memorization or leakage from model weights.",
        ],
        "flagged_records": flagged,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report["counts"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
