#!/usr/bin/env python3
"""Pre-publication checklist for the staged model cards (read-only: queries the Hub, uploads nothing).

For every ``docs/model-cards/revised-2026-11-06/<repo>/README.md``:
  1. YAML front matter parses and has language, license, library_name, pipeline_tag and model-index;
  2. every model-index accuracy equals the card's row in the shared six-task table (3 decimals);
  3. the common sections are present, in the common order;
  4. every relative link or image target exists in the published repository at its current revision
     or in the staged folder (files that a publication would add);
  5. every 40-hex revision quoted in a code block exists on the Hub for that repository;
  6. no leftover placeholder, Unicode replacement character, or '?' used in place of a minus sign;
  7. the HF model-card checklist items: usage snippet, limitations, training data, evaluation, license.
Writes a JSON report beside the cards and prints a summary; exit code 1 if any check fails.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import yaml
from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / "docs/model-cards/revised-2026-11-06"
SECTIONS = ["Contents", "Main result", "Benchmarks", "Local inference", "Use", "Architecture", "Training",
            "Traceability and explainability", "Intended use, out-of-scope use and safety",
            "Data, licenses and privacy", "Limitations", "License and citation", "The SLM-Tavenel family"]
TABLE_NAMES = {"slm-tavenel-1m": "SLM-Tavenel 1M", "slm-tavenel-10m": "SLM-Tavenel 10M",
               "slm-tavenel-25m": "SLM-Tavenel 25M", "slm-tavenel-25m-hplt-10pct": "SLM-Tavenel 25M HPLT 10%",
               "slm-tavenel-124m": "SLM-Tavenel 124M"}
ORDER = ["ARC-Easy", "ARC-Challenge", "HellaSwag", "Belebele (English)", "Belebele (French)", "WinoGrande (XL, validation)"]


def check(repo: str, card: Path, api: HfApi) -> list[str]:
    problems = []
    text = card.read_text(encoding="utf-8")
    match = re.match(r"^---\n(.*?)\n---\n", text, re.S)
    if not match:
        return ["no YAML front matter"]
    meta = yaml.safe_load(match.group(1))
    for key in ("language", "license", "library_name", "pipeline_tag", "model-index"):
        if key not in meta:
            problems.append(f"front matter lacks {key}")
    body = text[match.end():]
    row = next((l for l in body.splitlines() if l.startswith(f"| {TABLE_NAMES[repo]} |")), None)
    if row is None:
        problems.append("no row in the six-task table")
    else:
        table = [float(x) for x in row.strip("|").split("|")[2:8]]
        index = {r["dataset"]["name"]: r["metrics"][0]["value"] for r in meta["model-index"][0]["results"]}
        for name, value in zip(ORDER, table):
            if round(float(index.get(name, -1)), 3) != round(value, 3):
                problems.append(f"model-index {name}={index.get(name)} differs from table {value}")
    headings = re.findall(r"^## (.+)$", body, re.M)
    missing = [s for s in SECTIONS if s not in headings]
    if missing:
        problems.append(f"missing sections {missing}")
    elif [h for h in headings if h in SECTIONS] != SECTIONS:
        problems.append("sections out of the common order")
    info = api.model_info(f"athenista/{repo}")
    published = {s.rfilename for s in info.siblings}
    staged = {p.name for p in card.parent.iterdir()}
    for target in re.findall(r"\]\(([^)#\s]+)\)", body):
        if target.startswith(("http://", "https://")):
            continue
        if target not in published and target not in staged:
            problems.append(f"link target not found: {target}")
    for revision in set(re.findall(r'revision = "([0-9a-f]{40})"', body)):
        try:
            api.model_info(f"athenista/{repo}", revision=revision)
        except Exception as error:  # noqa: BLE001
            problems.append(f"revision {revision[:8]} not on the Hub: {type(error).__name__}")
    if "PLACEHOLDER" in text:
        problems.append("placeholder left")
    if "�" in text:
        problems.append("Unicode replacement character")
    if re.search(r"\(\?\d|\s\?\d", text):
        problems.append("'?' used in place of a minus sign")
    for needle, item in (("```python", "usage snippet"), ("## Limitations", "limitations"),
                         ("## Data, licenses and privacy", "training data"), ("## Benchmarks", "evaluation"),
                         ("## License and citation", "license")):
        if needle not in text:
            problems.append(f"HF checklist: no {item}")
    return problems


def main() -> int:
    api, report = HfApi(), {}
    for card in sorted(STAGE.glob("slm-tavenel-*/README.md")):
        repo = card.parent.name
        report[repo] = check(repo, card, api)
        print(f"{repo}: {'OK' if not report[repo] else '; '.join(report[repo])}")
    (STAGE / "check-report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 1 if any(report.values()) else 0


if __name__ == "__main__":
    sys.exit(main())
