"""Paired-document bootstrap for the once-scored EXP-0032 HPLT test split."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from analyze_exp0030_25m_data_diversity import load, paired  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--treatment", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20261028)
    args = parser.parse_args()
    control, treatment = load(args.control), load(args.treatment)
    if control["tokenizer_sha256"] != treatment["tokenizer_sha256"]:
        raise ValueError("Tokenizers differ; paired test comparison is invalid")
    key = "hplt_candidate_v2_test_all_documents"
    if key not in control["results"] or key not in treatment["results"]:
        raise ValueError("Both checkpoints must contain the complete pinned HPLT test split")
    results = {}
    for lang, offset in (("en", 0), ("fr", 1)):
        a = [r for r in control["results"][key]["per_document"] if r["language"] == lang]
        b = [r for r in treatment["results"][key]["per_document"] if r["language"] == lang]
        results[lang] = paired(a, b, args.draws, args.seed + offset)
    output = {
        "kind": "slm.exp0032_paired_hplt_test",
        "control_checkpoint_sha256": control["checkpoint_sha256"],
        "treatment_checkpoint_sha256": treatment["checkpoint_sha256"],
        "tokenizer_sha256": control["tokenizer_sha256"],
        "split": key,
        "interpretation": "negative treatment-minus-control BPB favors the frozen 10% recipe on the held-out HPLT test split",
        "results": results,
        "limitations": [
            "this is one fresh training seed pair; it does not estimate a population of training runs",
            "source-page rights and privacy for the HPLT-derived candidate remain unresolved",
            "the test split is scored once for the predeclared EXP-0032 comparison and must not guide further tuning",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
