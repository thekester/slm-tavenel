"""Summarize completed self-declared human blind forms; never invent or complete missing ratings."""
from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path

from serve_story_review import ROOT, TRAITS, write


def summarize(directory: Path, target="story-935k-100m-v2") -> dict:
    key = json.loads((directory / "key.json").read_text(encoding="utf-8"))
    forms = [json.loads(p.read_text(encoding="utf-8")) for p in directory.glob("ratings-*.json")]
    eligible = [f for f in forms if f.get("finished") and f.get("reviewer_type") == "human_self_declared"
                and set(f.get("ratings", {})) == set(key)
                and all(r.get("human") is True and r.get("blind") is True for r in f["ratings"].values())]
    report = {"kind": "slm.human_blind_pilot", "status": "awaiting_human_ratings",
              "pairs_per_form": len(key), "unique_prefixes": len({r["prompt_id"] for r in key.values()}),
              "forms_started": len(forms), "complete_blind_human_forms": len(eligible),
              "notes": ["Human identity and independence are self-declared, not independently verified",
                        "Five previously public development prefixes; not the proposed 500-prefix final study",
                        "No generated scores or unfinished forms count as completed human evidence"],
              "preference": None, "trait_means": None}
    if not eligible:
        return report
    choices, by_prefix, traits = [], {}, {}
    for form in eligible:
        for case_id, row in form["ratings"].items():
            spec = key[case_id]
            value = .5 if row["preference"] == "tie" else float(spec["models"][row["preference"]] == target)
            choices.append(value); by_prefix.setdefault(spec["prompt_id"], []).append(value)
            for label in ("A", "B"):
                model = spec["models"][label]
                for trait in TRAITS:
                    score = row["ratings"][label][trait]
                    if type(score) is not int or not 0 <= score <= 4:
                        raise ValueError("Invalid rating in completed form")
                    traits.setdefault(model, {}).setdefault(trait, []).append(score)
    prefix_means = [statistics.mean(rows) for rows in by_prefix.values()]
    rng = random.Random(421205)
    draws = sorted(statistics.mean(rng.choices(prefix_means, k=len(prefix_means))) for _ in range(10000))
    report.update(status="exploratory_blind_pilot_completed",
                  preference={"target": target, "ties": .5, "paired_ratings": len(choices),
                              "target_preference_share": statistics.mean(choices),
                              "ci95_prefix_cluster_bootstrap": [draws[249], draws[9749]],
                              "ci_limits": "Only five prefix clusters; no training-seed or independent-reviewer population uncertainty"},
                  trait_means={model: {trait: statistics.mean(values) for trait, values in scores.items()}
                               for model, scores in traits.items()})
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=ROOT / "runs/EXP-0045/human-blind-pilot-2026-11-10")
    args = parser.parse_args()
    report = summarize(args.directory)
    write(args.directory / "human-summary.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
