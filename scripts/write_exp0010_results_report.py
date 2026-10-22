"""Write the EXP-0010 public results report from recorded analysis artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs/EXP-0010-1m-paired-replication-2026-10-22"
DEFAULT_OUT = ROOT / "experiments/EXP-0010-results-2026-10-22.md"
DISPLAY = {
    "SLM-Tavenel 1M EXP-0010 gated seed 421101": "SLM-Tavenel 1M (selected candidate)",
    "SLM-Tavenel 10M FSC pilot": "SLM-Tavenel 10M FSC pilot",
    "tiny1m": "[TinyStories tiny1m](https://huggingface.co/shibatch/tiny1m)",
    "pythia-14m": "[Pythia-14M](https://huggingface.co/EleutherAI/pythia-14m)",
    "pythia-31m": "[Pythia-31M](https://huggingface.co/EleutherAI/pythia-31m)",
    "smollm2-135m": "[SmolLM2-135M](https://huggingface.co/HuggingFaceTB/SmolLM2-135M)",
}
TASKS = [
    ("arc_easy", "ARC-Easy"), ("arc_challenge", "ARC-Challenge"),
    ("hellaswag", "HellaSwag"), ("belebele_en", "Belebele EN"),
    ("belebele_fr", "Belebele FR"), ("winogrande", "WinoGrande XL"),
]


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    result = read(RUNS / "sealed-test-bpb-results.json")
    comparisons = read(RUNS / "benchmark-comparisons.json")
    data_replay = read(RUNS / "pipeline-reproduction-check.json")
    if result.get("status") != "complete" or data_replay.get("status") != "outputs_match_exactly":
        raise SystemExit("Complete BPB analysis and exact data-pipeline replay are required")

    interval = result["hierarchical_bootstrap_95_percent_interval_bpb"]
    decision = {
        "supports_predeclared_practical_gate_benefit_on_this_corpus_and_budget":
            "supports the predeclared +0.01 BPB threshold for this corpus and budget",
        "does_not_support_predeclared_practical_gate_benefit":
            "does not support the predeclared +0.01 BPB threshold",
        "inconclusive_for_predeclared_practical_gate_benefit":
            "is inconclusive against the predeclared +0.01 BPB threshold",
    }[result["decision"]]
    lines = [
        "# EXP-0010: paired 1M replication results",
        "",
        "**Status: complete.** The prospective protocol and its amendment are included with the model artifact. The amendment, issued after the first training run began and before test scoring, clarified that the schedule came from an inspected pilot and that the public source texts are operationally held out rather than secret.",
        "",
        "## Primary outcome",
        "",
        "The ten paired seeds compare a head-wise gated model with a matched ungated model. Both arms used exactly matching shared initial tensors and the same seed-indexed data order. The primary outcome is the equal-language macro-average English/French BPB on document-held-out text. The paired effect is `ungated BPB - gated BPB`; positive values favor the gate. The practical threshold was +0.01 BPB.",
        "",
        f"**Estimated effect:** {result['estimate_bpb']:+.4f} BPB; **hierarchical bootstrap 95% interval:** [{interval[0]:+.4f}, {interval[1]:+.4f}]. This {decision}.",
        "",
        "The bootstrap resampled paired training seeds and held-out documents within language (10,000 draws; seed 20261022). Per-seed values are below; full per-document hash-level scores are in `runs/EXP-0010-1m-paired-replication-2026-10-22/sealed-test-bpb-results.json`.",
        "",
        "![Observed paired-seed effects and pooled bootstrap interval](../docs/exp0010-paired-seed-effects.svg)",
        "",
        "| Seed | Gated EN | Gated FR | Gated macro | Ungated EN | Ungated FR | Ungated macro | Effect (ungated − gated) |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result["seed_results"]:
        lines.append("| {seed} | {gen:.4f} | {gfr:.4f} | {gm:.4f} | {uen:.4f} | {ufr:.4f} | {um:.4f} | {effect:+.4f} |".format(
            seed=row["seed"], gen=row["gated"]["en"], gfr=row["gated"]["fr"], gm=row["gated"]["equal_language_macro"],
            uen=row["ungated"]["en"], ufr=row["ungated"]["fr"], um=row["ungated"]["equal_language_macro"],
            effect=row["effect_ungated_minus_gated"]["equal_language_macro"]))

    lines += [
        "",
        "## Multiple-choice benchmarks",
        "",
        "The selected checkpoint was evaluated with the repository's log-likelihood evaluator on six pinned files. The exploratory SLM-Tavenel 10M checkpoint and public references use the same item files and method where scores are available; Pythia-31M has only two scored tasks. This table contains one SLM-Tavenel 1M, the selected candidate from EXP-0010. The tests are secondary, post-hoc comparisons. Their item-level paired intervals are not corrected for multiple comparisons and do not measure training-seed variation.",
        "",
        "| Model | Parameters | ARC-Easy | ARC-Challenge | HellaSwag | Belebele EN | Belebele FR | WinoGrande XL |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    models = [{"model": comparisons["candidate"]["model"], "parameters": comparisons["candidate"]["parameters"],
               "scores": comparisons["candidate"]["scores"]}] + comparisons["references"]
    for row in models:
        scores = " | ".join(f"{row['scores'][task]['acc']:.3f}" if task in row["scores"] else "not scored"
                             for task, _ in TASKS)
        lines.append(f"| {DISPLAY[row['model']]} | {row['parameters']/1_000_000:.3g}M | {scores} |")
    lines += [
        "",
        "![SLM-Tavenel EXP-0010 benchmark comparison](../docs/exp0010-benchmark-comparison.svg)",
        "",
        "The closest external parameter-count reference is TinyStories tiny1m (0.935M), trained on English stories. Pythia-14M and Pythia-31M are larger general-domain references; SmolLM2-135M is much larger. Pythia-31M was scored only on ARC-Challenge and WinoGrande XL validation; other cells are marked ‘not scored’. Differences in data, tokenizer, language mix, context, training tokens and possible benchmark exposure prevent causal or general superiority claims. Full raw outputs, model revisions, item IDs and paired intervals are in the `benchmark-comparisons.json` and `references/` training records.",
        "",
        "## Data and reproducibility",
        "",
        "![Document counts by language and split](../docs/exp0010-data-splits.svg)",
        "",
        f"The pipeline replay matched the recorded input hashes, retained-document hash (`{data_replay['documents_sha256_original']}`), decision-file hash (`{data_replay['decisions_sha256_original']}`), and processing statistics. The original preprocessing manifest still records a dirty worktree. The paired training runs themselves used clean commit `{result['code_commit']}`.",
        "",
        "Source and HAL metadata report CC0 for the selected records, but submitter authority is not independently established. Source-rights review and manual privacy attestation remain unsigned. The artifact provides source IDs, URIs, declared licenses, filters, hashes, code, configs, metrics and observability logs, and omits article text. The OSAID does not prescribe a particular legal mechanism for making parameters freely available; MIT is the instrument used here. We do not claim full OSI Open Source AI Definition conformance because the authority behind the source records and the data-information terms have not been independently established.",
        "",
        "Attention sink mass, entropy, gate values, residual statistics, loss, gradient norms and weight-update ratios are logged for observability. They are descriptive diagnostics, not causal explanations of generated behavior.",
        "",
        "![Training and development losses across the paired runs](../docs/exp0010-training-curves.svg)",
        "",
        "## Interpretation",
        "",
        "This study improves repeatability for a narrow architecture comparison on one small science-text corpus and one-pass token budget. It does not establish broad language ability, general scientific capability, absence of memorization, or legal clearance. No paper or DOI has been assigned.",
        "",
        "See the [protocol](EXP-0010-1m-paired-replication-2026-10-22.md), [amendment](EXP-0010-amendment-2026-10-22.md), [model card](../docs/model-cards/slm-tavenel-1m-exp0010-candidate.md) and [data card](../docs/model-cards/slm-tavenel-1m-exp0010-data-card.md).",
    ]
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(args.out), "effect_bpb": result["estimate_bpb"], "decision": result["decision"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
