"""Assemble the completed 25M scale comparison, paired intervals, figure and report."""

from __future__ import annotations

import html
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TASKS = [
    ("arc_easy", "ARC-Easy", 0.25),
    ("arc_challenge", "ARC-Challenge", 0.25),
    ("hellaswag", "HellaSwag", 0.25),
    ("belebele_en", "Belebele English", 0.25),
    ("belebele_fr", "Belebele French", 0.25),
    ("winogrande", "WinoGrande XL", 0.50),
]
MODELS = [
    ("SLM-Tavenel 1M EXP-0010", "Tav 1M", "#2563eb", "runs/fsc-additional/slm-tavenel-1m/fsc-pilot-v2-mc.json", "exports/slm-tavenel-1m-exp0010-release-candidate-v4/training_record/selected-checkpoint-mc.json", 1_181_920),
    ("SLM-Tavenel 10M EXP-0009", "Tav 10M", "#059669", "runs/fsc-additional/slm-tavenel-10m/fsc-pilot-v2-mc.json", "runs/fsc-scaling/10m-gated-s1234/analysis/fsc-pilot-v1-mc.json", 9_978_368),
    ("SLM-Tavenel grown 25M EXP-0009", "Tav 25M", "#7c3aed", None, "runs/fsc-scaling/25m-gated-grown-s1234/analysis/fsc-pilot-v2-mc.json", 25_723_648),
    ("TinyStories tiny1m", "Tiny 0.9M", "#0891b2", "runs/fsc-additional/tiny1m/fsc-pilot-v2-mc.json", "runs/reference/tiny1m/fsc-pilot-v1-mc.json", 935_040),
    ("Pythia-14M", "Pythia 14M", "#64748b", "runs/fsc-additional/pythia-14m/fsc-pilot-v2-mc.json", "runs/reference/pythia-14m/fsc-pilot-v1-mc.json", 14_067_712),
    ("Pythia-31M", "Pythia 31M", "#475569", "runs/fsc-additional/pythia-31m/fsc-pilot-v2-mc.json", None, 30_494_720),
    ("SmolLM2-135M", "SmolLM2 135M", "#db2777", "runs/fsc-additional/smollm2-135m/fsc-pilot-v2-mc.json", "runs/reference/smollm2-135m/fsc-pilot-v1-mc.json", 134_515_008),
]


def read(rel: str) -> dict:
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))


def benchmark_hashes(record: dict) -> dict[str, str]:
    entries = record.get("benchmarks", record.get("benchmark_files", []))
    return {Path(item["path"]).stem: item["sha256"] for item in entries}


def normalize_record(supplemental: str | None, original: str | None, param_count: int) -> dict:
    if supplemental is None and original is None:
        merged = read("runs/fsc-scaling/25m-gated-grown-s1234/analysis/fsc-pilot-v2-mc.json")
    elif original is None:
        merged = read(supplemental)
    elif original.endswith("selected-checkpoint-mc.json"):
        merged = read(original)
    else:
        merged = read(original)
        if supplemental:
            extra = read(supplemental)
            merged["results"].update(extra["results"])
            merged.setdefault("benchmarks", []).extend(extra["benchmarks"])
    merged["model_params"] = param_count
    expected = read("runs/fsc-scaling/25m-gated-grown-s1234/analysis/fsc-pilot-v2-mc.json")
    expected_hashes = benchmark_hashes(expected)
    observed_hashes = benchmark_hashes(merged)
    if any(expected_hashes.get(name) != digest for name, digest in observed_hashes.items()):
        raise ValueError(f"Benchmark files for {original or '25M'} do not match the frozen 25M inputs")
    missing = {name for name, _, _ in TASKS} - set(merged["results"])
    # Pythia-31M is only scored on the supplementary ARC-Challenge/WinoGrande files.
    if supplemental and supplemental.endswith("pythia-31m/fsc-pilot-v2-mc.json"):
        missing -= {"arc_easy", "hellaswag", "belebele_en", "belebele_fr"}
    if missing:
        raise ValueError(f"{original}: missing benchmark outputs {sorted(missing)}")
    return merged


def paired_bootstrap(base: dict, treatment: dict, seed: int, draws: int = 20000) -> dict:
    if benchmark_hashes(base) != benchmark_hashes(treatment):
        raise ValueError("Paired comparison requires identical benchmark files")
    rng = np.random.default_rng(seed)
    rows = {}
    for task, _, _ in TASKS:
        if task not in base["results"] or task not in treatment["results"]:
            continue
        a = base["results"][task]["per_item"]
        b = treatment["results"][task]["per_item"]
        if [x["id"] for x in a] != [x["id"] for x in b] or [x["label"] for x in a] != [x["label"] for x in b]:
            raise ValueError(f"{task}: item order or labels differ")
        wins = sum(bool(y["correct"]) and not bool(x["correct"]) for x, y in zip(a, b))
        losses = sum(bool(x["correct"]) and not bool(y["correct"]) for x, y in zip(a, b))
        ties = len(a) - wins - losses
        n = len(a)
        sample = rng.multinomial(n, [wins / n, losses / n, ties / n], size=draws)
        deltas = (sample[:, 0] - sample[:, 1]) / n
        rows[task] = {
            "items": n,
            "baseline_acc": sum(bool(x["correct"]) for x in a) / n,
            "treatment_acc": sum(bool(x["correct"]) for x in b) / n,
            "delta_treatment_minus_baseline": (wins - losses) / n,
            "paired_item_bootstrap_95ci": [float(x) for x in np.quantile(deltas, [0.025, 0.975])],
            "treatment_only_correct": wins,
            "baseline_only_correct": losses,
        }
    return {"bootstrap_seed": seed, "bootstrap_draws": draws, "benchmarks": rows}


def svg_chart(records: dict[str, dict]) -> str:
    width, height = 1500, 940
    margin_x, margin_y, gap_x, gap_y = 76, 112, 42, 45
    panel_w, panel_h = 430, 310
    plot_left, plot_top, plot_w, plot_h = 48, 40, 360, 194
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        '<title id="title">SLM-Tavenel 1M, 10M and grown 25M benchmark comparison</title>',
        '<desc id="desc">Six panels compare raw accuracy from the repository evaluator for SLM-Tavenel and public SLM references on identical pinned benchmark files. Error bars are one standard error across questions, conditional on each checkpoint; they do not represent training-seed variability. Pythia-31M has results only for ARC-Challenge and WinoGrande.</desc>',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="750" y="35" text-anchor="middle" font-family="Arial,sans-serif" font-size="26" font-weight="700" fill="#172033">SLM-Tavenel scaling and SLM reference benchmarks</text>',
        '<text x="750" y="62" text-anchor="middle" font-family="Arial,sans-serif" font-size="14" fill="#526078">Same local evaluator and pinned item files · bars: raw accuracy · whiskers: ±1 item standard error</text>',
    ]
    for ti, (task, label, chance) in enumerate(TASKS):
        col, row = ti % 3, ti // 3
        x0, y0 = margin_x + col * (panel_w + gap_x), margin_y + row * (panel_h + gap_y)
        svg.append(f'<text x="{x0+panel_w/2}" y="{y0+15}" text-anchor="middle" font-family="Arial,sans-serif" font-size="17" font-weight="700" fill="#172033">{html.escape(label)}</text>')
        for tick in (0, .25, .5, .75, 1):
            yy = y0 + plot_top + plot_h * (1 - tick)
            svg.append(f'<line x1="{x0+plot_left}" y1="{yy:.1f}" x2="{x0+plot_left+plot_w}" y2="{yy:.1f}" stroke="#e2e8f0"/>')
            svg.append(f'<text x="{x0+plot_left-8}" y="{yy+4:.1f}" text-anchor="end" font-family="Arial,sans-serif" font-size="10" fill="#64748b">{tick:.0%}</text>')
        cy = y0 + plot_top + plot_h * (1 - chance)
        svg.append(f'<line x1="{x0+plot_left}" y1="{cy:.1f}" x2="{x0+plot_left+plot_w}" y2="{cy:.1f}" stroke="#64748b" stroke-dasharray="5 4"/>')
        available = [(m, records[m[0]]) for m in MODELS if task in records[m[0]]["results"]]
        slot = plot_w / max(1, len(available))
        bar_w = min(31, slot * .55)
        for i, (meta, record) in enumerate(available):
            _, short, color, *_ = meta
            item = record["results"][task]
            score, stderr = float(item["acc"]), float(item.get("acc_stderr", 0))
            xx = x0 + plot_left + slot * (i + .5)
            yy = y0 + plot_top + plot_h * (1 - score)
            base_y = y0 + plot_top + plot_h
            svg.append(f'<rect x="{xx-bar_w/2:.1f}" y="{yy:.1f}" width="{bar_w:.1f}" height="{base_y-yy:.1f}" rx="3" fill="{color}"/>')
            yhi = y0 + plot_top + plot_h * (1 - min(1, score + stderr))
            ylo = y0 + plot_top + plot_h * (1 - max(0, score - stderr))
            svg.append(f'<line x1="{xx:.1f}" y1="{yhi:.1f}" x2="{xx:.1f}" y2="{ylo:.1f}" stroke="#172033" stroke-width="1.2"/><line x1="{xx-5:.1f}" y1="{yhi:.1f}" x2="{xx+5:.1f}" y2="{yhi:.1f}" stroke="#172033"/><line x1="{xx-5:.1f}" y1="{ylo:.1f}" x2="{xx+5:.1f}" y2="{ylo:.1f}" stroke="#172033"/>')
            svg.append(f'<text x="{xx:.1f}" y="{base_y+20:.1f}" transform="rotate(-35 {xx:.1f} {base_y+20:.1f})" text-anchor="end" font-family="Arial,sans-serif" font-size="10" fill="#334155">{html.escape(short)}</text>')
    legend_y = 830
    for i, (name, short, color, *_rest) in enumerate(MODELS):
        col, row = i % 4, i // 4
        xx, yy = 105 + col * 350, legend_y + row * 28
        svg.append(f'<rect x="{xx}" y="{yy-12}" width="14" height="14" rx="2" fill="{color}"/><text x="{xx+22}" y="{yy}" font-family="Arial,sans-serif" font-size="12" fill="#334155">{html.escape(name)}</text>')
    svg += [
        '<text x="750" y="900" text-anchor="middle" font-family="Arial,sans-serif" font-size="12" fill="#64748b">Chance: 25% on four-choice tasks; 50% on binary WinoGrande. Different training data, tokenizers, budgets and known exposure limit cross-model interpretation.</text>',
        '</svg>',
    ]
    return "\n".join(svg) + "\n"


def svg_bpb_chart(bootstrap: dict) -> str:
    """Forest plot for paired document-bootstrap BPB differences."""
    width, height = 1100, 360
    left, right, top, row_gap = 250, 80, 112, 64
    plot_width = width - left - right
    x_max = 0.07
    x = lambda value: left + plot_width * value / x_max
    rows = [
        ("en", "English", "685 documents"),
        ("fr", "French", "678 documents"),
        ("equal_language_mean", "Equal-language mean", "EN and FR resampled separately"),
    ]
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        '<title id="title">Paired document-bootstrap BPB difference: SLM-Tavenel 10M and grown 25M</title>',
        '<desc id="desc">The horizontal axis shows 10M BPB minus grown-25M BPB. Positive differences favor the grown 25M checkpoint. Error bars are 95 percent paired document-bootstrap intervals, conditional on these checkpoints.</desc>',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="550" y="34" text-anchor="middle" font-family="Arial,sans-serif" font-size="23" font-weight="700" fill="#172033">Held-out BPB: grown 25M and 10M</text>',
        '<text x="550" y="60" text-anchor="middle" font-family="Arial,sans-serif" font-size="13" fill="#526078">10M BPB − 25M BPB; positive values favor 25M · 20,000 paired document-bootstrap draws</text>',
    ]
    for tick in (0, .01, .02, .03, .04, .05, .06, .07):
        xx = x(tick)
        svg.append(f'<line x1="{xx:.1f}" y1="{top-17}" x2="{xx:.1f}" y2="{top+row_gap*3-18}" stroke="#e2e8f0"/>')
        svg.append(f'<text x="{xx:.1f}" y="{top+row_gap*3+3}" text-anchor="middle" font-family="Arial,sans-serif" font-size="11" fill="#526078">{tick:.2f}</text>')
    for index, (key, label, detail) in enumerate(rows):
        result = bootstrap["results"][key]
        point = result["delta_bpb_10m_minus_25m"]
        lo, hi = result["ci95_percentile"]
        yy = top + index * row_gap
        svg.append(f'<text x="{left-18}" y="{yy-3}" text-anchor="end" font-family="Arial,sans-serif" font-size="14" font-weight="600" fill="#172033">{html.escape(label)}</text>')
        svg.append(f'<text x="{left-18}" y="{yy+14}" text-anchor="end" font-family="Arial,sans-serif" font-size="10" fill="#64748b">{html.escape(detail)}</text>')
        svg.append(f'<line x1="{x(lo):.1f}" y1="{yy}" x2="{x(hi):.1f}" y2="{yy}" stroke="#7c3aed" stroke-width="3"/><line x1="{x(lo):.1f}" y1="{yy-7}" x2="{x(lo):.1f}" y2="{yy+7}" stroke="#7c3aed" stroke-width="2"/><line x1="{x(hi):.1f}" y1="{yy-7}" x2="{x(hi):.1f}" y2="{yy+7}" stroke="#7c3aed" stroke-width="2"/><circle cx="{x(point):.1f}" cy="{yy}" r="6" fill="#7c3aed" stroke="#ffffff" stroke-width="2"/>')
        svg.append(f'<text x="{x(hi)+10:.1f}" y="{yy+4}" font-family="Arial,sans-serif" font-size="11" fill="#334155">{point:.4f} [{lo:.4f}, {hi:.4f}]</text>')
    svg += [
        f'<text x="{left+plot_width/2:.1f}" y="{height-16}" text-anchor="middle" font-family="Arial,sans-serif" font-size="12" fill="#334155">Difference in bits per byte</text>',
        '</svg>',
    ]
    return "\n".join(svg) + "\n"


def main() -> int:
    records = {}
    for name, _short, _color, supplemental, original, params in MODELS:
        records[name] = normalize_record(supplemental, original, params)
    selected = records["SLM-Tavenel grown 25M EXP-0009"]
    for model_name, record in records.items():
        hashes = benchmark_hashes(record)
        if any(benchmark_hashes(selected).get(name) != digest for name, digest in hashes.items()):
            raise ValueError(f"{model_name} does not use exactly the same six benchmark files")

    out_json = ROOT / "runs/fsc-scaling/comparisons/exp0009-25m-all-model-benchmarks.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    comparison = {
        "kind": "slm.exp0009_25m_same_harness_benchmark_comparison",
        "models": {
            name: {
                "parameters": record.get("model_params"),
                "model_id": record.get("hf_model") or record.get("model"),
                "model_revision": record.get("hf_revision"),
                "checkpoint": record.get("checkpoint") or record.get("evaluation_inputs", {}).get("model", {}).get("checkpoint") or record.get("evaluation_inputs", {}).get("checkpoint", {}).get("path"),
                "checkpoint_sha256": record.get("checkpoint_sha256") or record.get("evaluation_inputs", {}).get("model", {}).get("checkpoint_sha256") or record.get("evaluation_inputs", {}).get("checkpoint", {}).get("sha256"),
                "evaluation_harness": record.get("evaluation_harness"),
                "benchmark_hashes": benchmark_hashes(record),
                "results": {task: {k: v for k, v in result.items() if k != "per_item"} for task, result in record["results"].items()},
            }
            for name, record in records.items()
        },
        "benchmark_files": benchmark_hashes(selected),
        "interpretation_limits": [
            "single checkpoint per model; item-level errors do not estimate training-seed variation",
            "cross-model training data, tokenizer, compute budget, token budget and possible benchmark exposure differ",
            "25M is grown from trained 10M weights and received 491.52M additional token positions; it is not a scratch 25M run",
            "FSC training data has unresolved source-license metadata issues and unsigned rights/privacy review",
            "no multiple-comparison correction is applied to these secondary tasks",
        ],
    }
    out_json.write_text(json.dumps(comparison, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    reference_dir = ROOT / "runs/fsc-scaling/comparisons/exp0009-25m-reference-records"
    reference_dir.mkdir(parents=True, exist_ok=True)
    for name, record in records.items():
        slug = "".join(char.lower() if char.isalnum() else "-" for char in name).strip("-")
        (reference_dir / f"{slug}.json").write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    chart_path = ROOT / "docs/exp0009-25m-benchmark-comparison.svg"
    chart_path.write_text(svg_chart(records), encoding="utf-8")

    comparisons = {}
    baseline_names = ["SLM-Tavenel 1M EXP-0010", "SLM-Tavenel 10M EXP-0009"]
    for i, baseline_name in enumerate(baseline_names):
        comparisons[f"25M-vs-{baseline_name}"] = paired_bootstrap(records[baseline_name], selected, seed=20261022 + i)
    paired_path = ROOT / "runs/fsc-scaling/comparisons/exp0009-25m-paired-item-intervals.json"
    paired_path.write_text(json.dumps(comparisons, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    bpb_bootstrap = read("runs/fsc-scaling/comparisons/exp0009-bpb-document-bootstrap.json")
    bpb_chart_path = ROOT / "docs/exp0009-25m-bpb-document-bootstrap.svg"
    bpb_chart_path.write_text(svg_bpb_chart(bpb_bootstrap), encoding="utf-8")
    bpb_records = {
        scale: {
            lang: read(f"runs/fsc-scaling/{run_id}/analysis/fsc-pilot-v1-bpb-{lang}.json")
            for lang in ("en", "fr")
        }
        for scale, run_id in (("1m", "1m-gated-s1234"), ("10m", "10m-gated-s1234"), ("25m", "25m-gated-grown-s1234"))
    }

    lines = [
        "# EXP-0009 grown 25M: same-harness scaling results",
        "",
        "**Status:** completed grown 25M checkpoint and all planned evaluations; one training trajectory, seed 1234. This report compares checkpoints and benchmark/document samples. It does not estimate run-to-run training variation.",
        "",
        "The 25M model was function-preservingly grown from the completed 10M checkpoint, then trained for 15,000 additional steps (491.52M token positions). It is therefore an inherited initialization path with approximately twice the training token budget of the scratch 10M, not a parameter-count-only or iso-compute comparison.",
        "",
        "![SLM-Tavenel 1M, 10M and grown 25M benchmark comparison](../docs/exp0009-25m-benchmark-comparison.svg)",
        "",
        "Bars show raw accuracy under this repository's evaluator on identical pinned files. Whiskers show plus or minus one standard error across benchmark items for a single checkpoint; they do not show training-seed uncertainty. The dashed line marks chance (25% for four-choice tasks and 50% for WinoGrande). Pythia-31M is shown only on ARC-Challenge and WinoGrande because the other four outputs were not scored.",
        "",
        "## Raw accuracy",
        "",
        "| Model | Parameters | ARC-Easy | ARC-Challenge | HellaSwag | Belebele EN | Belebele FR | WinoGrande XL |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, record in records.items():
        vals = [f"{record['results'][task]['acc']:.3f}" if task in record["results"] else "not scored" for task, _, _ in TASKS]
        lines.append(f"| {name} | {record['model_params']:,} | " + " | ".join(vals) + " |")
    lines += [
        "",
        "## Held-out bits per byte",
        "",
        "Lower is better. The 1M, 10M and grown-25M checkpoints use the same fixed FSC English/French validation texts and tokenizer. The paired 10M-versus-25M document bootstrap resamples documents with replacement within language; each draw compares the same sampled documents for both checkpoints. The equal-language mean is the arithmetic mean of the two language-specific differences.",
        "",
        "![Paired document-bootstrap intervals for held-out English and French BPB differences](../docs/exp0009-25m-bpb-document-bootstrap.svg)",
        "",
        "*Difference is 10M BPB minus grown-25M BPB, so positive values favor 25M. Intervals are conditional on the two saved checkpoints; they do not estimate training-seed variation or uncertainty about the source corpus.*",
        "",
        "| SLM-Tavenel checkpoint | EN BPB | FR BPB |",
        "|---|---:|---:|",
        f"| 1M EXP-0010, selected | {bpb_records['1m']['en']['bits_per_byte']:.4f} | {bpb_records['1m']['fr']['bits_per_byte']:.4f} |",
        f"| 10M EXP-0009 | {bpb_records['10m']['en']['bits_per_byte']:.4f} | {bpb_records['10m']['fr']['bits_per_byte']:.4f} |",
        f"| grown 25M EXP-0009 | **{bpb_records['25m']['en']['bits_per_byte']:.4f}** | **{bpb_records['25m']['fr']['bits_per_byte']:.4f}** |",
        "",
        "| Paired comparison (10M minus 25M) | Documents | BPB difference | 95% document-bootstrap interval |",
        "|---|---:|---:|---:|",
    ]
    for key, label, docs in (("en", "English", "685"), ("fr", "French", "678"), ("equal_language_mean", "Equal-language mean", "685 EN; 678 FR")):
        result = bpb_bootstrap["results"][key]
        lo, hi = result["ci95_percentile"]
        lines.append(f"| {label} | {docs} | {result['delta_bpb_10m_minus_25m']:+.4f} | [{lo:+.4f}, {hi:+.4f}] |")
    lines += [
        "",
        f"The equal-language estimate is +{bpb_bootstrap['results']['equal_language_mean']['delta_bpb_10m_minus_25m']:.4f} BPB (95% paired document-bootstrap interval [{bpb_bootstrap['results']['equal_language_mean']['ci95_percentile'][0]:+.4f}, {bpb_bootstrap['results']['equal_language_mean']['ci95_percentile'][1]:+.4f}]). This interval covers validation-document sampling for these checkpoints only; there is one seed per scale and no estimate of training-run variability. The bootstrap uses seed {bpb_bootstrap['uncertainty']['seed']} and {bpb_bootstrap['uncertainty']['draws']:,} draws. Detailed scoring provenance and checkpoint/data hashes are in `runs/fsc-scaling/comparisons/exp0009-bpb-document-bootstrap.json`.",
        "",
        "## Paired item-bootstrap intervals",
        "",
        "The estimates below compare per-item correctness for matched questions. The 95% intervals resample benchmark items only; they are conditional on these checkpoints and do not represent new training runs. The tasks are post-hoc secondary outcomes; no multiple-comparison correction is applied.",
        "",
        "| Comparison (25M minus baseline) | Task | Accuracy difference | 95% paired item interval |",
        "|---|---|---:|---:|",
    ]
    for label, report in comparisons.items():
        baseline_name = label.removeprefix("25M-vs-")
        for task, _, _ in TASKS:
            if task not in report["benchmarks"]:
                continue
            item = report["benchmarks"][task]
            lo, hi = item["paired_item_bootstrap_95ci"]
            lines.append(f"| 25M vs {baseline_name} | {task} | {item['delta_treatment_minus_baseline']:+.3f} | [{lo:+.3f}, {hi:+.3f}] |")
    lines += [
        "",
        "## Interpretation and provenance limits",
        "",
        "These scores are exploratory comparisons, not a ranking or evidence of general superiority. Training corpora, tokenizers, token and compute budgets, context limits, and pretraining benchmark exposure differ across references. The external reference runs use this evaluator and these item files where scored; they are not controlled training comparisons. Exact-overlap checks for ARC-Challenge and WinoGrande are post-hoc and cannot rule out paraphrase, translation or prior exposure.",
        "",
        "The 10M and grown 25M use the same exploratory mixed-license FSC corpus. A post-training HAL query found matching current main/document-file license metadata for 31,262 of 31,417 retained IDs; 72 had explicit mismatches, 3 conflicting declarations, 24 missing license metadata, and 56 were not found. This audit does not prove rights-holder authority or historical license status. Human source-rights and privacy review remain unsigned. See the [dated audit](../docs/reviews/EXP-0009-HAL-FILE-LICENSE-AUDIT-2026-10-22.md) and [resume diagnostic](../docs/reviews/EXP-0009-25M-RESUME-DIAGNOSTIC-2026-10-22.md). The full input hashes, per-item scores and comparison JSON remain in `runs/fsc-scaling/`.",
        "",
    ]
    report_path = ROOT / "experiments/EXP-0009-25m-scaling-results-2026-10-22.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    candidate_report = ROOT / "exports/slm-tavenel-25m-exp0009-candidate/training_record/EXP-0009-25m-scaling-results-2026-10-22.md"
    if candidate_report.parent.exists():
        package_text = "\n".join(lines).replace(
            "runs/fsc-scaling/comparisons/exp0009-bpb-document-bootstrap.json",
            "training_record/exp0009-bpb-document-bootstrap.json",
        )
        candidate_report.write_text(package_text, encoding="utf-8")
    print(json.dumps({"comparison": out_json.relative_to(ROOT).as_posix(), "chart": chart_path.relative_to(ROOT).as_posix(), "bpb_chart": bpb_chart_path.relative_to(ROOT).as_posix(), "report": report_path.relative_to(ROOT).as_posix(), "paired": paired_path.relative_to(ROOT).as_posix()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
