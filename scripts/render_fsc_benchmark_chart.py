"""Render an SVG comparison from validated EXP-0009 multiple-choice outputs."""

from __future__ import annotations

import argparse
import json
import math
import os
from html import escape
from pathlib import Path


MODELS = [
    ("SLM-Tavenel 1M", Path("runs/fsc-scaling/1m-gated-s1234/analysis/fsc-pilot-v1-mc.json"), "#2563eb"),
    ("SLM-Tavenel 10M", Path("runs/fsc-scaling/10m-gated-s1234/analysis/fsc-pilot-v1-mc.json"), "#059669"),
    ("SLM-Tavenel 25M grown", Path("runs/fsc-scaling/25m-gated-grown-s1234/analysis/fsc-pilot-v1-mc.json"), "#7c3aed"),
    ("Pythia-14M (reference)", Path("runs/reference/pythia-14m/fsc-pilot-v1-mc.json"), "#64748b"),
    ("SmolLM2-135M (reference)", Path("runs/reference/smollm2-135m/fsc-pilot-v1-mc.json"), "#db2777"),
    ("TinyStories tiny1m, 0.9M (reference)", Path("runs/reference/tiny1m/fsc-pilot-v1-mc.json"), "#0891b2"),
]
TASKS = [
    ("arc_easy", "ARC-Easy"), ("hellaswag", "HellaSwag"),
    ("belebele_en", "Belebele EN"), ("belebele_fr", "Belebele FR"),
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="docs/fsc-scaling-benchmark-comparison.svg")
    parser.add_argument("--readme", default="README.md")
    args = parser.parse_args()
    readme = Path(args.readme)
    readme_text = readme.read_text(encoding="utf-8")
    begin, end = "<!-- BEGIN FSC SCALING BENCHMARK RESULTS -->", "<!-- END FSC SCALING BENCHMARK RESULTS -->"
    if readme_text.count(begin) != 1 or readme_text.count(end) != 1:
        raise ValueError("README result markers must each occur exactly once")
    marker_start = readme_text.index(begin)
    if readme_text.index(end) < marker_start:
        raise ValueError("README result markers are in reverse order")
    marker_finish = readme_text.index(end, marker_start) + len(end)
    outputs = {}
    fingerprint = None
    available = [(label, path, color) for label, path, color in MODELS if path.is_file()]
    if len(available) < 2:
        raise FileNotFoundError("at least two validated evaluation outputs are required")
    for label, path, color in available:
        result = json.loads(path.read_text(encoding="utf-8"))
        if not result.get("benchmarks") or any(not item.get("sha256") for item in result["benchmarks"]):
            raise ValueError(f"missing benchmark hashes in {path}")
        if fingerprint is None:
            fingerprint = result["benchmarks"]
        elif result["benchmarks"] != fingerprint:
            raise ValueError(f"benchmark file hashes differ in {path}; refusing to compare")
        for task, _ in TASKS:
            if task not in result["results"]:
                raise ValueError(f"{path} is missing {task}")
        outputs[label] = {"results": result["results"], "color": color, "model_params": result.get("model_params")}

    width, height = 1120, 600
    left, right, top, bottom = 70, 45, 86, 260
    plot_w, plot_h = width - left - right, height - top - bottom
    max_y = min(1.0, max(0.60, math.ceil(max(
        values["results"][task]["acc"] + values["results"][task].get("acc_stderr", 0)
        for values in outputs.values() for task, _ in TASKS
    ) * 10) / 10))
    y = lambda v: top + plot_h * (1 - v / max_y)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        '<title id="title">SLM-Tavenel scaling and external reference benchmarks</title>',
        '<desc id="desc">Raw accuracy on four shared multiple-choice benchmarks for the available evaluated models; whiskers show plus or minus one standard error. Chance is approximately 25 percent.</desc>',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="560" y="34" text-anchor="middle" font-family="Arial,sans-serif" font-size="22" font-weight="700" fill="#172033">SLM-Tavenel scale and SLM references</text>',
        '<text x="560" y="57" text-anchor="middle" font-family="Arial,sans-serif" font-size="13" fill="#526078">Raw accuracy · same local evaluator and pinned benchmark files · error bars: ±1 SE</text>',
    ]
    for tick in (max_y * i / 4 for i in range(5)):
        yy = y(tick)
        parts.append(f'<line x1="{left}" y1="{yy:.1f}" x2="{width-right}" y2="{yy:.1f}" stroke="#dce2eb"/>')
        parts.append(f'<text x="{left-10}" y="{yy+4:.1f}" text-anchor="end" font-family="Arial,sans-serif" font-size="12" fill="#526078">{tick:.0%}</text>')
    chance_y = y(0.25)
    parts.append(f'<line x1="{left}" y1="{chance_y:.1f}" x2="{width-right}" y2="{chance_y:.1f}" stroke="#4b5563" stroke-width="1.5" stroke-dasharray="6 5"/>')
    parts.append(f'<text x="{width-right-2}" y="{chance_y-6:.1f}" text-anchor="end" font-family="Arial,sans-serif" font-size="11" fill="#4b5563">four-choice chance · 25%</text>')

    group_w = plot_w / len(TASKS)
    bar_w = 18
    gap = 5
    total = len(available) * bar_w + (len(available) - 1) * gap
    for task_index, (task, task_label) in enumerate(TASKS):
        center = left + group_w * (task_index + 0.5)
        start = center - total / 2
        for model_index, (label, _, _) in enumerate(available):
            item = outputs[label]["results"][task]
            acc = item["acc"]
            se = item.get("acc_stderr", 0)
            xx = start + model_index * (bar_w + gap)
            yy = y(acc)
            base_y = y(0)
            parts.append(f'<rect x="{xx:.1f}" y="{yy:.1f}" width="{bar_w}" height="{base_y-yy:.1f}" rx="2" fill="{outputs[label]["color"]}"/>')
            mid = y(acc)
            cap_top, cap_bot = y(min(max_y, acc + se)), y(max(0, acc - se))
            cx = xx + bar_w / 2
            parts.append(f'<line x1="{cx:.1f}" y1="{cap_top:.1f}" x2="{cx:.1f}" y2="{cap_bot:.1f}" stroke="#172033" stroke-width="1"/><line x1="{cx-4:.1f}" y1="{cap_top:.1f}" x2="{cx+4:.1f}" y2="{cap_top:.1f}" stroke="#172033"/><line x1="{cx-4:.1f}" y1="{cap_bot:.1f}" x2="{cx+4:.1f}" y2="{cap_bot:.1f}" stroke="#172033"/>')
        parts.append(f'<text x="{center:.1f}" y="{height-bottom+30}" text-anchor="middle" font-family="Arial,sans-serif" font-size="13" font-weight="600" fill="#172033">{escape(task_label)}</text>')

    legend_y = height - 196
    legend_x0 = 95
    for i, (label, _, color) in enumerate(available):
        col = i % 3
        row = i // 3
        xx, yy = legend_x0 + col * 330, legend_y + row * 27
        parts.append(f'<rect x="{xx}" y="{yy-12}" width="14" height="14" rx="2" fill="{color}"/>')
        parts.append(f'<text x="{xx+22}" y="{yy}" font-family="Arial,sans-serif" font-size="12" fill="#334155">{escape(label)}</text>')
    parts.append('<text x="560" y="555" text-anchor="middle" font-family="Arial,sans-serif" font-size="11" fill="#64748b">Internal runs share FSC data, tokenizer and split; 25M grown inherits the 10M training budget plus its own training.</text>')
    parts.append('<text x="560" y="575" text-anchor="middle" font-family="Arial,sans-serif" font-size="11" fill="#64748b">External models have different training data, context limits and compute. Exploratory results.</text>')
    parts.append('</svg>')
    out = Path(args.out)

    table = [
        "### Same-harness multiple-choice results",
        "",
        "Raw accuracy (`acc`) on the same pinned benchmark files; values are proportions.",
        "",
        "| Model | Parameters | ARC-Easy | HellaSwag | Belebele EN | Belebele FR |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label, path, _ in available:
        params = outputs[label]["model_params"]
        parameter_label = f"{params / 1_000_000:.3g}M" if isinstance(params, (int, float)) else "not recorded"
        scores = [outputs[label]["results"][task]["acc"] for task, _ in TASKS]
        table.append(f"| {label} | {parameter_label} | " + " | ".join(f"{score:.3f}" for score in scores) + " |")
    table += [
        "",
        f"![Same-harness benchmark comparison across SLM-Tavenel and SLM references]({Path(os.path.relpath(out.resolve(), readme.resolve().parent)).as_posix()})",
        "",
        "Whiskers show ±1 standard error for each model's accuracy. Paired bootstrap reports, when generated, are stored locally under `runs/fsc-scaling/comparisons/` and are not distributed in this repository. These intervals condition on the evaluated checkpoints and do not measure training-run variation.",
        "",
    ]
    replacement = begin + "\n" + "\n".join(table) + "\n" + end
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(parts) + "\n", encoding="utf-8")
    readme.write_text(readme_text[:marker_start] + replacement + readme_text[marker_finish:], encoding="utf-8")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
