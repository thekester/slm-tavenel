"""Render a readable, task-by-task comparison with binomial confidence intervals."""

from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "runs/EXP-0010-1m-paired-replication-2026-10-22/benchmark-comparisons.json"
OUTPUT = ROOT / "docs/exp0010-benchmark-comparison.svg"
TASKS = [
    ("arc_easy", "ARC-Easy", 0.25),
    ("arc_challenge", "ARC-Challenge", 0.25),
    ("hellaswag", "HellaSwag", 0.25),
    ("belebele_en", "Belebele English", 0.25),
    ("belebele_fr", "Belebele French", 0.25),
    ("winogrande", "WinoGrande XL validation", 0.50),
]
MODELS = [
    ("SLM-Tavenel 1M (selected)", "#155eef"),
    ("SLM-Tavenel 10M (exploratory)", "#16a34a"),
    ("TinyStories tiny1m (0.935M)", "#d97706"),
    ("Pythia-14M", "#7c3aed"),
    ("Pythia-31M", "#9333ea"),
    ("SmolLM2-135M", "#db2777"),
]
ALIASES = {
    "SLM-Tavenel 1M (selected)": "SLM-Tavenel 1M EXP-0010 gated seed 421101",
    "SLM-Tavenel 10M (exploratory)": "SLM-Tavenel 10M FSC pilot",
    "TinyStories tiny1m (0.935M)": "tiny1m",
    "Pythia-14M": "pythia-14m", "Pythia-31M": "pythia-31m",
    "SmolLM2-135M": "smollm2-135m",
}


def wilson_interval(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return center - half, center + half


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=INPUT)
    parser.add_argument("--out", type=Path, default=OUTPUT)
    args = parser.parse_args()
    data = json.loads(args.input.read_text(encoding="utf-8"))
    if data.get("status") != "exploratory_posthoc":
        raise ValueError("Expected completed descriptive benchmark comparisons")
    models = [{"model": data["candidate"]["model"], "scores": data["candidate"]["scores"]}]
    models.extend(data["references"])
    by_name = {model["model"]: model for model in models}

    width, height = 1480, 1090
    margin, col_gap, panel_w, panel_h, row_gap = 26, 20, 704, 296, 18
    plot_left_offset, plot_right_offset = 246, 652
    plot_width = plot_right_offset - plot_left_offset
    plot_top_offset, line_height = 70, 25
    max_x = 0.70
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        '<title id="title">SLM-Tavenel EXP-0010 and small-model benchmark comparison</title>',
        '<desc id="desc">Six separate benchmark panels compare one selected SLM-Tavenel 1 million parameter checkpoint with an exploratory 10 million parameter checkpoint and public references. Each dot is raw accuracy; whiskers are 95 percent Wilson intervals across items and do not estimate run-to-run variation. Pythia-31M was scored only on ARC-Challenge and WinoGrande.</desc>',
        '<rect width="100%" height="100%" fill="#f8fafc"/>',
        '<text x="740" y="34" text-anchor="middle" font-family="Arial,sans-serif" font-size="26" font-weight="700" fill="#172033">SLM-Tavenel 1M benchmark comparison</text>',
        '<text x="740" y="60" text-anchor="middle" font-family="Arial,sans-serif" font-size="14" fill="#526078">Same local evaluator and pinned items where scored · raw accuracy with 95% Wilson intervals</text>',
    ]
    for idx, (task, label, chance) in enumerate(TASKS):
        row, col = divmod(idx, 2)
        ox, oy = margin + col * (panel_w + col_gap), 82 + row * (panel_h + row_gap)
        pl, pr = ox + plot_left_offset, ox + plot_right_offset
        pt = oy + plot_top_offset
        axis_y = pt + len(MODELS) * line_height + 8
        svg.append(f'<rect x="{ox}" y="{oy}" width="{panel_w}" height="{panel_h}" rx="12" fill="#ffffff" stroke="#dbe3ef"/>')
        svg.append(f'<text x="{ox+18}" y="{oy+27}" font-family="Arial,sans-serif" font-size="17" font-weight="700" fill="#172033">{html.escape(label)}</text>')
        first = by_name[ALIASES[MODELS[0][0]]]["scores"][task]
        svg.append(f'<text x="{ox+panel_w-16}" y="{oy+26}" text-anchor="end" font-family="Arial,sans-serif" font-size="11" fill="#526078">n = {first["items"]:,} · chance {chance:.0%}</text>')

        def x(value: float) -> float:
            return pl + value / max_x * plot_width

        for tick in (0.0, 0.25, 0.50, 0.70):
            xx = x(tick)
            svg.append(f'<line x1="{xx:.1f}" y1="{pt-5}" x2="{xx:.1f}" y2="{axis_y}" stroke="#e5eaf2"/>')
            svg.append(f'<text x="{xx:.1f}" y="{axis_y+19}" text-anchor="middle" font-family="Arial,sans-serif" font-size="10" fill="#64748b">{tick:.0%}</text>')
        chance_x = x(chance)
        svg.append(f'<line x1="{chance_x:.1f}" y1="{pt-5}" x2="{chance_x:.1f}" y2="{axis_y}" stroke="#64748b" stroke-width="1.5" stroke-dasharray="5 4"/>')

        for model_idx, (model_name, color) in enumerate(MODELS):
            yy = pt + (model_idx + 0.5) * line_height
            score = by_name[ALIASES[model_name]]["scores"].get(task)
            selected = model_idx == 0
            label_color = color if selected else "#334155"
            weight = "700" if selected else "400"
            svg.append(f'<text x="{ox+16}" y="{yy+4:.1f}" font-family="Arial,sans-serif" font-size="11" font-weight="{weight}" fill="{label_color}">{html.escape(model_name)}</text>')
            if score is None:
                svg.append(f'<text x="{pl+8}" y="{yy+4:.1f}" font-family="Arial,sans-serif" font-size="10" fill="#94a3b8">not scored</text>')
                continue
            accuracy, n = score["acc"], score["items"]
            lower, upper = wilson_interval(accuracy, n)
            left, right, point = x(lower), x(upper), x(accuracy)
            svg.append(f'<line x1="{left:.1f}" y1="{yy:.1f}" x2="{right:.1f}" y2="{yy:.1f}" stroke="{color}" stroke-width="2"/>')
            svg.append(f'<line x1="{left:.1f}" y1="{yy-4:.1f}" x2="{left:.1f}" y2="{yy+4:.1f}" stroke="{color}"/>')
            svg.append(f'<line x1="{right:.1f}" y1="{yy-4:.1f}" x2="{right:.1f}" y2="{yy+4:.1f}" stroke="{color}"/>')
            radius = 5 if selected else 4
            svg.append(f'<circle cx="{point:.1f}" cy="{yy:.1f}" r="{radius}" fill="{color}" stroke="#ffffff" stroke-width="1.5"/>')
            anchor = "end" if point > pr - 42 else "start"
            dx = -8 if anchor == "end" else 8
            svg.append(f'<text x="{point+dx:.1f}" y="{yy+3.5:.1f}" text-anchor="{anchor}" font-family="Arial,sans-serif" font-size="10" font-weight="{weight}" fill="#172033">{accuracy:.1%}</text>')
        svg.append(f'<text x="{(pl+pr)/2:.1f}" y="{axis_y+38}" text-anchor="middle" font-family="Arial,sans-serif" font-size="10" fill="#64748b">Raw accuracy (dashed line = four-choice or binary chance)</text>')

    footer_y = height - 36
    svg.append(f'<text x="740" y="{footer_y}" text-anchor="middle" font-family="Arial,sans-serif" font-size="12" fill="#526078">Intervals are conditional on benchmark items; they do not measure training-seed variability. Cross-model gaps do not control data, tokenizers or compute.</text>')
    svg.append('</svg>')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(svg) + "\n", encoding="utf-8")
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
