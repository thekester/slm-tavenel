"""Render paired benchmark effects for the EXP-0012/0013 mixture pilots."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TASKS = [
    ("arc_easy", "ARC-Easy"),
    ("arc_challenge", "ARC-Challenge"),
    ("hellaswag", "HellaSwag"),
    ("belebele_en", "Belebele English"),
    ("belebele_fr", "Belebele French"),
    ("winogrande", "WinoGrande XL"),
]
COLORS = {"EXP-0012 10M": "#136f63", "EXP-0013 1M": "#3155a6"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "experiments/EXP-0012-0013-mixture-pilot-results.json")
    parser.add_argument("--out", type=Path, default=ROOT / "docs/exp0012-13-mixture-benchmark-effects.svg")
    args = parser.parse_args()
    result = json.loads(args.input.read_text(encoding="utf-8"))
    width, height = 1480, 625
    panel_width, panel_gap = 710, 28
    x_min, x_max = -0.08, 0.08
    plot_left, plot_right = 210, 655
    row_top, row_gap = 177, 54
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" role="img" aria-labelledby="title desc" viewBox="0 0 {width} {height}">',
        '<title id="title">Benchmark accuracy changes in SLM-Tavenel mixture pilots</title>',
        '<desc id="desc">Two panels show the change in raw multiple-choice accuracy after adding 20 percent MOT data, relative to the matched FSC-only arm. Each point is a paired accuracy difference and each whisker is a 95 percent bootstrap interval across benchmark items. Every interval includes zero. The two pilots each use one seed pair, so these intervals do not estimate variation across training runs.</desc>',
        '<rect width="100%" height="100%" fill="#f4f7fb"/>',
        '<text x="740" y="38" text-anchor="middle" font-family="Arial,sans-serif" font-size="25" font-weight="700" fill="#172033">Adding MOT data: paired benchmark outcomes</text>',
        '<text x="740" y="66" text-anchor="middle" font-family="Arial,sans-serif" font-size="14" fill="#526078">Mixture minus matched FSC-only control · raw accuracy difference in percentage points</text>',
    ]
    panels = [(result["studies"][0], 16), (result["studies"][1], 16 + panel_width + panel_gap)]
    for study, ox in panels:
        title = html.escape(study["study"] + " paired pilot")
        svg.append(f'<rect x="{ox}" y="96" width="{panel_width}" height="441" rx="14" fill="#fff" stroke="#dbe3ef"/>')
        svg.append(f'<text x="{ox+22}" y="128" font-family="Arial,sans-serif" font-size="18" font-weight="700" fill="{COLORS[study["study"]]}">{title}</text>')
        pl, pr = ox + plot_left, ox + plot_right
        axis_top, axis_bottom = 151, 479

        def xpos(v: float) -> float:
            return pl + (v - x_min) / (x_max - x_min) * (pr - pl)

        for tick in (-0.08, -0.04, 0, 0.04, 0.08):
            xx = xpos(tick)
            stroke = "#475569" if tick == 0 else "#e5eaf2"
            dash = ' stroke-dasharray="5 4"' if tick == 0 else ""
            svg.append(f'<line x1="{xx:.1f}" y1="{axis_top}" x2="{xx:.1f}" y2="{axis_bottom}" stroke="{stroke}"{dash}/>')
            svg.append(f'<text x="{xx:.1f}" y="{axis_bottom+20}" text-anchor="middle" font-family="Arial,sans-serif" font-size="11" fill="#64748b">{tick*100:+.0f}</text>')
        for idx, (task, label) in enumerate(TASKS):
            yy = row_top + idx * row_gap
            item = study["benchmark_outcomes"][task]
            delta = item["delta_accuracy_mixture_minus_control"]
            low, high = item["paired_item_bootstrap_95ci"]
            color = COLORS[study["study"]]
            svg.append(f'<text x="{ox+20}" y="{yy+4}" font-family="Arial,sans-serif" font-size="12" fill="#334155">{html.escape(label)}</text>')
            svg.append(f'<line x1="{xpos(low):.1f}" y1="{yy}" x2="{xpos(high):.1f}" y2="{yy}" stroke="{color}" stroke-width="3" stroke-linecap="round"/>')
            svg.append(f'<line x1="{xpos(low):.1f}" y1="{yy-5}" x2="{xpos(low):.1f}" y2="{yy+5}" stroke="{color}"/>')
            svg.append(f'<line x1="{xpos(high):.1f}" y1="{yy-5}" x2="{xpos(high):.1f}" y2="{yy+5}" stroke="{color}"/>')
            svg.append(f'<circle cx="{xpos(delta):.1f}" cy="{yy}" r="5" fill="{color}" stroke="#fff" stroke-width="1.5"/>')
            svg.append(f'<text x="{pr+14}" y="{yy+4}" font-family="Arial,sans-serif" font-size="11" fill="#172033">{delta*100:+.1f} pp</text>')
        svg.append(f'<text x="{(pl+pr)/2:.1f}" y="{axis_bottom+40}" text-anchor="middle" font-family="Arial,sans-serif" font-size="11" fill="#64748b">Accuracy change (percentage points)</text>')
        svg.append(f'<text x="{ox+20}" y="518" font-family="Arial,sans-serif" font-size="11" fill="#526078">95% paired item-bootstrap intervals; n varies by benchmark</text>')

    svg.extend([
        '<text x="740" y="574" text-anchor="middle" font-family="Arial,sans-serif" font-size="12" fill="#334155">All intervals include zero. One matched training-seed pair per size; item intervals do not measure run-to-run variation.</text>',
        '<text x="740" y="596" text-anchor="middle" font-family="Arial,sans-serif" font-size="12" fill="#526078">Secondary post-hoc outcomes; no multiple-comparison correction. No benchmark checkpoint was selected from these scores.</text>',
        '</svg>',
    ])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(svg) + "\n", encoding="utf-8")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
