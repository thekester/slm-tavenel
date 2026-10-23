"""Render the paired 0/10/20% wiki-mixture BPB dose response as an SVG."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/exp0017-wiki-dose-response.svg"


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    arm10 = read(ROOT / "runs/EXP-0017-25m-wiki-10pct-2026-10-23/analysis-vs-fsc.json")
    arm20 = read(ROOT / "runs/EXP-0016-25m-wiki-mixture-2026-10-23/analysis.json")
    b10 = arm10["text_bpb_vs_fsc_control"]
    b20 = arm20["text_bpb_vs_fsc_control"]

    width, height = 1120, 650
    top, bottom = 154, 535
    ymin, ymax = -0.205, 0.012
    ratios = [0, 10, 20]
    panels = [
        ("English held-out text", "en", 80),
        ("French held-out text", "fr", 585),
    ]
    series = [
        ("FSC validation", "fsc", "#2563eb", "circle"),
        ("Wikipedia holdout", "wiki", "#c2410c", "square"),
    ]

    def y(value: float) -> float:
        return bottom - (value - ymin) * (bottom - top) / (ymax - ymin)

    def x(panel_x: float, ratio: float) -> float:
        return panel_x + 75 + ratio / 20 * 320

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:Inter,Segoe UI,Arial,sans-serif;fill:#18212f}.title{font-size:27px;font-weight:700}.sub{font-size:14px;fill:#485568}.panel{font-size:18px;font-weight:700}.axis{font-size:12px;fill:#536174}.small{font-size:12px;fill:#475569}</style>',
        '<text x="70" y="47" class="title">SLM-Tavenel 25M: wiki-mixture dose response</text>',
        '<text x="70" y="76" class="sub">BPB change relative to the matched FSC-only continuation; lower is better. Points show estimates; bars show 95% paired document intervals.</text>',
    ]

    # Legend
    for j, (label, _, color, shape) in enumerate(series):
        lx = 715 + j * 190
        if shape == "circle":
            parts.append(f'<circle cx="{lx}" cy="111" r="6" fill="{color}"/>')
        else:
            parts.append(f'<rect x="{lx-6}" y="105" width="12" height="12" fill="{color}"/>')
        parts.append(f'<text x="{lx+13}" y="116" class="small">{label}</text>')

    ticks = [0.0, -0.05, -0.10, -0.15, -0.20]
    for title, lang, panel_x in panels:
        parts.append(f'<text x="{panel_x+75}" y="137" class="panel">{title}</text>')
        for tick in ticks:
            yy = y(tick)
            parts.append(f'<line x1="{panel_x+55}" y1="{yy:.1f}" x2="{panel_x+430}" y2="{yy:.1f}" stroke="{"#8b95a5" if tick == 0 else "#e3e8ef"}" stroke-width="{"1.6" if tick == 0 else "1"}"/>')
            parts.append(f'<text x="{panel_x+47}" y="{yy+4:.1f}" text-anchor="end" class="axis">{tick:.2f}</text>')
        parts.append(f'<line x1="{panel_x+55}" y1="{top}" x2="{panel_x+55}" y2="{bottom}" stroke="#637083"/>')
        parts.append(f'<line x1="{panel_x+55}" y1="{bottom}" x2="{panel_x+430}" y2="{bottom}" stroke="#637083"/>')
        for ratio in ratios:
            xx = x(panel_x, ratio)
            parts.append(f'<line x1="{xx:.1f}" y1="{bottom}" x2="{xx:.1f}" y2="{bottom+5}" stroke="#637083"/>')
            parts.append(f'<text x="{xx:.1f}" y="{bottom+23}" text-anchor="middle" class="axis">{ratio}%</text>')

        for label, key, color, marker in series:
            names = (f"fsc_{lang}_validation", f"wiki_{lang}_bpb_holdout")
            points = [(0, 0.0, [0.0, 0.0])]
            for ratio, results in ((10, b10), (20, b20)):
                row = results[names[0] if key == "fsc" else names[1]]
                points.append((ratio, row["delta_treatment_minus_control_bpb"],
                               row["paired_document_bootstrap_95ci"]))
            coords = " ".join(f'{x(panel_x, r):.1f},{y(v):.1f}' for r, v, _ in points)
            parts.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="2.5"/>')
            for ratio, value, interval in points:
                xx, yy = x(panel_x, ratio), y(value)
                low, high = y(interval[0]), y(interval[1])
                parts.append(f'<line x1="{xx:.1f}" y1="{low:.1f}" x2="{xx:.1f}" y2="{high:.1f}" stroke="{color}" stroke-width="2"/>')
                parts.append(f'<line x1="{xx-5:.1f}" y1="{low:.1f}" x2="{xx+5:.1f}" y2="{low:.1f}" stroke="{color}"/>')
                parts.append(f'<line x1="{xx-5:.1f}" y1="{high:.1f}" x2="{xx+5:.1f}" y2="{high:.1f}" stroke="{color}"/>')
                if marker == "circle":
                    parts.append(f'<circle cx="{xx:.1f}" cy="{yy:.1f}" r="5.5" fill="{color}"/>')
                else:
                    parts.append(f'<rect x="{xx-5:.1f}" y="{yy-5:.1f}" width="10" height="10" fill="{color}"/>')

    parts.extend([
        '<text x="560" y="593" text-anchor="middle" class="axis">Wikipedia fraction in available training stream</text>',
        '<text x="26" y="345" transform="rotate(-90 26 345)" text-anchor="middle" class="axis">Δ bits per byte vs FSC-only</text>',
        '<text x="70" y="630" class="small">Intervals describe held-out document sampling conditional on these checkpoints; they do not include seed uncertainty. The 10% and 20% runs share source-selection rules.</text>',
        '</svg>',
    ])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(parts) + "\n", encoding="utf-8")
    print(f"Wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
