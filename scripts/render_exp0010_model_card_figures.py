"""Render data-split, training-dynamics and paired-seed figures for EXP-0010."""

from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs/EXP-0010-1m-paired-replication-2026-10-22"
SPLITS = ROOT / "data/processed/fsc-cc0-open-weights-v4/splits/split_manifest.json"
COLORS = {"gated": "#155eef", "ungated": "#d97706"}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def quantile(values: list[float], q: float) -> float:
    values = sorted(values)
    pos = (len(values) - 1) * q
    low = math.floor(pos)
    high = math.ceil(pos)
    if low == high:
        return values[low]
    return values[low] * (high - pos) + values[high] * (pos - low)


def svg_text(x: float, y: float, value: str, *, size: int = 12, fill: str = "#334155",
             anchor: str = "start", weight: str = "400") -> str:
    return (f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" '
            f'font-family="Arial,sans-serif" font-size="{size}" font-weight="{weight}" '
            f'fill="{fill}">{html.escape(value)}</text>')


def render_data_split(out: Path) -> None:
    manifest = read_json(SPLITS)
    counts = {split: {"en": 0, "fr": 0} for split in ("train", "dev", "test")}
    for row in manifest["splits"]:
        counts[row["split"]][row["language"]] = row["documents"]
    width, height = 1200, 370
    left, bar_w = 210, 780
    top, gap = 132, 75
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        '<title id="title">SLM-Tavenel EXP-0010 document split by language</title>',
        '<desc id="desc">Document counts in the training, development and held-out test splits, divided into English and French. Splits are document-disjoint.</desc>',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        svg_text(width / 2, 38, "A bilingual corpus split by complete documents", size=22, fill="#172033", anchor="middle", weight="700"),
        svg_text(width / 2, 65, "Document counts by split; bar widths show the language share within each split", size=13, fill="#526078", anchor="middle"),
    ]
    for i, split in enumerate(("train", "dev", "test")):
        y = top + i * gap
        en, fr = counts[split]["en"], counts[split]["fr"]
        total = en + fr
        en_w = bar_w * en / total
        fr_w = bar_w - en_w
        parts.append(svg_text(50, y + 26, split.capitalize(), size=16, fill="#172033", weight="700"))
        parts.append(f'<rect x="{left}" y="{y}" width="{en_w:.1f}" height="42" rx="6" fill="#155eef"/>')
        parts.append(f'<rect x="{left+en_w:.1f}" y="{y}" width="{fr_w:.1f}" height="42" rx="6" fill="#d97706"/>')
        parts.append(svg_text(left + en_w / 2, y + 26, f"English  {en:,} ({en/total:.1%})", size=13, fill="#ffffff", anchor="middle", weight="700"))
        parts.append(svg_text(left + en_w + fr_w / 2, y + 26, f"French  {fr:,} ({fr/total:.1%})", size=13, fill="#ffffff", anchor="middle", weight="700"))
        parts.append(svg_text(left + bar_w + 24, y + 26, f"{total:,} documents", size=13, fill="#334155"))
    parts += [
        svg_text(width / 2, 365, " / ".join(f"English {counts[split]['en']} {split}" for split in ("train", "dev", "test")) + " ; " + " / ".join(f"French {counts[split]['fr']} {split}" for split in ("train", "dev", "test")), size=12, fill="#64748b", anchor="middle"),
        '</svg>',
    ]
    out.write_text("\n".join(parts) + "\n", encoding="utf-8")


def render_paired_effects(out: Path) -> None:
    result = read_json(RUNS / "sealed-test-bpb-results.json")
    estimate = result["estimate_bpb"]
    low, high = result["hierarchical_bootstrap_95_percent_interval_bpb"]
    width, height = 1200, 500
    plot_left, plot_right, plot_top = 300, 1000, 116
    row_h = 27
    min_x, max_x = -0.02, 0.06

    def x(value: float) -> float:
        return plot_left + (value - min_x) / (max_x - min_x) * (plot_right - plot_left)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        '<title id="title">Paired gate effects across ten EXP-0010 seeds</title>',
        '<desc id="desc">Each seed shows its observed ungated minus gated English/French macro bits-per-byte difference; positive values favor the gated model. The final row shows the mean effect and the predeclared hierarchical bootstrap 95 percent interval.</desc>',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        svg_text(width / 2, 36, "Ten paired seeds support a positive gate effect on this corpus", size=21, fill="#172033", anchor="middle", weight="700"),
        svg_text(width / 2, 62, "Per-seed points are observed effects; only the pooled estimate has a bootstrap interval", size=13, fill="#526078", anchor="middle"),
    ]
    for tick in (-0.02, 0.00, 0.02, 0.04, 0.06):
        xx = x(tick)
        parts.append(f'<line x1="{xx:.1f}" y1="{plot_top-16}" x2="{xx:.1f}" y2="{plot_top+11*row_h}" stroke="{"#64748b" if tick == 0 else "#e5eaf2"}" stroke-dasharray="{"5 4" if tick == 0 else "0"}"/>')
        parts.append(svg_text(xx, plot_top - 28, f"{tick:+.2f}", size=10, fill="#64748b", anchor="middle"))
    for i, row in enumerate(result["seed_results"]):
        yy = plot_top + i * row_h
        val = row["effect_ungated_minus_gated"]["equal_language_macro"]
        parts.append(svg_text(260, yy + 4, f"Seed {row['seed']}", size=11, fill="#475569", anchor="end"))
        parts.append(f'<circle cx="{x(val):.1f}" cy="{yy:.1f}" r="5" fill="{"#155eef" if val >= 0 else "#64748b"}"/>')
        parts.append(svg_text(x(val) + (9 if val >= 0 else -9), yy + 4, f"{val:+.4f}", size=10, fill="#334155", anchor="start" if val >= 0 else "end"))
    pooled_y = plot_top + 10 * row_h + 17
    parts.append(f'<line x1="{plot_left-12}" y1="{pooled_y-17}" x2="{plot_right+10}" y2="{pooled_y-17}" stroke="#cbd5e1"/>')
    parts.append(svg_text(260, pooled_y + 4, "Pooled estimate", size=12, fill="#172033", anchor="end", weight="700"))
    parts.append(f'<line x1="{x(low):.1f}" y1="{pooled_y}" x2="{x(high):.1f}" y2="{pooled_y}" stroke="#155eef" stroke-width="4"/>')
    parts.append(f'<line x1="{x(low):.1f}" y1="{pooled_y-7}" x2="{x(low):.1f}" y2="{pooled_y+7}" stroke="#155eef" stroke-width="2"/>')
    parts.append(f'<line x1="{x(high):.1f}" y1="{pooled_y-7}" x2="{x(high):.1f}" y2="{pooled_y+7}" stroke="#155eef" stroke-width="2"/>')
    parts.append(f'<circle cx="{x(estimate):.1f}" cy="{pooled_y}" r="6" fill="#155eef" stroke="#ffffff" stroke-width="2"/>')
    parts.append(svg_text(x(high) + 12, pooled_y + 4, f"{estimate:+.4f} [{low:+.4f}, {high:+.4f}]", size=11, fill="#172033", weight="700"))
    parts.append(svg_text(width / 2, height - 20, "Effect = ungated BPB − gated BPB; positive values favor the gate. Pooled 95% interval resamples seeds and documents within language.", size=11, fill="#64748b", anchor="middle"))
    parts.append('</svg>')
    out.write_text("\n".join(parts) + "\n", encoding="utf-8")


def render_training_curves(out: Path) -> None:
    series = {arm: {"train": {}, "dev": {}} for arm in COLORS}
    for seed in range(421101, 421111):
        for arm in COLORS:
            base = RUNS / f"seed-{seed}/{arm}"
            for line in (base / "metrics.jsonl").read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                series[arm]["train"].setdefault(row["tokens"], []).append(float(row["loss"]))
            for line in (base / "eval.jsonl").read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                series[arm]["dev"].setdefault(row["step"] * 32768, []).append(float(row["val_loss"]))

    width, height = 1460, 620
    panel_w, panel_h = 690, 460
    panels = [(30, 102, "Training batch loss"), (740, 102, "Development-set loss")]
    total_tokens = 15_990_784
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        '<title id="title">Training and development loss across ten paired seeds</title>',
        '<desc id="desc">Median gated and ungated loss across ten training seeds. Shaded areas show the interquartile range. Development evaluation uses the same development split for all runs.</desc>',
        '<rect width="100%" height="100%" fill="#f8fafc"/>',
        svg_text(width / 2, 36, "Training traces across the ten paired runs", size=22, fill="#172033", anchor="middle", weight="700"),
        svg_text(width / 2, 62, "Lines show medians; shaded bands show the interquartile range across seeds", size=13, fill="#526078", anchor="middle"),
    ]
    for px, py, title in panels:
        parts.append(f'<rect x="{px}" y="{py}" width="{panel_w}" height="{panel_h}" rx="12" fill="#ffffff" stroke="#dbe3ef"/>')
        parts.append(svg_text(px + panel_w / 2, py + 27, title, size=16, fill="#172033", anchor="middle", weight="700"))
        left, right, top, bottom = px + 68, px + panel_w - 20, py + 62, py + panel_h - 54
        kind = "train" if title.startswith("Training") else "dev"
        observed = [value for arm in COLORS for values in series[arm][kind].values() for value in values]
        ymin = math.floor(min(observed) * 10) / 10 - 0.1
        ymax = math.ceil(max(observed) * 10) / 10 + 0.1

        def xp(token_count: int) -> float:
            return left + token_count / total_tokens * (right - left)

        def yp(value: float) -> float:
            return bottom - (value - ymin) / (ymax - ymin) * (bottom - top)

        for tick_i in range(5):
            value = ymin + (ymax - ymin) * tick_i / 4
            yy = yp(value)
            parts.append(f'<line x1="{left}" y1="{yy:.1f}" x2="{right}" y2="{yy:.1f}" stroke="#e5eaf2"/>')
            parts.append(svg_text(left - 10, yy + 4, f"{value:.1f}", size=10, fill="#64748b", anchor="end"))
        for tick in (0, 4_000_000, 8_000_000, 12_000_000, 16_000_000):
            xx = xp(tick)
            parts.append(f'<line x1="{xx:.1f}" y1="{top}" x2="{xx:.1f}" y2="{bottom}" stroke="#f1f5f9"/>')
            parts.append(svg_text(xx, bottom + 20, f"{tick/1e6:.0f}M", size=10, fill="#64748b", anchor="middle"))
        for arm, color in COLORS.items():
            entries = sorted(series[arm][kind].items())
            if kind == "train":
                # Training logs may contain a different final token position only if a run was cut short.
                entries = [(token, vals) for token, vals in entries if len(vals) == 10]
            stats = [(token, quantile(values, 0.25), quantile(values, 0.5), quantile(values, 0.75))
                     for token, values in entries]
            if not stats:
                continue
            upper = " ".join(("M" if i == 0 else "L") + f" {xp(token):.1f} {yp(q75):.1f}" for i, (token, _q25, _med, q75) in enumerate(stats))
            lower = " ".join("L" + f" {xp(token):.1f} {yp(q25):.1f}" for token, q25, _med, _q75 in reversed(stats))
            parts.append(f'<path d="{upper} {lower} Z" fill="{color}" fill-opacity="0.14" stroke="none"/>')
            path = " ".join(("M" if i == 0 else "L") + f" {xp(token):.1f} {yp(med):.1f}" for i, (token, _q25, med, _q75) in enumerate(stats))
            parts.append(f'<path d="{path}" fill="none" stroke="{color}" stroke-width="2.5"/>')
        parts.append(svg_text((left + right) / 2, bottom + 42, "Training token positions presented", size=10, fill="#64748b", anchor="middle"))
    for i, (arm, color) in enumerate(COLORS.items()):
        x0 = 580 + i * 190
        parts.append(f'<line x1="{x0}" y1="590" x2="{x0+30}" y2="590" stroke="{color}" stroke-width="3"/>')
        parts.append(svg_text(x0 + 40, 594, "Gated" if arm == "gated" else "Ungated control", size=12, fill="#334155"))
    parts.append(svg_text(width / 2, 614, "Training loss is logged on minibatches; development loss is evaluated at steps 122, 244, 366 and 488. Curves describe these runs, not uncertainty about future runs.", size=10, fill="#64748b", anchor="middle"))
    parts.append('</svg>')
    out.write_text("\n".join(parts) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "docs")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    render_data_split(args.out_dir / "exp0010-data-splits.svg")
    render_training_curves(args.out_dir / "exp0010-training-curves.svg")
    render_paired_effects(args.out_dir / "exp0010-paired-seed-effects.svg")
    print("Rendered EXP-0010 data split, training curve and paired-effect figures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
