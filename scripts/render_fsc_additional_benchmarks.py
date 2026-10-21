"""Render the two added benchmark results without mixing them into the four-task chart."""

from __future__ import annotations

import json
import math
import os
from html import escape
from pathlib import Path


MODELS = ["slm-tavenel-1m", "tiny1m", "slm-tavenel-10m", "pythia-14m", "pythia-31m", "smollm2-135m"]
LABELS = {"slm-tavenel-1m": "SLM-Tavenel 1M", "tiny1m": "TinyStories 0.9M",
          "slm-tavenel-10m": "SLM-Tavenel 10M", "pythia-14m": "Pythia 14M",
          "pythia-31m": "Pythia 31M", "smollm2-135m": "SmolLM2 135M"}
COLORS = ["#2563eb", "#0891b2", "#059669", "#64748b", "#7c3aed", "#db2777"]
TASKS = [("arc_challenge", "ARC-Challenge", 0.25), ("winogrande", "WinoGrande XL validation", 0.5)]
BEGIN = "<!-- BEGIN FSC ADDITIONAL BENCHMARK RESULTS -->"
END = "<!-- END FSC ADDITIONAL BENCHMARK RESULTS -->"


def main() -> int:
    readme = Path("README.md")
    text = readme.read_text(encoding="utf-8")
    if text.count(BEGIN) != 1 or text.count(END) != 1 or text.index(END) < text.index(BEGIN):
        raise ValueError("README additional-results markers are missing, duplicated or reversed")
    start = text.index(BEGIN)
    finish = text.index(END, start) + len(END)
    results = {}
    benchmark_fingerprint = None
    for name in MODELS:
        path = Path("runs/fsc-additional") / name / "fsc-pilot-v2-mc.json"
        if not path.is_file():
            continue
        record = json.loads(path.read_text(encoding="utf-8"))
        if not record.get("benchmarks") or any(not x.get("sha256") for x in record["benchmarks"]):
            raise ValueError(f"Missing input hashes in {path}")
        if benchmark_fingerprint is None:
            benchmark_fingerprint = record["benchmarks"]
        elif benchmark_fingerprint != record["benchmarks"]:
            raise ValueError(f"Benchmark input mismatch in {path}")
        results[name] = record
    if len(results) < 2:
        raise FileNotFoundError("At least two evaluated models are required")

    width, height = 1120, 590
    panel_w, left, top, plot_h = 470, 75, 85, 300
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img">',
             '<title>SLM-Tavenel and reference SLM benchmark results</title>',
             '<desc>Bar colors map to full model names in the legend below. SLM-Tavenel 1M and 10M are project models; TinyStories, Pythia and SmolLM2 are external reference checkpoints. Accuracy is scored with the SLM-Tavenel local evaluator. Chance baselines are 25 and 50 percent.</desc>',
             '<rect width="100%" height="100%" fill="white"/>',
             '<text x="560" y="35" text-anchor="middle" font-family="Arial" font-size="21" font-weight="bold" fill="#172033">SLM-Tavenel and reference SLM benchmark results</text>',
             '<text x="560" y="58" text-anchor="middle" font-family="Arial" font-size="12" fill="#526078">Same local evaluator; exploratory scores; whiskers show plus or minus one standard error</text>']
    for ti, (task, label, chance) in enumerate(TASKS):
        x0 = 65 + ti * 545
        parts.append(f'<text x="{x0+panel_w/2}" y="82" text-anchor="middle" font-family="Arial" font-size="16" font-weight="bold" fill="#172033">{escape(label)}</text>')
        for tick in (0, .25, .5, .75, 1):
            yy = top + plot_h * (1-tick)
            parts.append(f'<line x1="{x0+40}" y1="{yy}" x2="{x0+panel_w}" y2="{yy}" stroke="#dce2eb"/>')
            parts.append(f'<text x="{x0+34}" y="{yy+4}" text-anchor="end" font-family="Arial" font-size="10" fill="#526078">{tick:.0%}</text>')
        cy = top + plot_h * (1-chance)
        parts.append(f'<line x1="{x0+40}" y1="{cy}" x2="{x0+panel_w}" y2="{cy}" stroke="#4b5563" stroke-dasharray="5 4"/>')
        parts.append(f'<text x="{x0+panel_w-3}" y="{cy-5}" text-anchor="end" font-family="Arial" font-size="10" fill="#4b5563">chance {chance:.0%}</text>')
        group_w = (panel_w-40) / len(results)
        bar_w = min(42, group_w*.55)
        for mi, name in enumerate(results):
            item = results[name]["results"][task]
            acc, se = item["acc"], item.get("acc_stderr", 0)
            x = x0+40+mi*group_w+(group_w-bar_w)/2
            y = top+plot_h*(1-acc)
            base = top+plot_h
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{base-y:.1f}" rx="2" fill="{COLORS[MODELS.index(name)]}"/>')
            cx = x+bar_w/2
            ya, yb = top+plot_h*(1-min(1,acc+se)), top+plot_h*(1-max(0,acc-se))
            parts.append(f'<line x1="{cx:.1f}" y1="{ya:.1f}" x2="{cx:.1f}" y2="{yb:.1f}" stroke="#172033"/><line x1="{cx-4:.1f}" y1="{ya:.1f}" x2="{cx+4:.1f}" y2="{ya:.1f}" stroke="#172033"/><line x1="{cx-4:.1f}" y1="{yb:.1f}" x2="{cx+4:.1f}" y2="{yb:.1f}" stroke="#172033"/>')
    parts.append('<text x="560" y="423" text-anchor="middle" font-family="Arial" font-size="12" font-weight="bold" fill="#334155">Model key; colors match the bars above</text>')
    for mi, name in enumerate(results):
        col, row = mi % 3, mi // 3
        x, y = 95 + col * 325, 450 + row * 25
        parts.append(f'<rect x="{x}" y="{y-12}" width="14" height="14" rx="2" fill="{COLORS[MODELS.index(name)]}"/>')
        parts.append(f'<text x="{x+22}" y="{y}" font-family="Arial" font-size="12" fill="#334155">{escape(LABELS[name])}</text>')
    parts.append('<text x="560" y="520" text-anchor="middle" font-family="Arial" font-size="11" fill="#64748b">Same scoring code does not equalize training data, token budgets, pretraining exposure, or context limits.</text>')
    parts.append('<text x="560" y="542" text-anchor="middle" font-family="Arial" font-size="11" fill="#64748b">Whiskers are plus or minus one standard error across evaluation items, conditional on each checkpoint.</text>')
    parts.append('</svg>')
    out = Path("docs/fsc-additional-benchmark-comparison.svg")

    rows = ["### Supplementary benchmark results", "",
            "ARC-Challenge uses the pinned test split; WinoGrande uses the labeled XL validation split because the public test labels are withheld.",
            "Scores are same-harness, post-hoc and exploratory. The exact-overlap audit is stored locally at `runs/fsc-additional/contamination-audit.json`; exposure in external models' pretraining remains unknown.",
            "", "| Model | Parameters | ARC-Challenge | WinoGrande XL validation |", "|---|---:|---:|---:|"]
    for name, record in results.items():
        params = record["model"]["model_params"]
        rows.append(f"| {LABELS[name]} | {params/1_000_000:.3g}M | " +
                    " | ".join(f"{record['results'][task]['acc']:.3f}" for task, _, _ in TASKS) + " |")
    rows += ["", f"![Additional SLM benchmark comparison]({os.path.relpath(out.resolve(), Path('README.md').resolve().parent).replace(os.sep, '/')})",
             "", "Error bars show ±1 standard error across evaluation items, conditional on each checkpoint. They do not measure training-seed variation.",
             "Paired item-bootstrap reports for the predeclared comparisons are stored under `runs/fsc-additional/comparisons/` and are not committed.", ""]
    text = text[:start] + BEGIN + "\n" + "\n".join(rows) + END + text[finish:]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(parts)+"\n", encoding="utf-8")
    readme.write_text(text, encoding="utf-8")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
