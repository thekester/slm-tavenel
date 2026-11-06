#!/usr/bin/env python3
"""Per-layer internal diagnostics for the model cards, drawn from each repository's published
``observability.jsonl`` (fixed probe batch, logged during training; the last logged record is used and
its step is printed in the title; it may precede the final checkpoint by up to one logging interval).

Panels (one line per diagnostic, x = layer): attention entropy (nats, mean over heads), attention-sink
mass (attention weight that queries after position 0 put on the first token, mean over heads), fraction
of tokens whose head-wise gate is below 0.1 (mean over heads; ``slm.interp.gate_statistics``) and
residual-stream RMS after each block. These describe the trained network on
the probe; they are observability measurements, not causal explanations of any output.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from huggingface_hub import HfApi, hf_hub_download  # noqa: E402

FILES = {
    "slm-tavenel-1m": "training_record/replicates/seed-421101/gated/observability.jsonl",
    "slm-tavenel-10m": "training_record/observability.jsonl",
    "slm-tavenel-25m": "training_record/observability.jsonl",
    "slm-tavenel-124m": "training_record/observability.jsonl",
}
PANELS = [("attn_entropy_mean", "Attention entropy (nats)"), ("sink_mass_mean", "Attention-sink mass"),
          ("gate_closed_frac", "Closed gate fraction"), ("resid_rms", "Residual RMS")]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    api, summary = HfApi(), {}
    for repo, name in FILES.items():
        sha = api.model_info(f"athenista/{repo}").sha
        path = hf_hub_download(f"athenista/{repo}", name, revision=sha, local_dir=args.cache / repo)
        records = [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]
        final = records[-1]
        fig, axes = plt.subplots(1, 4, figsize=(13, 3.0), constrained_layout=True)
        for ax, (key, title) in zip(axes, PANELS):
            values = final[key]
            ax.plot(range(len(values)), values, marker="o", ms=3, lw=1.5, color="#3b6ea5")
            ax.set_title(title, fontsize=10)
            ax.set_xlabel("layer", fontsize=9)
            ax.grid(alpha=0.3)
            ax.tick_params(labelsize=8)
        fig.suptitle(f"{repo}: internal diagnostics at step {final['step']} (fixed probe batch)", fontsize=10)
        for ext in ("png", "svg"):
            fig.savefig(args.out / f"{repo}-observability.{ext}", dpi=150)
        plt.close(fig)
        summary[repo] = {"revision": sha, "file": name, "step": final["step"], "layers": len(final["resid_rms"]),
                         **{k: [round(x, 4) for x in final[k]] for k, _ in PANELS}}
        print(repo, "step", final["step"], "layers", len(final["resid_rms"]), flush=True)
    (args.out / "observability-summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
