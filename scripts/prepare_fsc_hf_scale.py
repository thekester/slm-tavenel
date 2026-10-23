"""Prepare traceable Hugging Face artifacts for the EXP-0009 FSC scaling runs.

The 10M artifact is a completed single-seed research release. The completed
25M artifact is prepared as a local candidate and is not uploaded.
Source texts and benchmark prompts are never copied into either artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import save_file
from tokenizers import Tokenizer


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
TOKENIZER = DATA / "tokenizer/fsc-pilot-v1-train-only/tokenizer.json"
PROCESSED = DATA / "processed/fsc-pilot-v1"
TOKEN_BALANCE = DATA / "processed/fsc-pilot-v1-token-balanced/final_token_balance_manifest.json"
RUNS = ROOT / "runs/fsc-scaling"
RUN_INFO = {
    "10m": {
        "run_id": "10m-gated-s1234",
        "checkpoint": "step_00015000.pt",
        "repo_id": "athenista/slm-tavenel-10m-exp0009",
        "out": ROOT / "exports/slm-tavenel-10m-exp0009",
    },
    "25m": {
        "run_id": "25m-gated-grown-s1234",
        "checkpoint": "step_00015000.pt",
        "repo_id": "athenista/slm-tavenel-25m-exp0009",
        "out": ROOT / "exports/slm-tavenel-25m-exp0009-candidate",
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def copy_file(source: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, dest)


def svg_line_chart(metrics: list[dict], evaluations: list[dict], *, preview: bool, scale: str) -> str:
    width, height = 1120, 430
    left, right, top, bottom = 78, 36, 68, 76
    max_step = max(1, max(x["step"] for x in metrics))
    values = [float(x["loss"]) for x in metrics]
    values += [float(x["val_loss"]) for x in evaluations]
    low, high = min(values) - 0.05, max(values) + 0.05
    if high <= low:
        high = low + 1
    x = lambda step: left + (width - left - right) * step / max_step
    y = lambda value: top + (height - top - bottom) * (high - value) / (high - low)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-labelledby="t d">',
        '<title id="t">SLM-Tavenel training and validation loss</title>',
        f'<desc id="d">Training loss and validation loss against completed optimizer steps. {"This is an incomplete 25M preview through its durable checkpoint." if preview else "The plotted run is complete."}</desc>',
        '<rect width="100%" height="100%" fill="#fff"/>',
        '<text x="560" y="30" text-anchor="middle" font-family="Arial,sans-serif" font-size="21" font-weight="700" fill="#172033">Training and validation loss</text>',
        f'<text x="560" y="51" text-anchor="middle" font-family="Arial,sans-serif" font-size="12" fill="#526078">{"25M grown preview · durable checkpoint at step 3,000 of 15,000" if preview else ("25M grown from 10M · seed 1234 · completed 15,000 additional steps" if scale == "25m" else "10M gated run · seed 1234 · completed 15,000 steps")}</text>',
    ]
    for i in range(5):
        val = low + (high - low) * i / 4
        yy = y(val)
        parts.append(f'<line x1="{left}" y1="{yy:.1f}" x2="{width-right}" y2="{yy:.1f}" stroke="#e2e8f0"/>')
        parts.append(f'<text x="{left-10}" y="{yy+4:.1f}" text-anchor="end" font-family="Arial,sans-serif" font-size="11" fill="#526078">{val:.2f}</text>')
    for i in range(6):
        step = max_step * i / 5
        xx = x(step)
        parts.append(f'<line x1="{xx:.1f}" y1="{top}" x2="{xx:.1f}" y2="{height-bottom}" stroke="#f1f5f9"/>')
        parts.append(f'<text x="{xx:.1f}" y="{height-bottom+22}" text-anchor="middle" font-family="Arial,sans-serif" font-size="11" fill="#526078">{step:.0f}</text>')
    for values_, color, label in ((metrics, "#2563eb", "Training"), (evaluations, "#e11d48", "Validation")):
        ordered = sorted((v for v in values_ if v["step"] <= max_step), key=lambda z: z["step"])
        if not ordered:
            continue
        points = " ".join(f'{x(v["step"]):.1f},{y(float(v["loss"] if "loss" in v else v["val_loss"])):.1f}' for v in ordered)
        parts.append(f'<polyline fill="none" stroke="{color}" stroke-width="2.2" points="{points}"/>')
        parts.append(f'<line x1="{left+14}" y1="{height-27}" x2="{left+42}" y2="{height-27}" stroke="{color}" stroke-width="3"/><text x="{left+50}" y="{height-23}" font-family="Arial,sans-serif" font-size="12" fill="#334155">{label}</text>')
    parts.append(f'<text x="{width/2}" y="{height-8}" text-anchor="middle" font-family="Arial,sans-serif" font-size="12" fill="#334155">Optimizer step</text>')
    parts.append('</svg>')
    return "\n".join(parts) + "\n"


def benchmark_svg() -> tuple[str, list[dict]]:
    sources = [
        ("SLM-Tavenel 1M EXP-0010 (seed 421101)", ROOT / "exports/slm-tavenel-1m-exp0010-release-candidate-v4/training_record/selected-checkpoint-mc.json", "#2563eb"),
        ("TinyStories tiny1m", ROOT / "runs/reference/tiny1m/fsc-pilot-v1-mc.json", "#0891b2"),
        ("SLM-Tavenel 10M", RUNS / "10m-gated-s1234/analysis/fsc-pilot-v1-mc.json", "#059669"),
        ("Pythia-14M", ROOT / "runs/reference/pythia-14m/fsc-pilot-v1-mc.json", "#64748b"),
        ("SmolLM2-135M", ROOT / "runs/reference/smollm2-135m/fsc-pilot-v1-mc.json", "#db2777"),
    ]
    tasks = [("arc_easy", "ARC-Easy"), ("hellaswag", "HellaSwag"), ("belebele_en", "Belebele EN"), ("belebele_fr", "Belebele FR")]
    rows = []
    results = []
    fingerprint = None
    for name, path, color in sources:
        if not path.is_file():
            continue
        record = json.loads(path.read_text(encoding="utf-8"))
        record = normalize_benchmark_record(record)
        if fingerprint is None:
            fingerprint = record["benchmarks"]
        elif record.get("benchmarks") != fingerprint:
            raise ValueError(f"Benchmark files differ for {name}")
        if any(task not in record["results"] for task, _ in tasks):
            raise ValueError(f"Incomplete benchmark tasks for {name}")
        results.append((name, record, color))
        rows.append({"model": name, "parameters": record.get("model_params"), "scores": {task: record["results"][task]["acc"] for task, _ in tasks}})
    if len(results) < 2:
        raise FileNotFoundError("Need the 10M run and at least one reference evaluation")

    width, height = 1260, 650
    left, right, top, bottom = 74, 32, 78, 182
    plot_w, plot_h = width - left - right, height - top - bottom
    max_y = max(0.6, min(1.0, (max(r[1]["results"][task]["acc"] + r[1]["results"][task].get("acc_stderr", 0) for r in results for task, _ in tasks) * 10 // 1 + 1) / 10))
    y = lambda value: top + plot_h * (1 - value / max_y)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-labelledby="t d">',
        '<title id="t">SLM-Tavenel 10M benchmark comparison</title>',
        '<desc id="d">Raw accuracy on four shared multiple-choice benchmark files. Bars compare one checkpoint each; error bars show one standard error across questions. These are exploratory scores, not repeated-seed comparisons.</desc>',
        '<rect width="100%" height="100%" fill="#fff"/>',
        '<text x="630" y="32" text-anchor="middle" font-family="Arial,sans-serif" font-size="22" font-weight="700" fill="#172033">SLM-Tavenel 10M and SLM references</text>',
        '<text x="630" y="55" text-anchor="middle" font-family="Arial,sans-serif" font-size="12" fill="#526078">Same pinned item files and local evaluator · whiskers: ±1 standard error across items</text>',
    ]
    for tick in (0, .25, .5, .75, 1):
        yy = y(tick)
        parts.append(f'<line x1="{left}" y1="{yy:.1f}" x2="{width-right}" y2="{yy:.1f}" stroke="#e2e8f0"/>')
        parts.append(f'<text x="{left-10}" y="{yy+4:.1f}" text-anchor="end" font-family="Arial,sans-serif" font-size="11" fill="#526078">{tick:.0%}</text>')
    cy = y(.25)
    parts.append(f'<line x1="{left}" y1="{cy:.1f}" x2="{width-right}" y2="{cy:.1f}" stroke="#64748b" stroke-dasharray="5 4"/><text x="{width-right-2}" y="{cy-5:.1f}" text-anchor="end" font-family="Arial,sans-serif" font-size="10" fill="#64748b">four-choice chance ≈ 25%</text>')
    group_w = plot_w / len(tasks)
    bar_w = min(24, group_w / (len(results) * 1.8))
    gap = 5
    for ti, (task, label) in enumerate(tasks):
        center = left + group_w * (ti + .5)
        total = len(results) * bar_w + (len(results)-1) * gap
        start = center - total/2
        for mi, (name, record, color) in enumerate(results):
            item = record["results"][task]
            score, se = item["acc"], item.get("acc_stderr", 0)
            xx = start + mi * (bar_w + gap)
            yy, base = y(score), y(0)
            parts.append(f'<rect x="{xx:.1f}" y="{yy:.1f}" width="{bar_w:.1f}" height="{base-yy:.1f}" rx="2" fill="{color}"/>')
            cx, ya, yb = xx + bar_w/2, y(min(max_y, score+se)), y(max(0, score-se))
            parts.append(f'<line x1="{cx:.1f}" y1="{ya:.1f}" x2="{cx:.1f}" y2="{yb:.1f}" stroke="#172033"/><line x1="{cx-4:.1f}" y1="{ya:.1f}" x2="{cx+4:.1f}" y2="{ya:.1f}" stroke="#172033"/><line x1="{cx-4:.1f}" y1="{yb:.1f}" x2="{cx+4:.1f}" y2="{yb:.1f}" stroke="#172033"/>')
        parts.append(f'<text x="{center:.1f}" y="{height-bottom+26}" text-anchor="middle" font-family="Arial,sans-serif" font-size="13" font-weight="600" fill="#172033">{html.escape(label)}</text>')
    legend_y = height - 112
    for i, (name, _, color) in enumerate(results):
        x0 = 94 + (i % 3) * 390
        yy = legend_y + (i // 3) * 28
        parts.append(f'<rect x="{x0}" y="{yy-12}" width="14" height="14" rx="2" fill="{color}"/><text x="{x0+22}" y="{yy}" font-family="Arial,sans-serif" font-size="12" fill="#334155">{html.escape(name)}</text>')
    parts += [
        '<text x="630" y="610" text-anchor="middle" font-family="Arial,sans-serif" font-size="11" fill="#64748b">Different corpora, tokenizers, training budgets and possible benchmark exposure limit cross-model interpretation.</text>',
        '</svg>',
    ]
    return "\n".join(parts) + "\n", rows


def normalize_benchmark_record(record: dict) -> dict:
    """Normalize the EXP-0010 evaluation schema to the FSC comparison schema."""
    if "benchmarks" not in record and "benchmark_files" in record:
        record = dict(record)
        record["benchmarks"] = [
            {"path": item["path"], "sha256": item["sha256"]}
            for item in record["benchmark_files"]
            if item["path"].split("/")[-1] in {
                "arc_easy.jsonl", "hellaswag.jsonl", "belebele_en.jsonl", "belebele_fr.jsonl"
            }
        ]
    if "model_params" not in record and "parameters" in record:
        record = dict(record)
        record["model_params"] = record["parameters"]
    return record


def build_attribution(out: Path) -> tuple[dict[str, int], int]:
    source_path = PROCESSED / "documents.jsonl"
    dest = out / "training_record/source-record-manifest.jsonl"
    dest.parent.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    total = 0
    with source_path.open("r", encoding="utf-8") as src, dest.open("w", encoding="utf-8", newline="\n") as dst:
        for line in src:
            row = json.loads(line)
            meta = row.get("source_metadata") or {}
            rec = {
                "id": row.get("id"),
                "source": row.get("source"),
                "source_revision": row.get("source_revision"),
                "source_doc_id": row.get("source_doc_id"),
                "license_declared": row.get("license"),
                "processed_text_sha256": row.get("input_sha256"),
                "source_content_sha256": row.get("source_content_sha256"),
                "title": meta.get("title"),
                "author": meta.get("author"),
                "publication_date": meta.get("publication_date"),
                "doi": meta.get("doi"),
                "language": meta.get("language"),
                "source_labels": meta.get("source_labels"),
            }
            dst.write(json.dumps(rec, ensure_ascii=False, separators=(",", ":")) + "\n")
            counts[str(row.get("source"))] = counts.get(str(row.get("source")), 0) + 1
            total += 1
    return counts, total


def prepare(scale: str, out_arg: Path | None) -> Path:
    spec = RUN_INFO[scale]
    run = RUNS / spec["run_id"]
    checkpoint_path = run / "checkpoints" / spec["checkpoint"]
    manifest_path = run / "run_manifest.json"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    out = (out_arg or spec["out"]).resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Refusing to overwrite a non-empty candidate directory: {out}")
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    step = int(checkpoint["step"])
    expected_steps = int(manifest["train_config"]["max_steps"])
    if scale == "10m" and step != expected_steps:
        raise ValueError(f"Expected completed 10M run at {expected_steps}, found step {step}")
    if scale == "25m" and step != expected_steps:
        raise ValueError(f"Expected the completed 25M checkpoint at {expected_steps}, found {step}")

    state = checkpoint["model"]
    if manifest["model_config"].get("tie_embeddings"):
        state = {k: v for k, v in state.items() if k != "lm_head.weight"}
    state = {k: v.detach().to(device="cpu", dtype=torch.float32).contiguous() for k, v in state.items()}
    source_count = None
    source_counts = None
    if scale == "10m":
        source_counts, source_count = build_attribution(out)
    else:
        attribution_from = RUN_INFO["10m"]["out"] / "training_record/source-record-manifest.jsonl"
        if not attribution_from.is_file():
            raise FileNotFoundError(f"Prepare the 10M bundle first to reuse its source manifest: {attribution_from}")
        copy_file(attribution_from, out / "training_record/source-record-manifest.jsonl")
        source_count = sum(1 for _ in (out / "training_record/source-record-manifest.jsonl").open(encoding="utf-8"))
        source_counts = json.loads((RUN_INFO["10m"]["out"] / "training_record/source-counts.json").read_text(encoding="utf-8"))

    for rel in ("hf_config.py", "hf_model.py"):
        copy_file(ROOT / "src/slm" / rel, out / rel)
    for rel in ("model.py", "config.py", "train.py", "optim.py", "tokenizer.py", "runlog.py"):
        copy_file(ROOT / "src/slm" / rel, out / "training_record/src/slm" / rel)
    for rel in ("data/pipeline.py", "data/filters.py", "data/dedup.py", "data/pii.py", "data/contamination.py", "data/registry.py"):
        copy_file(ROOT / "src/slm" / rel, out / "training_record/src/slm" / rel)
    copy_file(Path(__file__).resolve(), out / "training_record/scripts/prepare_fsc_hf_scale.py")
    copy_file(ROOT / "scripts/verify_fsc_hf_candidate.py", out / "training_record/scripts/verify_fsc_hf_candidate.py")
    if scale == "25m":
        for rel in ("scripts/evaluate_fsc_grown_25m.py", "scripts/evaluate_fsc_grown_25m.ps1", "scripts/prepare_exp0009_25m_results.py", "scripts/analyze_exp0009_bpb_document_intervals.py"):
            copy_file(ROOT / rel, out / "training_record" / rel)
    copy_file(TOKENIZER, out / "tokenizer.json")
    copy_file(ROOT / "LICENSE", out / "LICENSE")
    copy_file(ROOT / "LICENSE", out / "CODE-LICENSE.txt")
    copy_file(ROOT / "src/slm/hf_config.py", out / "training_record/src/slm/hf_config.py")
    copy_file(ROOT / "src/slm/hf_model.py", out / "training_record/src/slm/hf_model.py")
    copy_file(ROOT / "src/slm/data/pipeline.py", out / "training_record/src/slm/data/pipeline.py")

    record_dir = out / "training_record"
    for name in ("run_manifest.json", "metrics.jsonl", "eval.jsonl", "observability.jsonl"):
        src = run / name
        if src.exists():
            copy_file(src, record_dir / name)
    for src, dest in (
        (DATA / "tokens/fsc-pilot-v1-scaling/provenance.json", record_dir / "data-provenance.json"),
        (DATA / "tokens/fsc-pilot-v1-scaling/manifest.json", record_dir / "token-shard-manifest.json"),
        (DATA / "processed/fsc-pilot-v1/manifest.json", record_dir / "processing-manifest.json"),
        (DATA / "processed/fsc-pilot-v1/stats.json", record_dir / "processing-stats.json"),
        (DATA / "processed/fsc-pilot-v1/decisions.jsonl", record_dir / "processing-decisions.jsonl"),
        (DATA / "processed/fsc-pilot-v1-token-balanced/final_token_balance_manifest.json", record_dir / "token-balance-manifest.json"),
        (DATA / "tokenizer/fsc-pilot-v1-train-only/manifest.json", record_dir / "tokenizer-manifest.json"),
        (ROOT / "configs/data/sources.yaml", record_dir / "source-registry.yaml"),
        (ROOT / "configs/data/fsc-pilot-v1.yaml", record_dir / "data-pipeline.yaml"),
        (ROOT / manifest["train_config"]["model"], record_dir / "model-config.yaml"),
    ):
        if src.exists():
            copy_file(src, dest)
    eval_path = run / "analysis/fsc-pilot-v1-mc.json"
    if eval_path.exists():
        copy_file(eval_path, record_dir / "multiple-choice-evaluation.json")
    if scale == "25m":
        for name in ("fsc-pilot-v2-mc.json", "fsc-pilot-v1-bpb-en.json", "fsc-pilot-v1-bpb-fr.json"):
            source = run / "analysis" / name
            if source.exists():
                copy_file(source, record_dir / name)
        bpb_bootstrap = ROOT / "runs/fsc-scaling/comparisons/exp0009-bpb-document-bootstrap.json"
        if bpb_bootstrap.exists():
            copy_file(bpb_bootstrap, record_dir / bpb_bootstrap.name)
        audit_bundle = RUN_INFO["10m"]["out"] / "training_record"
        for name in ("hal-file-license-audit-2026-10-22.json",):
            source = audit_bundle / name
            if source.exists():
                copy_file(source, record_dir / name)
        audit_script = ROOT / "scripts/audit_fsc_mixed_hal_files.py"
        if audit_script.exists():
            copy_file(audit_script, record_dir / "scripts/audit_fsc_mixed_hal_files.py")
        reference_records = ROOT / "runs/fsc-scaling/comparisons/exp0009-25m-reference-records"
        if reference_records.exists():
            for source in sorted(reference_records.glob("*.json")):
                copy_file(source, record_dir / "benchmark-references" / source.name)
        for rel in (
            "experiments/EXP-0009-25m-scaling-results-2026-10-22.md",
            "docs/reviews/EXP-0009-25M-RESUME-DIAGNOSTIC-2026-10-22.md",
            "docs/reviews/EXP-0009-HAL-FILE-LICENSE-AUDIT-2026-10-22.md",
            "docs/exp0009-25m-benchmark-comparison.svg",
            "docs/exp0009-25m-bpb-document-bootstrap.svg",
            "runs/fsc-scaling/comparisons/exp0009-25m-all-model-benchmarks.json",
            "runs/fsc-scaling/comparisons/exp0009-25m-paired-item-intervals.json",
        ):
            source = ROOT / rel
            if source.exists():
                if rel.startswith("docs/"):
                    copy_file(source, out / rel)
                else:
                    dest = record_dir / Path(rel).name
                    copy_file(source, dest)
                    if rel.startswith("experiments/"):
                        report_text = dest.read_text(encoding="utf-8")
                        report_text = report_text.replace(
                            "The full input hashes, per-item scores and comparison JSON remain in `runs/fsc-scaling/`.",
                            "The model package includes checkpoint, data and benchmark hashes, per-item outputs and comparison JSON under `training_record/`.",
                        )
                        dest.write_text(report_text, encoding="utf-8")
    if scale == "10m":
        card_svg, rows = benchmark_svg()
        (out / "benchmark-comparison.svg").write_text(card_svg, encoding="utf-8")
        write_json(record_dir / "benchmark-scores.json", rows)
        reference_files = {
            "slm-tavenel-1m-exp0010-selected.json": ROOT / "exports/slm-tavenel-1m-exp0010-release-candidate-v4/training_record/selected-checkpoint-mc.json",
            "tinystories-tiny1m.json": ROOT / "runs/reference/tiny1m/fsc-pilot-v1-mc.json",
            "pythia-14m.json": ROOT / "runs/reference/pythia-14m/fsc-pilot-v1-mc.json",
            "smollm2-135m.json": ROOT / "runs/reference/smollm2-135m/fsc-pilot-v1-mc.json",
        }
        for name, source in reference_files.items():
            copy_file(source, record_dir / "benchmark-references" / name)
        report = ROOT / "experiments/EXP-0009-fsc-scaling-and-references.md"
        if report.exists():
            copy_file(report, record_dir / report.name)
    metrics = [json.loads(x) for x in (run / "metrics.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    evals = [json.loads(x) for x in (run / "eval.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    if scale == "25m":
        metrics = [x for x in metrics if int(x["step"]) <= step]
        evals = [x for x in evals if int(x["step"]) <= step]
    (out / "training-curve.svg").write_text(svg_line_chart(metrics, evals, preview=scale == "25m" and step < expected_steps, scale=scale), encoding="utf-8")

    tok = Tokenizer.from_file(str(TOKENIZER))
    eos_id, pad_id = tok.token_to_id("<|endoftext|>"), tok.token_to_id("<|pad|>")
    model_config = manifest["model_config"]
    config = {
        "model_type": "slm_tavenel", "architectures": ["SlmTavenelForCausalLM"],
        "auto_map": {"AutoConfig": "hf_config.SlmTavenelConfig", "AutoModelForCausalLM": "hf_model.SlmTavenelForCausalLM"},
        **model_config, "max_position_embeddings": model_config["max_seq_len"],
        "tie_word_embeddings": model_config["tie_embeddings"], "torch_dtype": "float32",
        "use_cache": False, "eos_token_id": eos_id, "bos_token_id": None, "pad_token_id": pad_id,
    }
    write_json(out / "config.json", config)
    write_json(out / "generation_config.json", {"eos_token_id": eos_id, "pad_token_id": pad_id, "use_cache": False})
    write_json(out / "tokenizer_config.json", {
        "tokenizer_class": "PreTrainedTokenizerFast", "eos_token": "<|endoftext|>",
        "pad_token": "<|pad|>", "bos_token": None, "unk_token": None,
        "model_max_length": model_config["max_seq_len"], "clean_up_tokenization_spaces": False,
        "add_bos_token": False, "model_input_names": ["input_ids", "attention_mask"],
    })
    write_json(out / "special_tokens_map.json", {"eos_token": "<|endoftext|>", "pad_token": "<|pad|>"})
    save_file(state, str(out / "model.safetensors"), metadata={"format": "pt"})
    (out / "requirements.txt").write_text("torch>=2.3\ntransformers==4.57.6\ntokenizers==0.22.2\nsafetensors==0.8.0\n", encoding="utf-8")

    stats = json.loads((PROCESSED / "stats.json").read_text(encoding="utf-8"))
    license_counts = {
        "CC0-1.0": stats["per_source"]["fsc-en-cc0"]["out"] + stats["per_source"]["fsc-fr-cc0"]["out"],
        "CC-BY-4.0": stats["per_source"]["fsc-en-ccby4"]["out"] + stats["per_source"]["fsc-fr-ccby4"]["out"],
        "Licence-Ouverte-2.0": stats["per_source"]["fsc-en-lo2"]["out"] + stats["per_source"]["fsc-fr-lo2"]["out"],
    }
    if scale == "10m":
        write_json(record_dir / "source-counts.json", source_counts)
        data_card = f"""# EXP-0009 FSC pilot data card

## Source and scope

The run used the `french_open_science` subset of PleIAs/French-Science-Commons, pinned to `{manifest['data']['provenance']['revision']}`. The source registry grouped records into six source classes: English and French records labeled CC0-1.0, CC-BY-4.0, and Licence-Ouverte-2.0. The exact per-record metadata retained by the preprocessing pipeline is listed in `source-record-manifest.jsonl`; article text is not included in this model package.

The pipeline began with 43,011 documents and retained 31,417. The retained source-declared license labels total {license_counts['CC0-1.0']:,} CC0-1.0, {license_counts['CC-BY-4.0']:,} CC-BY-4.0, and {license_counts['Licence-Ouverte-2.0']:,} Licence-Ouverte-2.0 records. The training source count and processed-text hashes are traceability metadata, not independent proof of rights.

## Processing

The recorded pipeline normalized text, applied length and Gopher-style quality filters, redacted email, phone, IBAN and IP-shaped strings, deduplicated exact and near-duplicate documents, and applied 13-gram benchmark-overlap filtering against the recorded four-task benchmark files. No benchmark documents were removed by the logged overlap filter. Detailed decisions and aggregate statistics are included in this folder.

Automated redaction and pattern scans do not constitute a complete privacy review. The data used for training has not been independently reviewed line by line for personal or sensitive information. No assertion is made that learned parameters cannot reproduce memorized content.

## License and attribution

CC-BY-4.0 and Licence-Ouverte-2.0 labels carry attribution obligations. `ATTRIBUTION.md` and the record-level metadata manifest preserve title, declared author, DOI, source identifiers, license labels and content hashes when those fields were available. Source metadata labels are reported as supplied; submitter authority and all originating publisher terms were not independently verified. No rights clearance is claimed.

## Reproducibility and limitations

The pinned dataset revision, preprocessing manifest, filtering decisions, token-split manifests, tokenizer manifest, token-shard manifest and processing scripts are included. Source text and token shards are not redistributed. The run used one seed. Its recorded training worktree was dirty, and the uncommitted diff was not captured, so exact source-state reproduction is not claimed. A fixed-input native-to-Hugging-Face parity report is included, with its narrow scope stated in the report.
"""
        (record_dir / "data-card.md").write_text(data_card, encoding="utf-8")
    card = model_card(scale, spec["repo_id"], manifest, checkpoint, license_counts, source_count)
    (out / "README.md").write_text(card, encoding="utf-8")
    notice = f"""# Notices and data provenance

The SLM-Tavenel code and project-authored model artifact are offered under the MIT License in `LICENSE`. The training dataset was not relicensed by this notice.

Training source: [PleIAs/French-Science-Commons](https://huggingface.co/datasets/PleIAs/French-Science-Commons), subset `french_open_science`, revision `bfa114c09d7c28dcd0a8a47fe445c64b4a3c6156`. Selected and processed records carry source-declared labels CC0-1.0, CC-BY-4.0, or Licence-Ouverte-2.0. The retained manifest contains {source_count:,} metadata-only records: {license_counts['CC0-1.0']:,} CC0, {license_counts['CC-BY-4.0']:,} CC-BY-4.0, and {license_counts['Licence-Ouverte-2.0']:,} Licence-Ouverte-2.0.

The CC-BY-4.0 and Licence-Ouverte labels require attribution. Per-record titles, declared authors, identifiers, DOI fields, source identifiers and hashes are provided in `training_record/source-record-manifest.jsonl` to support source tracing and attribution. The records do not independently prove each submitter's authority to apply the declared license. The project did not independently verify every originating publisher notice, and no legal rights clearance is claimed. Source article text is not redistributed here.

See `training_record/data-provenance.json`, `training_record/processing-manifest.json`, `training_record/source-registry.yaml`, `training_record/processing-decisions.jsonl`, and the model card for dataset scope, filtering, privacy screens, attribution obligations and limitations.
"""
    (out / "NOTICE.md").write_text(notice, encoding="utf-8")
    (out / "ATTRIBUTION.md").write_text(notice, encoding="utf-8")

    provenance = {
        "schema": "slm.hf_fsc_scale_release.v1", "status": "local candidate; not uploaded",
        "uploaded": False, "repo_id": spec["repo_id"], "scale": scale,
        "training_run": spec["run_id"], "checkpoint_step": step, "target_steps": expected_steps,
        "checkpoint_sha256": sha256(checkpoint_path), "run_manifest_sha256": sha256(manifest_path),
        "parameters": manifest["params"], "seed": manifest["train_config"]["seed"],
        "code_commit": manifest["code"]["commit"], "code_worktree_was_dirty": manifest["code"]["dirty"],
        "dataset": {"name": "PleIAs/French-Science-Commons", "revision": manifest["data"]["provenance"]["revision"], "subset": manifest["data"]["provenance"]["subset"], "declared_license_record_counts": license_counts, "source_record_manifest_count": source_count, "text_redistributed": False, "review_gaps": ["independently verify each selected record's license against its originating publisher or repository", "complete human privacy review beyond pattern-based filtering", "review whether the provided attribution metadata fully satisfies all applicable source-license obligations"]},
        "release_limitations": ["single training seed", "training run captured a dirty worktree", "source rights and privacy were not independently cleared", "source license metadata does not establish submitter authority"],
        "inference_parity": {"status": "not yet checked for this scale"},
        "files_sha256": {},
    }
    if scale == "25m":
        if step < expected_steps:
            provenance["status"] = "incomplete local preview; not a final model and not uploaded"
            provenance["training_state"] = {"durable_checkpoint_step": step, "configured_total_steps": expected_steps, "resumable_checkpoint_included": True}
            copy_file(checkpoint_path, record_dir / "resume-checkpoint.pt")
        else:
            provenance["status"] = "completed local research candidate; not uploaded; source-rights and privacy review gaps disclosed"
            provenance["training_state"] = {"durable_checkpoint_step": step, "configured_total_steps": expected_steps, "completed": True, "resumable_checkpoint_included": False}
            audit_path = record_dir / "hal-file-license-audit-2026-10-22.json"
            if audit_path.exists():
                audit = json.loads(audit_path.read_text(encoding="utf-8"))
                provenance["dataset"]["hal_file_license_audit"] = {
                    "report": "training_record/hal-file-license-audit-2026-10-22.json",
                    "report_sha256": sha256(audit_path),
                    "status": audit["result"],
                    "counts": audit["counts"],
                    "scope": "current HAL metadata only; not a historical or rights-holder verification",
                }
    for file in sorted(out.rglob("*")):
        if file.is_file() and file.name != "provenance.json":
            provenance["files_sha256"][file.relative_to(out).as_posix()] = sha256(file)
    write_json(out / "provenance.json", provenance)
    print(json.dumps({"out": str(out), "scale": scale, "parameters": manifest["params"], "checkpoint_step": step, "source_records": source_count, "license_counts": license_counts, "bundle_files": len(provenance["files_sha256"]) + 1}, indent=2))
    return out


def model_card(scale: str, repo_id: str, manifest: dict, checkpoint: dict, license_counts: dict[str, int], source_count: int) -> str:
    params = manifest["params"]
    step = int(checkpoint["step"])
    train_config = manifest["train_config"]
    expected_steps = int(train_config["max_steps"])
    eval_file = RUNS / "10m-gated-s1234/analysis/fsc-pilot-v1-mc.json"
    eval_data = json.loads(eval_file.read_text(encoding="utf-8")) if eval_file.exists() else None
    if scale == "10m":
        model_rows = [
            ("SLM-Tavenel 10M", params, eval_data),
            ("SLM-Tavenel 1M EXP-0010 (seed 421101)", 1_181_920, json.loads((ROOT / "exports/slm-tavenel-1m-exp0010-release-candidate-v4/training_record/selected-checkpoint-mc.json").read_text(encoding="utf-8"))),
            ("TinyStories tiny1m", 935_000, json.loads((ROOT / "runs/reference/tiny1m/fsc-pilot-v1-mc.json").read_text(encoding="utf-8"))),
            ("Pythia-14M", 14_100_000, json.loads((ROOT / "runs/reference/pythia-14m/fsc-pilot-v1-mc.json").read_text(encoding="utf-8"))),
            ("SmolLM2-135M", 135_000_000, json.loads((ROOT / "runs/reference/smollm2-135m/fsc-pilot-v1-mc.json").read_text(encoding="utf-8"))),
        ]
        tasks = [("arc_easy", "ARC-Easy"), ("hellaswag", "HellaSwag"), ("belebele_en", "Belebele EN"), ("belebele_fr", "Belebele FR")]
        table = ["| Model | Parameters | ARC-Easy | HellaSwag | Belebele EN | Belebele FR |", "|---|---:|---:|---:|---:|---:|"]
        for name, nparams, record in model_rows:
            record = normalize_benchmark_record(record)
            scores = " | ".join(f"{record['results'][task]['acc']:.3f}" for task, _ in tasks)
            table.append(f"| {name} | {nparams/1e6:.3g}M | {scores} |")
        model_table = "\n".join(table)
        header = f"""---
language:
- en
- fr
license: mit
library_name: transformers
pipeline_tag: text-generation
datasets:
- PleIAs/French-Science-Commons
tags:
- slm-tavenel
- experimental
- research-artifact
- base-model
- open-weights
---

# SLM-Tavenel 10M | EXP-0009 FSC scaling pilot

> **Experimental base model.** One completed 10M gated-attention checkpoint, trained from scratch for 15,000 steps on a bilingual science-text pilot. Project weights and code are offered under MIT. The source corpus mixes CC0, CC-BY-4.0 and Licence-Ouverte-2.0 records; the declared source licenses and unresolved rights/privacy checks are disclosed below. This is not an OSI certification or a rights clearance.

| Model facts | |
|---|---|
| Parameters | {params:,} |
| Languages | English and French |
| Architecture | 10-layer decoder-only transformer, head-wise attention gate |
| Context | 1,024 tokens |
| Tokenizer | 8,192-token byte-level BPE |
| Training | 15,000 steps, seed 1234, AdamW, bfloat16 |
| Training token positions | 491,520,000 |
| Final train / validation loss | 2.29647 / 2.33488 |

## Results at a glance

The checkpoint scored with the same local evaluator and pinned benchmark files used for the listed SLM references. Accuracy is close to the four-choice chance level on these tasks. The external comparisons are descriptive only because the models have different corpora, tokenizers, training budgets and possible benchmark exposure.

![Training and validation loss across the completed 10M run](training-curve.svg)

*Figure 1. Training and validation loss over optimizer steps.*

{model_table}

![SLM-Tavenel 10M and same-harness SLM reference scores](benchmark-comparison.svg)

*Figure 2. Raw accuracy. Whiskers show plus or minus one standard error across questions, conditional on each checkpoint. They do not measure seed variation.*

## Intended use

This base model is intended for research and education about small-model training, bilingual text modeling, reproducibility and training observability. It is not instruction-tuned, is not a conversational assistant, and should not be used as a reliable factual or scientific advisor.

## Load with Transformers

The model uses custom Transformers code, requires `trust_remote_code=True`, and does not implement a KV cache.

```bash
pip install -r requirements.txt
```

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "{repo_id}"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
prompt = "La recherche scientifique"
inputs = tokenizer(prompt, return_tensors="pt")
print(tokenizer.decode(model.generate(**inputs, max_new_tokens=48, do_sample=False)[0]))
```

## Training and observability

The run used a single seed (1234), an NVIDIA GeForce RTX 4060 Laptop GPU with 8 GB, a 1,024-token sequence length, micro-batch size 16, two gradient-accumulation steps and AdamW. It processed 491,520,000 token positions over 15,000 steps. The selected checkpoint contains {params:,} parameters. Per-step metrics, validation losses and observability records are included in `training_record/`.

The recorded training worktree was dirty. The package includes the current model implementation, architecture and training configuration, run manifest and logs, but the original uncommitted source diff was not captured at run time. Exact byte-for-byte reproduction of the training code state is therefore not claimed.

## Data and source attribution

Training data came from the `french_open_science` subset of [PleIAs/French-Science-Commons](https://huggingface.co/datasets/PleIAs/French-Science-Commons), pinned to `{manifest['data']['provenance']['revision']}`. The processing pipeline started with 43,011 records and retained 31,417 after quality, PII-pattern, exact-duplicate and near-duplicate processing. The retained source metadata reports {license_counts['CC0-1.0']:,} records labeled CC0-1.0, {license_counts['CC-BY-4.0']:,} CC-BY-4.0, and {license_counts['Licence-Ouverte-2.0']:,} Licence-Ouverte-2.0. The record-level metadata-only manifest lists titles, declared authors, identifiers, DOI fields, source IDs, declared labels and hashes. Source texts are not included in this repository.

CC-BY-4.0 and Licence-Ouverte-2.0 require attribution. `ATTRIBUTION.md` and `training_record/source-record-manifest.jsonl` provide source-level attribution information to support tracing. The metadata label does not independently establish that each submitter had authority to apply it. The project has not independently verified every originating publisher's terms; no blanket rights clearance is claimed. Dataset and HAL metadata also do not prove that every record is free of personal or sensitive information. Pattern-based redaction and screening were limited, not a complete privacy audit.

## Limitations

- One seed and one completed run; no training-seed uncertainty estimate.
- Exploratory benchmark accuracy is near chance on several tasks and does not establish general model quality.
- The training worktree was dirty and the exact uncommitted diff is unavailable.
- Source license labels, attribution obligations, submitter authority and privacy risks remain incompletely reviewed.
- The source article text is not redistributed. MIT applies to the project code and model artifact as offered by the project; it does not relicense source records or guarantee the absence of third-party claims.

## Reproducibility files

`training_record/` contains the checkpoint provenance, run manifest, train/eval/observability logs, source registry, processing manifest and decisions, tokenizer and token-shard manifests, source-record metadata manifest, benchmark score record, model implementation, and configurations. The SHA-256 inventory is in `provenance.json`.

## License and citation

Model weights, project code and included Data Information are offered under MIT. Source records remain subject to their separately declared licenses. Cite this model as SLM-Tavenel EXP-0009, seed 1234, and include the immutable Hugging Face revision used. The experiment report and protocol are in the SLM-Tavenel repository under `experiments/EXP-0009-fsc-scaling-and-references.md`.
"""
    elif step < expected_steps:
        header = f"""---
language:
- en
- fr
license: mit
library_name: transformers
pipeline_tag: text-generation
datasets:
- PleIAs/French-Science-Commons
tags:
- slm-tavenel
- experimental
- incomplete-checkpoint
- research-artifact
- base-model
- open-weights
---

# SLM-Tavenel 25M | EXP-0009 grown preview

> **Incomplete local preview. Do not cite this as a completed 25M model and do not publish this package as a final release.** This checkpoint is the 25.72M-parameter model grown from the completed 10M SLM-Tavenel checkpoint and continued for 3,000 of 15,000 planned steps. The durable checkpoint is included for resuming the experiment. No final evaluation or benchmark comparison exists for this checkpoint.

| Preview facts | |
|---|---|
| Parameters | {params:,} |
| Durable checkpoint | step {step:,} of {expected_steps:,} |
| Additional tokens processed | {int(checkpoint.get('tokens', step * 32768)):,} (checkpoint metadata) |
| Initialization | Block expansion and MLP widening from the completed 10M gated checkpoint |
| Data | Same exploratory FSC pilot v1 as the 10M run |
| Status | Training paused; resume from `training_record/resume-checkpoint.pt` |

## Training snapshot

![Training and validation loss through the durable 25M checkpoint](training-curve.svg)

*Figure 1. Preview metrics are shown only through step {step:,}. No final score is available.*

## Resume

This package preserves the durable full checkpoint, run manifest, logs, model weights and config. Resume training with the repository's `configs/train/fsc-pilot-v1-25m-grown.yaml` after restoring the checkpoint into the run directory. The 10M source checkpoint and growth configuration are referenced in `training_record/run_manifest.json`. A fixed-input native-to-Hugging-Face load and generation check is included; it does not validate training completion or model quality.

## Data, license and release status

The data source and license labels are the same as those described in the 10M experiment. The source-record metadata manifest is included; source article texts are omitted. CC-BY-4.0 and Licence-Ouverte-2.0 attribution obligations and the unverified source-rights/privacy limitations are described in `ATTRIBUTION.md` and `training_record/data-provenance.json`.

Model weights and project code are offered under MIT as project terms. This package is local and incomplete. It has not been uploaded to Hugging Face, evaluated as a final checkpoint, or self-assessed against OSAID 1.0.
"""
    else:
        metric_path = RUNS / "25m-gated-grown-s1234/metrics.jsonl"
        eval_path = RUNS / "25m-gated-grown-s1234/eval.jsonl"
        metrics = [json.loads(line) for line in metric_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        evals = [json.loads(line) for line in eval_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        final_train_loss = float(metrics[-1]["loss"])
        final_val = next((float(item["val_loss"]) for item in reversed(evals) if int(item["step"]) == step), None)
        if final_val is None:
            raise FileNotFoundError("Final 25M validation loss is missing from eval.jsonl")
        comparison_path = ROOT / "runs/fsc-scaling/comparisons/exp0009-25m-all-model-benchmarks.json"
        if not comparison_path.exists():
            raise FileNotFoundError("Prepare the same-harness 25M comparison before building its local candidate")
        comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
        labels = ["ARC-Easy", "ARC-Challenge", "HellaSwag", "Belebele EN", "Belebele FR", "WinoGrande"]
        tasks = ["arc_easy", "arc_challenge", "hellaswag", "belebele_en", "belebele_fr", "winogrande"]
        rows = ["| Model | Parameters | " + " | ".join(labels) + " |", "|---|---:|" + "---:|" * len(tasks)]
        for model_name, model_data in comparison["models"].items():
            scores = [f"{model_data['results'][task]['acc']:.3f}" if task in model_data["results"] else "not scored" for task in tasks]
            rows.append(f"| {model_name} | {model_data['parameters']:,} | " + " | ".join(scores) + " |")
        score_table = "\n".join(rows)
        bpb_en = json.loads((RUNS / "25m-gated-grown-s1234/analysis/fsc-pilot-v1-bpb-en.json").read_text(encoding="utf-8"))["bits_per_byte"]
        bpb_fr = json.loads((RUNS / "25m-gated-grown-s1234/analysis/fsc-pilot-v1-bpb-fr.json").read_text(encoding="utf-8"))["bits_per_byte"]
        bpb_bootstrap = json.loads((ROOT / "runs/fsc-scaling/comparisons/exp0009-bpb-document-bootstrap.json").read_text(encoding="utf-8"))
        bpb_rows = ["| Paired comparison (10M minus grown 25M) | Documents | BPB difference | 95% document-bootstrap interval |", "|---|---:|---:|---:|"]
        for key, label, docs in (("en", "English", "685"), ("fr", "French", "678"), ("equal_language_mean", "Equal-language mean", "685 EN; 678 FR")):
            item = bpb_bootstrap["results"][key]
            lo, hi = item["ci95_percentile"]
            bpb_rows.append(f"| {label} | {docs} | {item['delta_bpb_10m_minus_25m']:+.4f} | [{lo:+.4f}, {hi:+.4f}] |")
        bpb_interval_table = "\n".join(bpb_rows)
        header = f"""---
language:
- en
- fr
license: mit
library_name: transformers
pipeline_tag: text-generation
datasets:
- PleIAs/French-Science-Commons
tags:
- slm-tavenel
- experimental
- research-artifact
- base-model
- open-weights
---

# SLM-Tavenel grown 25M | EXP-0009

> **Completed experimental base model.** This 25.72M-parameter checkpoint was grown from the trained SLM-Tavenel 10M model and then trained for 15,000 additional steps. It is one seed and one initialization path. Its weights are a local candidate and are not uploaded. Rights and privacy review gaps for the mixed-license source corpus are disclosed below.

| Model facts | |
|---|---|
| Parameters | {params:,} |
| Architecture | 20-layer decoder-only transformer, head-wise attention gate |
| Context | 1,024 tokens |
| Tokenizer | 8,192-token byte-level BPE |
| Additional training | 15,000 steps, seed 1234, 491,520,000 token positions |
| Total lineage budget | Approximately 983,040,000 token positions, including the trained 10M ancestor |
| Final train / validation loss | {final_train_loss:.5f} / {final_val:.5f} |
| Held-out validation BPB, EN / FR | {bpb_en:.4f} / {bpb_fr:.4f} |

## Results at a glance

The 25M checkpoint and references were scored with this repository's evaluator on the same pinned benchmark files where available. The graph compares raw accuracy and shows one-standard-error bars across items. Those bars condition on each checkpoint and do not measure variation across training runs.

![SLM-Tavenel 1M, 10M and grown 25M benchmark results compared with SLM references](docs/exp0009-25m-benchmark-comparison.svg)

{score_table}

### Held-out English and French bits per byte

Lower is better. These point estimates use the same fixed FSC validation texts and tokenizer for the three SLM-Tavenel checkpoints. The 10M-versus-25M confidence intervals below use a paired document bootstrap with 20,000 draws, seed {bpb_bootstrap['uncertainty']['seed']}, and documents resampled within each language. Positive differences favor the grown 25M checkpoint.

![Paired document-bootstrap intervals for held-out BPB differences](docs/exp0009-25m-bpb-document-bootstrap.svg)

| SLM-Tavenel checkpoint | English BPB | French BPB |
|---|---:|---:|
| 1M EXP-0010, selected | {json.loads((RUNS / '1m-gated-s1234/analysis/fsc-pilot-v1-bpb-en.json').read_text(encoding='utf-8'))['bits_per_byte']:.4f} | {json.loads((RUNS / '1m-gated-s1234/analysis/fsc-pilot-v1-bpb-fr.json').read_text(encoding='utf-8'))['bits_per_byte']:.4f} |
| 10M EXP-0009 | {json.loads((RUNS / '10m-gated-s1234/analysis/fsc-pilot-v1-bpb-en.json').read_text(encoding='utf-8'))['bits_per_byte']:.4f} | {json.loads((RUNS / '10m-gated-s1234/analysis/fsc-pilot-v1-bpb-fr.json').read_text(encoding='utf-8'))['bits_per_byte']:.4f} |
| grown 25M EXP-0009 | **{bpb_en:.4f}** | **{bpb_fr:.4f}** |

{bpb_interval_table}

The intervals quantify validation-document sampling for these two saved checkpoints only. There is one training seed at each scale, so the results do not estimate training-run variability. See `training_record/exp0009-bpb-document-bootstrap.json` for the hashes, method and detailed results.

This is not a parameter-count-only comparison: the 25M model inherits the 10M training history and receives another 491.52M token positions. The single run cannot estimate seed variation. External models use different corpora, tokenizers, compute and training budgets, and may have seen benchmark material. Results are descriptive and exploratory, not a general ranking.

## Intended use

Research and education on small-model training, bilingual text modeling, reproducibility and observability. This is a base model, not an instruction-tuned assistant or a reliable scientific advisor.

## Load with Transformers

The model uses custom Transformers code and requires `trust_remote_code=True`.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "{repo_id}"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
```

## Training, observability and reproducibility

The run resumed from checkpoint 3,000 in the recorded Python 3.11.9 / PyTorch 2.10.0+cu130 environment after a mismatched-runtime attempt was stopped and archived. Recomputed validation at the resume point and the step-3,500 losses matched the recorded trajectory. The configuration is not bitwise deterministic, and the original worktree was dirty, so exact source-state reproduction is not claimed. Per-step losses, validation, gradient, optimizer and observability records are included in `training_record/`. The [resume diagnostic](docs/reviews/EXP-0009-25M-RESUME-DIAGNOSTIC-2026-10-22.md) describes the archived attempt.

## Data, licenses and privacy

Training used the `french_open_science` subset of [PleIAs/French-Science-Commons](https://huggingface.co/datasets/PleIAs/French-Science-Commons), revision `{manifest['data']['provenance']['revision']}`. The retained 31,417-row manifest preserves source identifiers, source-declared license labels, titles, declared authors, DOI fields and hashes. No source text is included. A post-training HAL metadata audit found matching main/document-file labels for 31,262 source IDs; 72 explicit mismatches, 3 conflicts, 24 missing file-license fields and 56 not-found IDs remain unresolved. Current metadata does not prove historical status or submitter authority. The full metadata-only audit is included in `training_record/hal-file-license-audit-2026-10-22.json`.

Automated redaction is not a complete privacy review. Human source-rights and privacy sign-off remain pending. No rights clearance or absence-of-memorization claim is made. The MIT terms for project-authored code and weights do not relicense the source records or guarantee absence of third-party claims.

## Evaluation files

`training_record/` includes the final checkpoint provenance, source manifest, data and token manifests, training metrics, validation, benchmark outputs, paired item-bootstrap comparisons, model code, configurations, audit report, and SHA-256 inventory. Benchmark intervals quantify item-sampling uncertainty for these checkpoints only; they do not estimate training variability. Exact-overlap contamination checks do not rule out paraphrase, translation or prior exposure.

## Citation and release status

Cite this local research candidate as SLM-Tavenel EXP-0009 grown 25M, seed 1234, and identify the checkpoint SHA-256 from `provenance.json`. It has not been uploaded to Hugging Face or certified by OSI. The local candidate and its provenance are not a rights or privacy approval.
"""
    return header


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scale", choices=RUN_INFO)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    prepare(args.scale, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
