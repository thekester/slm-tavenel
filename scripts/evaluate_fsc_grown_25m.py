"""Evaluate the completed grown-25M FSC run on the frozen v2 benchmark files and held-out text."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from slm.benchmarks import load_items, multiple_choice  # noqa: E402
from slm.config import ModelConfig  # noqa: E402
from slm.evaluation import bits_per_byte, read_texts  # noqa: E402
from slm.model import SLM  # noqa: E402
from slm.tokenizer import load as load_tokenizer  # noqa: E402


RUN = ROOT / "runs/fsc-scaling/25m-gated-grown-s1234"
CHECKPOINT = RUN / "checkpoints/step_00015000.pt"
TOKENIZER = ROOT / "data/tokenizer/fsc-pilot-v1-train-only/tokenizer.json"
BENCHMARKS = [
    ROOT / "data/benchmarks/arc_easy.jsonl",
    ROOT / "data/benchmarks/arc_challenge.jsonl",
    ROOT / "data/benchmarks/hellaswag.jsonl",
    ROOT / "data/benchmarks/belebele_en.jsonl",
    ROOT / "data/benchmarks/belebele_fr.jsonl",
    ROOT / "data/benchmarks/winogrande.jsonl",
]
VALIDATION = {
    "en": ROOT / "data/processed/fsc-pilot-v1-token-balanced/en-val.jsonl",
    "fr": ROOT / "data/processed/fsc-pilot-v1-token-balanced/fr-val.jsonl",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(f"Final 25M checkpoint not found: {CHECKPOINT}")
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    if int(checkpoint["step"]) != 15000:
        raise ValueError(f"Expected final step 15000; found {checkpoint['step']}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SLM(ModelConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    tokenizer = load_tokenizer(TOKENIZER)

    benchmark_results = {}
    for path in BENCHMARKS:
        result = multiple_choice(model, tokenizer, load_items(path), device)
        benchmark_results[path.stem] = result
        print(json.dumps({"benchmark": path.stem, "acc": result["acc"], "acc_norm": result["acc_norm"]}), flush=True)

    inputs = {
        "checkpoint": {"path": CHECKPOINT.relative_to(ROOT).as_posix(), "sha256": sha256(CHECKPOINT), "step": 15000},
        "tokenizer": {"path": TOKENIZER.relative_to(ROOT).as_posix(), "sha256": sha256(TOKENIZER)},
        "benchmarks": [{"path": p.relative_to(ROOT).as_posix(), "sha256": sha256(p)} for p in BENCHMARKS],
        "validation": {
            lang: {"path": path.relative_to(ROOT).as_posix(), "sha256": sha256(path)}
            for lang, path in VALIDATION.items()
        },
    }
    reference_paths = [
        ROOT / "runs/fsc-scaling/10m-gated-s1234/analysis/fsc-pilot-v1-mc.json",
        ROOT / "runs/fsc-additional/slm-tavenel-10m/fsc-pilot-v2-mc.json",
    ]
    expected = {}
    for reference_path in reference_paths:
        ref = json.loads(reference_path.read_text(encoding="utf-8"))
        expected.update({Path(x["path"]).name: x["sha256"] for x in ref["benchmarks"]})
    observed = {Path(x["path"]).name: x["sha256"] for x in inputs["benchmarks"]}
    if observed != expected:
        raise ValueError("Frozen benchmark file hashes differ from the completed 10M same-harness evaluation")

    output = {
        "kind": "slm.exp0009_grown_25m_v2_evaluation",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model": "SLM-Tavenel grown 25M EXP-0009",
        "parameters": model.num_params(),
        "seed": 1234,
        "training_run": "25m-gated-grown-s1234",
        "evaluation_harness": "SLM-Tavenel local log-likelihood evaluator",
        "evaluation_runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
        },
        "evaluator_code_sha256": {
            "slm_benchmarks.py": sha256(ROOT / "src/slm/benchmarks.py"),
            "slm_evaluation.py": sha256(ROOT / "src/slm/evaluation.py"),
            "evaluation_script.py": sha256(Path(__file__).resolve()),
        },
        "evaluation_inputs": inputs,
        "contamination_audit": {
            "path": "runs/fsc-additional/contamination-audit.json",
            "sha256": sha256(ROOT / "runs/fsc-additional/contamination-audit.json"),
            "scope": "post-hoc exact 13-gram overlap check for ARC-Challenge and WinoGrande only",
        },
        "contamination_status": "post-hoc exact-overlap check only; paraphrase, translation, residual and pretraining exposure remain unknown",
        "benchmarks": [
            {"path": p.relative_to(ROOT).as_posix(), "sha256": sha256(p), "split": "test" if p.stem.startswith("arc_") else "validation" if p.stem == "winogrande" else "validation"}
            for p in BENCHMARKS
        ],
        "results": benchmark_results,
        "limitations": [
            "one grown training trajectory and one seed; no seed-level uncertainty estimate",
            "post-hoc benchmark item intervals are conditional on these checkpoints and do not measure training variability",
            "the model inherits the trained 10M checkpoint and twice the 10M-from-scratch token budget; not an iso-compute comparison",
            "the mixed FSC training corpus remains under unresolved rights and privacy review",
        ],
    }
    out_dir = RUN / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "fsc-pilot-v2-mc.json").write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    for lang, path in VALIDATION.items():
        result = bits_per_byte(model, tokenizer, read_texts(path), device)
        result.update({
            "checkpoint": inputs["checkpoint"]["path"],
            "checkpoint_sha256": inputs["checkpoint"]["sha256"],
            "data": path.relative_to(ROOT).as_posix(),
            "data_sha256": inputs["validation"][lang]["sha256"],
            "tokenizer_sha256": inputs["tokenizer"]["sha256"],
            "metric_scope": "FSC-v1 language-specific validation split; same text and tokenizer as the 10M scaling run",
        })
        (out_dir / f"fsc-pilot-v1-bpb-{lang}.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps({"validation": lang, **result}, ensure_ascii=False), flush=True)
    print(json.dumps({"status": "complete", "output_dir": out_dir.relative_to(ROOT).as_posix(), "checkpoint_sha256": inputs["checkpoint"]["sha256"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
