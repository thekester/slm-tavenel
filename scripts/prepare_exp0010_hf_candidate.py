"""Build a local, hash-indexed Hugging Face package for the EXP-0010 1M study.

This script prepares an artifact only. It never uploads to the Hub.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import save_file
from tokenizers import Tokenizer
from exp0010_integrity import validate_code_root, validate_hash, TOKENIZER_SHA256


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs/EXP-0010-1m-paired-replication-2026-10-22"
TOKENIZER = ROOT / "data/tokenizers/fsc-cc0-open-weights-v4/tokenizer.json"
CANDIDATE_ID = "slm-tavenel-1m-exp0010-s421101"
HF_REPO_ID = "athenista/slm-tavenel-1m-exp0010"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def copy_hashed(source: Path, target: Path, files: dict[str, str], root: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    files[target.relative_to(root).as_posix()] = sha256(target)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code-root", type=Path, required=True, help="clean training worktree")
    parser.add_argument("--out", type=Path, default=ROOT / "exports/slm-tavenel-1m-exp0010-candidate-ready")
    parser.add_argument("--model-card", type=Path,
                        default=ROOT / "docs/model-cards/slm-tavenel-1m-exp0010-candidate.md")
    args = parser.parse_args()
    code_root, out = args.code_root.resolve(), args.out.resolve()
    validate_code_root(code_root)
    validate_hash(TOKENIZER, TOKENIZER_SHA256)
    summary_path = RUNS / "sealed-test-bpb-results.json"
    if not summary_path.is_file():
        raise SystemExit("Locked sealed-test analysis must finish before packaging")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary["status"] != "complete" or summary["seeds"] != list(range(421101, 421111)):
        raise SystemExit("The complete predeclared ten-seed analysis is required")
    benchmark_path = RUNS / "selected-checkpoint-mc.json"
    comparison_path = RUNS / "benchmark-comparisons.json"
    data_replay_path = RUNS / "pipeline-reproduction-check.json"
    if not data_replay_path.is_file() or json.loads(data_replay_path.read_text(encoding="utf-8-sig"))["status"] != "outputs_match_exactly":
        raise SystemExit("Exact replay of the processed-data and decisions hashes is required before packaging")
    results_report_path = ROOT / "experiments/EXP-0010-results-2026-10-22.md"
    if not results_report_path.is_file():
        raise SystemExit("The complete human-readable results report is required before packaging")
    if not benchmark_path.is_file() or not comparison_path.is_file():
        raise SystemExit("Selected-checkpoint benchmark scores and same-harness comparisons must finish before packaging")
    if not args.model_card.is_file():
        raise FileNotFoundError(args.model_card)

    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Refusing to mix with an existing export directory: {out}")
    out.mkdir(parents=True, exist_ok=True)

    selected_run = RUNS / "seed-421101/gated"
    checkpoint_path = selected_run / "checkpoints/step_00000488.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    run_manifest = json.loads((selected_run / "run_manifest.json").read_text(encoding="utf-8"))
    if run_manifest["code"] != {"commit": "02396acd15ef9bcdc453bcd53aade40819e2f69f", "dirty": False}:
        raise ValueError("Selected release run was not trained from the predeclared clean code commit")
    if checkpoint["step"] != 488 or checkpoint["train_config"]["seed"] != 421101:
        raise ValueError("Selected release checkpoint differs from the prospectively selected run")
    tok = Tokenizer.from_file(str(TOKENIZER))
    eos_id, pad_id = tok.token_to_id("<|endoftext|>"), tok.token_to_id("<|pad|>")
    if eos_id is None or pad_id is None:
        raise ValueError("Training tokenizer is missing EOS or PAD")

    files: dict[str, str] = {}
    source_manifest = out / "training_record/source-record-manifest.jsonl"
    source_manifest.parent.mkdir(parents=True, exist_ok=True)
    import subprocess
    import sys
    subprocess.run([
        sys.executable, str(ROOT / "scripts/export_open_weight_data_manifest.py"),
        "--documents", str(ROOT / "data/processed/fsc-cc0-open-weights-v4/documents.jsonl"),
        "--splits", str(ROOT / "data/processed/fsc-cc0-open-weights-v4/splits"),
        "--out", str(source_manifest),
    ], check=True)
    files[source_manifest.relative_to(out).as_posix()] = sha256(source_manifest)

    selected_state = checkpoint["model"]
    if checkpoint["model_config"].get("tie_embeddings"):
        selected_state = {k: v for k, v in selected_state.items() if k != "lm_head.weight"}
    cpu_state = {k: v.detach().cpu().float().contiguous() for k, v in selected_state.items()}
    save_file(cpu_state, str(out / "model.safetensors"), metadata={"format": "pt"})
    files["model.safetensors"] = sha256(out / "model.safetensors")

    # Include all paired final weights and logs so the reported seed-level result can be inspected.
    for seed in range(421101, 421111):
        copy_hashed(RUNS / f"seed-{seed}/pair-record.json",
                    out / "training_record/replicates" / f"seed-{seed}" / "pair-record.json", files, out)
        for arm in ("gated", "ungated"):
            run_dir = RUNS / f"seed-{seed}/{arm}"
            run_manifest_path = run_dir / "run_manifest.json"
            state = torch.load(run_dir / "checkpoints/step_00000488.pt", map_location="cpu", weights_only=False)
            if state["step"] != 488:
                raise ValueError(f"Incomplete checkpoint: {run_dir}")
            model_state = state["model"]
            if state["model_config"].get("tie_embeddings"):
                model_state = {k: v for k, v in model_state.items() if k != "lm_head.weight"}
            rel = Path("training_record/replicates") / f"seed-{seed}" / f"{arm}.safetensors"
            dest = out / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            save_file({k: v.detach().cpu().float().contiguous() for k, v in model_state.items()}, str(dest),
                      metadata={"format": "pt", "seed": str(seed), "arm": arm})
            files[rel.as_posix()] = sha256(dest)
            for filename in ("run_manifest.json", "metrics.jsonl", "eval.jsonl", "observability.jsonl"):
                copy_hashed(run_dir / filename, out / "training_record/replicates" / f"seed-{seed}" / arm / filename, files, out)
            copy_hashed(RUNS / f"seed-{seed}/{arm}-train.yaml",
                        out / "training_record/replicates" / f"seed-{seed}" / arm / "train-config.yaml", files, out)

    shared_files = [
        (TOKENIZER, out / "tokenizer.json"),
        (args.model_card.resolve(), out / "README.md"),
        (ROOT / "LICENSE", out / "LICENSE"),
        (ROOT / "LICENSE", out / "CODE-LICENSE.txt"),
        (code_root / "src/slm/hf_config.py", out / "hf_config.py"),
        (code_root / "src/slm/hf_model.py", out / "hf_model.py"),
        (code_root / "src/slm/__main__.py", out / "training_record/src/slm/__main__.py"),
        (code_root / "src/slm/cli.py", out / "training_record/src/slm/cli.py"),
        (code_root / "src/slm/config.py", out / "training_record/src/slm/config.py"),
        (code_root / "src/slm/train.py", out / "training_record/src/slm/train.py"),
        (code_root / "src/slm/model.py", out / "training_record/src/slm/model.py"),
        (code_root / "src/slm/tokenizer.py", out / "training_record/src/slm/tokenizer.py"),
        (code_root / "src/slm/optim.py", out / "training_record/src/slm/optim.py"),
        (code_root / "src/slm/runlog.py", out / "training_record/src/slm/runlog.py"),
        (code_root / "src/slm/data/__init__.py", out / "training_record/src/slm/data/__init__.py"),
        (code_root / "src/slm/data/pipeline.py", out / "training_record/src/slm/data/pipeline.py"),
        (code_root / "src/slm/data/filters.py", out / "training_record/src/slm/data/filters.py"),
        (code_root / "src/slm/data/registry.py", out / "training_record/src/slm/data/registry.py"),
        (code_root / "src/slm/data/pii.py", out / "training_record/src/slm/data/pii.py"),
        (code_root / "src/slm/data/dedup.py", out / "training_record/src/slm/data/dedup.py"),
        (code_root / "src/slm/data/contamination.py", out / "training_record/src/slm/data/contamination.py"),
        (code_root / "src/slm/data/shards.py", out / "training_record/src/slm/data/shards.py"),
        (selected_run / "run_manifest.json", out / "training_record/selected-run-manifest.json"),
        (selected_run / "metrics.jsonl", out / "training_record/selected-metrics.jsonl"),
        (selected_run / "eval.jsonl", out / "training_record/selected-eval.jsonl"),
        (selected_run / "observability.jsonl", out / "training_record/selected-observability.jsonl"),
        (ROOT / "configs/model/slm-1m-gated.yaml", out / "training_record/gated-model-config.yaml"),
        (ROOT / "configs/model/slm-1m-ungated-match.yaml", out / "training_record/ungated-model-config.yaml"),
        (ROOT / "configs/data/pipeline-fsc-cc0-open-weights-v4.yaml", out / "training_record/data-pipeline.yaml"),
        (ROOT / "configs/data/sources.yaml", out / "training_record/source-registry.yaml"),
        (ROOT / "data/corpus-audits/fsc-cc0-hal-complete-audit.json", out / "training_record/hal-license-audit.json"),
        (ROOT / "data/processed/fsc-cc0-open-weights-v4/manifest.json", out / "training_record/data-manifest.json"),
        (ROOT / "data/processed/fsc-cc0-open-weights-v4/stats.json", out / "training_record/data-stats.json"),
        (ROOT / "data/processed/fsc-cc0-open-weights-v4/decisions.jsonl", out / "training_record/data-decisions.jsonl"),
        (ROOT / "data/processed/fsc-cc0-open-weights-v4/splits/split_manifest.json", out / "training_record/split-manifest.json"),
        (ROOT / "data/tokenizers/fsc-cc0-open-weights-v4/manifest.json", out / "training_record/tokenizer-manifest.json"),
        (ROOT / "data/tokens/fsc-cc0-open-weights-v4/training/manifest.json", out / "training_record/token-manifest.json"),
        (ROOT / "data/tokens/fsc-cc0-open-weights-v4/training/provenance.json", out / "training_record/data-provenance.json"),
        (ROOT / "data/corpus-audits/v4-privacy-screen-report.json", out / "training_record/privacy-screen-report.json"),
        (ROOT / "docs/model-cards/slm-tavenel-1m-exp0010-data-card.md", out / "training_record/data-card.md"),
        (data_replay_path, out / "training_record/data-preprocessing-reproduction-check.json"),
        (RUNS / "pipeline-reproduction-check-config.yaml", out / "training_record/data-preprocessing-replay-config.yaml"),
        (RUNS / "pipeline-reproduction-manifest.json", out / "training_record/data-preprocessing-replay-manifest.json"),
        (RUNS / "pipeline-reproduction-stats.json", out / "training_record/data-preprocessing-replay-stats.json"),
        (ROOT / "scripts/audit_fsc_cc0_hal.py", out / "training_record/scripts/audit_fsc_cc0_hal.py"),
        (ROOT / "scripts/fetch_french_science_commons.py", out / "training_record/scripts/fetch_french_science_commons.py"),
        (ROOT / "scripts/select_french_science_commons.py", out / "training_record/scripts/select_french_science_commons.py"),
        (ROOT / "scripts/audit_french_science_documents.py", out / "training_record/scripts/audit_french_science_documents.py"),
        (ROOT / "scripts/aggregate_french_science_pages.py", out / "training_record/scripts/aggregate_french_science_pages.py"),
        (ROOT / "scripts/split_open_weight_corpus.py", out / "training_record/scripts/split_open_weight_corpus.py"),
        (ROOT / "scripts/encode_open_weight_splits.py", out / "training_record/scripts/encode_open_weight_splits.py"),
        (ROOT / "scripts/export_open_weight_data_manifest.py", out / "training_record/scripts/export_open_weight_data_manifest.py"),
        (ROOT / "scripts/run_exp0010_paired_replicates.py", out / "training_record/scripts/run_exp0010_paired_replicates.py"),
        (ROOT / "scripts/analyze_exp0010_paired_replicates.py", out / "training_record/scripts/analyze_exp0010_paired_replicates.py"),
        (ROOT / "scripts/exp0010_integrity.py", out / "training_record/scripts/exp0010_integrity.py"),
        (ROOT / "scripts/prepare_exp0010_hf_candidate.py", out / "training_record/scripts/prepare_exp0010_hf_candidate.py"),
        (ROOT / "scripts/evaluate_exp0010_release_checkpoint.py", out / "training_record/scripts/evaluate_exp0010_release_checkpoint.py"),
        (ROOT / "scripts/compare_exp0010_benchmarks.py", out / "training_record/scripts/compare_exp0010_benchmarks.py"),
        (ROOT / "scripts/render_exp0010_benchmark_chart.py", out / "training_record/scripts/render_exp0010_benchmark_chart.py"),
        (ROOT / "scripts/render_exp0010_model_card_figures.py", out / "training_record/scripts/render_exp0010_model_card_figures.py"),
        (ROOT / "scripts/write_exp0010_hf_model_card.py", out / "training_record/scripts/write_exp0010_hf_model_card.py"),
        (ROOT / "scripts/write_exp0010_results_report.py", out / "training_record/scripts/write_exp0010_results_report.py"),
        (ROOT / "scripts/verify_exp0010_hf_candidate.py", out / "training_record/scripts/verify_exp0010_hf_candidate.py"),
        (ROOT / "scripts/subset_mc_results.py", out / "training_record/scripts/subset_mc_results.py"),
        (ROOT / "scripts/compare_mc.py", out / "training_record/scripts/compare_mc.py"),
        (code_root / "src/slm/benchmarks.py", out / "training_record/benchmarks.py"),
        (ROOT / "experiments/EXP-0010-1m-paired-replication-2026-10-22.md", out / "training_record/protocol.md"),
        (ROOT / "experiments/EXP-0010-amendment-2026-10-22.md", out / "training_record/protocol-amendment.md"),
        (results_report_path, out / "training_record/results-report.md"),
        (summary_path, out / "training_record/sealed-test-bpb-results.json"),
        (benchmark_path, out / "training_record/selected-checkpoint-mc.json"),
        (comparison_path, out / "training_record/benchmark-comparisons.json"),
        (ROOT / "docs/exp0010-benchmark-comparison.svg", out / "exp0010-benchmark-comparison.svg"),
        (ROOT / "docs/exp0010-data-splits.svg", out / "exp0010-data-splits.svg"),
        (ROOT / "docs/exp0010-training-curves.svg", out / "exp0010-training-curves.svg"),
        (ROOT / "docs/exp0010-paired-seed-effects.svg", out / "exp0010-paired-seed-effects.svg"),
        (RUNS / "driver-record.json", out / "training_record/replication-driver-record.json"),
        (RUNS / "source-record-manifest.jsonl", out / "training_record/source-record-manifest.jsonl"),
    ]
    for source, dest in shared_files:
        copy_hashed(source, dest, files, out)
    for reference in ("tiny1m", "pythia-14m", "pythia-31m", "smollm2-135m"):
        full_path = ROOT / f"runs/reference/{reference}/fsc-pilot-v1-mc.json"
        if full_path.is_file():
            copy_hashed(full_path,
                        out / f"training_record/references/{reference}/four-task-mc.json", files, out)
        copy_hashed(ROOT / f"runs/fsc-additional/{reference}/fsc-pilot-v2-mc.json",
                    out / f"training_record/references/{reference}/additional-task-mc.json", files, out)
    for size in ("10m",):
        copy_hashed(ROOT / f"runs/fsc-scaling/{size}-gated-s1234/analysis/fsc-pilot-v1-mc.json",
                    out / f"training_record/references/slm-tavenel-{size}/four-task-mc.json", files, out)
        copy_hashed(ROOT / f"runs/fsc-additional/slm-tavenel-{size}/fsc-pilot-v2-mc.json",
                    out / f"training_record/references/slm-tavenel-{size}/additional-task-mc.json", files, out)

    write_json(out / "config.json", {
        "model_type": "slm_tavenel", "architectures": ["SlmTavenelForCausalLM"],
        "auto_map": {"AutoConfig": "hf_config.SlmTavenelConfig", "AutoModelForCausalLM": "hf_model.SlmTavenelForCausalLM"},
        **checkpoint["model_config"], "max_position_embeddings": checkpoint["model_config"]["max_seq_len"],
        "tie_word_embeddings": checkpoint["model_config"]["tie_embeddings"], "torch_dtype": "float32",
        "use_cache": False, "eos_token_id": eos_id, "bos_token_id": None, "pad_token_id": pad_id,
    })
    write_json(out / "generation_config.json", {"eos_token_id": eos_id, "pad_token_id": pad_id, "use_cache": False})
    write_json(out / "tokenizer_config.json", {"tokenizer_class": "PreTrainedTokenizerFast", "eos_token": "<|endoftext|>",
        "pad_token": "<|pad|>", "bos_token": None, "unk_token": None, "model_max_length": checkpoint["model_config"]["max_seq_len"],
        "clean_up_tokenization_spaces": False, "add_bos_token": False, "model_input_names": ["input_ids", "attention_mask"]})
    write_json(out / "special_tokens_map.json", {"eos_token": "<|endoftext|>", "pad_token": "<|pad|>"})
    for filename in ("config.json", "generation_config.json", "tokenizer_config.json", "special_tokens_map.json"):
        files[filename] = sha256(out / filename)

    (out / "requirements.txt").write_text(
        "torch==2.10.0\ntransformers==4.57.6\ntokenizers==0.22.2\nsafetensors==0.8.0\n",
        encoding="utf-8")
    files["requirements.txt"] = sha256(out / "requirements.txt")
    (out / "requirements-training.txt").write_text(
        "# Recorded training environment: Python 3.11.9, PyTorch CUDA 13.0, RTX 4060 Laptop GPU.\n"
        "--extra-index-url https://download.pytorch.org/whl/cu130\n"
        "torch==2.10.0+cu130\nnumpy==2.4.4\nPyYAML==6.0.2\n"
        "tokenizers==0.22.2\nsafetensors==0.8.0\ntransformers==4.57.6\n",
        encoding="utf-8")
    files["requirements-training.txt"] = sha256(out / "requirements-training.txt")
    environment = {**run_manifest["environment"], "python": run_manifest["environment"]["python"]}
    write_json(out / "training_record/environment.json", environment)
    files["training_record/environment.json"] = sha256(out / "training_record/environment.json")
    (out / "NOTICE.md").write_text(
        "# Source notice\n\n"
        "The model was trained on a filtered English/French subset of [PleIAs/French-Science-Commons]"
        "(https://huggingface.co/datasets/PleIAs/French-Science-Commons), selected from records labeled CC0 and"
        " cross-checked against HAL license metadata. The source article text is not included. The exact record"
        " identifiers, URIs, declared license values, split assignments and hashes are in `training_record/`."
        " The metadata audit does not prove that each depositor had authority to apply CC0. This notice is for"
        " provenance and is not a legal clearance statement.\n",
        encoding="utf-8")
    files["NOTICE.md"] = sha256(out / "NOTICE.md")

    provenance = {
        "candidate_id": CANDIDATE_ID, "hub_repo_id": HF_REPO_ID,
        "status": "local experimental open-weight candidate; not uploaded",
        "weight_license": "MIT", "code_license": "MIT", "data_information_license": "MIT",
        "maintainer_authorized_local_candidate_preparation": True,
        "uploaded": False,
        "publication_status": "local candidate prepared; not uploaded; source-rights and manual privacy reviews unsigned",
        "source_rights_review_signed": False, "manual_privacy_review_signed": False,
        "training": {
            "selected_seed": 421101, "paired_seeds": list(range(421101, 421111)),
            "selected_run_id": "runs/EXP-0010-1m-paired-replication-2026-10-22/seed-421101/gated",
            "optimizer_state_included": False,
            "checkpoint_export": "model parameters converted from the final .pt checkpoint to float32 safetensors; original .pt files are not included",
            "steps": checkpoint["step"], "tokens": checkpoint["train_config"]["max_steps"] * checkpoint["train_config"]["micro_batch_size"] * checkpoint["train_config"]["grad_accum_steps"] * checkpoint["train_config"]["seq_len"],
            "parameters": run_manifest["params"], "code": run_manifest["code"], "environment": run_manifest["environment"],
            "replication_decision": summary["decision"], "primary_effect_bpb": summary["estimate_bpb"],
            "primary_effect_ci95_bpb": summary["hierarchical_bootstrap_95_percent_interval_bpb"],
            "sealed_test_scored": True,
            "multiple_choice_benchmarks": "post-hoc descriptive only; full item-level records are in training_record/selected-checkpoint-mc.json",
        },
        "data": {
            "dataset": "PleIAs/French-Science-Commons", "revision": "bfa114c09d7c28dcd0a8a47fe445c64b4a3c6156",
            "subset": "french_open_science", "retained_documents": 1229,
            "license_metadata": "CC0 labels matched across all versions returned by paginated HAL API; metadata match is not independent legal clearance",
            "training_text_in_bundle": False,
            "source_record_manifest_sha256": files["training_record/source-record-manifest.jsonl"],
            "audit_report_sha256": sha256(ROOT / "data/corpus-audits/fsc-cc0-hal-complete-audit.json"),
        },
        "checkpoint_sha256": sha256(checkpoint_path), "files_sha256": files,
    }
    write_json(out / "provenance.json", provenance)
    print(json.dumps({"bundle": str(out), "candidate_id": CANDIDATE_ID, "hub_repo_id": HF_REPO_ID,
        "publication_status": provenance["publication_status"], "parameters": run_manifest["params"],
        "replicates": 10, "files": len(files), "source_review_signed": False, "privacy_review_signed": False}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
