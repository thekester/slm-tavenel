"""Package the privacy-risk-filtered CC0 v2 1M checkpoint as a local MIT-licensed HF candidate.

This is a packaging step only. It never uploads to the Hub, and the generated card keeps the
privacy/source review status explicit.
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


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/open-weight-candidate/1m-gated-s20261022-v2-pilot"
CHECKPOINT = RUN / "checkpoints/step_00000805.pt"
TOKENIZER = ROOT / "data/tokenizer/fsc-cc0-open-weights-v2/tokenizer.json"
CARD = ROOT / "docs/model-cards/slm-tavenel-1m-cc0-v2-candidate.md"
EOS = "<|endoftext|>"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def copy_record(source: Path, target: Path, files: dict[str, str], bundle_root: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    files[target.relative_to(bundle_root).as_posix()] = sha256(target)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "exports/slm-tavenel-1m-cc0-v2-candidate")
    args = parser.parse_args()
    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        marker = out / "provenance.json"
        if not marker.is_file() or json.loads(marker.read_text(encoding="utf-8")).get("candidate_id") != "slm-tavenel-1m-cc0-v2-pilot-s20261022":
            raise FileExistsError(f"Refusing to overwrite unrelated export directory: {out}")
    out.mkdir(parents=True, exist_ok=True)

    run_manifest = json.loads((RUN / "run_manifest.json").read_text(encoding="utf-8"))
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    if checkpoint["step"] != run_manifest["train_config"]["max_steps"]:
        raise ValueError("Checkpoint step differs from the completed training manifest")
    tokenizer = Tokenizer.from_file(str(TOKENIZER))
    eos_id = tokenizer.token_to_id(EOS)
    pad_id = tokenizer.token_to_id("<|pad|>")
    if eos_id is None or pad_id is None:
        raise ValueError("Training tokenizer is missing EOS or PAD")

    files: dict[str, str] = {}
    source_state = checkpoint["model"]
    if run_manifest["model_config"].get("tie_embeddings"):
        source_state = {key: value for key, value in source_state.items() if key != "lm_head.weight"}
    state = {key: value.detach().to(device="cpu", dtype=torch.float32).contiguous()
             for key, value in source_state.items()}
    save_file(state, str(out / "model.safetensors"), metadata={"format": "pt"})
    files["model.safetensors"] = sha256(out / "model.safetensors")

    for src, dst in (
        (TOKENIZER, out / "tokenizer.json"),
        (CARD, out / "README.md"),
        (ROOT / "LICENSE", out / "LICENSE"),
        (ROOT / "LICENSE", out / "CODE-LICENSE.txt"),
        (ROOT / "src/slm/hf_config.py", out / "hf_config.py"),
        (ROOT / "src/slm/hf_model.py", out / "hf_model.py"),
        (RUN / "run_manifest.json", out / "training_record/run_manifest.json"),
        (RUN / "metrics.jsonl", out / "training_record/metrics.jsonl"),
        (RUN / "eval.jsonl", out / "training_record/eval.jsonl"),
        (RUN / "observability.jsonl", out / "training_record/observability.jsonl"),
        (ROOT / "configs/train/fsc-cc0-open-weight-1m-v2-pilot.yaml", out / "training_record/train-config.yaml"),
        (ROOT / "configs/model/slm-1m-gated.yaml", out / "training_record/model-config.yaml"),
        (ROOT / "configs/data/pipeline-fsc-cc0-open-weights-v2.yaml", out / "training_record/data-pipeline.yaml"),
        (ROOT / "configs/data/sources.yaml", out / "training_record/source-registry.yaml"),
        (ROOT / "data/corpus-audits/fsc-cc0-hal-license-audit.json", out / "training_record/hal-license-audit.json"),
        (ROOT / "data/processed/fsc-cc0-open-weights-v2/manifest.json", out / "training_record/data-manifest.json"),
        (ROOT / "data/processed/fsc-cc0-open-weights-v2/stats.json", out / "training_record/data-stats.json"),
        (ROOT / "data/processed/fsc-cc0-open-weights-v2/decisions.jsonl", out / "training_record/data-decisions.jsonl"),
        (ROOT / "data/processed/fsc-cc0-open-weights-v2/splits/split_manifest.json", out / "training_record/split-manifest.json"),
        (ROOT / "data/tokenizer/fsc-cc0-open-weights-v2/manifest.json", out / "training_record/tokenizer-manifest.json"),
        (ROOT / "data/tokens/fsc-cc0-open-weights-v2/training/manifest.json", out / "training_record/token-manifest.json"),
        (ROOT / "data/tokens/fsc-cc0-open-weights-v2/training/provenance.json", out / "training_record/data-provenance.json"),
        (ROOT / "docs/reviews/OPEN-WEIGHTS-1M-2026-10-21.md", out / "training_record/open-weight-review.md"),
        (ROOT / "scripts/audit_fsc_cc0_hal.py", out / "training_record/audit_fsc_cc0_hal.py"),
        (ROOT / "scripts/split_open_weight_corpus.py", out / "training_record/split_open_weight_corpus.py"),
        (ROOT / "scripts/encode_open_weight_splits.py", out / "training_record/encode_open_weight_splits.py"),
        (ROOT / "scripts/write_open_weight_v2_provenance.py", out / "training_record/write_open_weight_v2_provenance.py"),
        (ROOT / "src/slm/data/pipeline.py", out / "training_record/pipeline.py"),
        (ROOT / "scripts/prepare_open_weight_v2_candidate.py", out / "training_record/prepare_open_weight_v2_candidate.py"),
        (ROOT / "scripts/verify_open_weight_v2_candidate.py", out / "training_record/verify_open_weight_v2_candidate.py"),
        (ROOT / "scripts/subset_mc_results.py", out / "training_record/subset_mc_results.py"),
        (ROOT / "scripts/compare_mc.py", out / "training_record/compare_mc.py"),
        (ROOT / "src/slm/benchmarks.py", out / "training_record/benchmarks.py"),
        (ROOT / "runs/open-weight-candidate/1m-gated-s20261022-v2-pilot/analysis/mc-six-benchmarks.json", out / "training_record/mc-six-benchmarks.json"),
        (ROOT / "runs/open-weight-candidate/1m-gated-s20261022-v2-pilot/analysis/mc-four-shared.json", out / "training_record/mc-four-shared.json"),
        (ROOT / "runs/open-weight-candidate/1m-gated-s20261022-v2-pilot/analysis/mc-two-shared.json", out / "training_record/mc-two-shared.json"),
        (ROOT / "runs/open-weight-candidate/1m-gated-s20261022-v2-pilot/analysis/paired-tiny1m.json", out / "training_record/paired-tiny1m.json"),
        (ROOT / "runs/open-weight-candidate/1m-gated-s20261022-v2-pilot/analysis/paired-slm-tavenel-1m-fsc.json", out / "training_record/paired-slm-tavenel-1m-fsc.json"),
        (ROOT / "runs/open-weight-candidate/1m-gated-s20261022-v2-pilot/analysis/paired-slm-tavenel-10m.json", out / "training_record/paired-slm-tavenel-10m.json"),
        (ROOT / "runs/open-weight-candidate/1m-gated-s20261022-v2-pilot/analysis/paired-pythia-14m.json", out / "training_record/paired-pythia-14m.json"),
        (ROOT / "runs/open-weight-candidate/1m-gated-s20261022-v2-pilot/analysis/paired-smollm2-135m.json", out / "training_record/paired-smollm2-135m.json"),
        (ROOT / "docs/open-weight-1m-benchmark-comparison.svg", out / "benchmark-comparison.svg"),
    ):
        copy_record(src, dst, files, out)

    (out / "requirements.txt").write_text(
        "torch>=2.3\ntransformers==4.57.6\ntokenizers==0.22.2\nsafetensors==0.8.0\n", encoding="utf-8")
    files["requirements.txt"] = sha256(out / "requirements.txt")
    (out / "NOTICE.md").write_text(
        "# Source notice\n\n"
        "This model was trained on a filtered English/French CC0-only subset of "
        "[PleIAs/French-Science-Commons](https://huggingface.co/datasets/PleIAs/French-Science-Commons), "
        "cross-checked against HAL record license metadata. Source record identifiers and URIs, "
        "the selection and processing recipe, and hashes are in `training_record/`. The records "
        "remain attributable to their respective contributors; no author names were inferred. "
        "CC0 has no attribution requirement, but this notice is provided for provenance. The "
        "training text itself is not included.\n",
        encoding="utf-8")
    files["NOTICE.md"] = sha256(out / "NOTICE.md")

    model_config = run_manifest["model_config"]
    hf_config = {
        "model_type": "slm_tavenel",
        "architectures": ["SlmTavenelForCausalLM"],
        "auto_map": {"AutoConfig": "hf_config.SlmTavenelConfig", "AutoModelForCausalLM": "hf_model.SlmTavenelForCausalLM"},
        **model_config,
        "max_position_embeddings": model_config["max_seq_len"],
        "tie_word_embeddings": model_config["tie_embeddings"],
        "torch_dtype": "float32",
        "use_cache": False,
        "eos_token_id": eos_id,
        "bos_token_id": None,
        "pad_token_id": pad_id,
    }
    write_json(out / "config.json", hf_config)
    write_json(out / "generation_config.json", {"eos_token_id": eos_id, "pad_token_id": pad_id, "use_cache": False})
    write_json(out / "tokenizer_config.json", {"tokenizer_class": "PreTrainedTokenizerFast", "eos_token": EOS,
        "pad_token": "<|pad|>", "bos_token": None, "unk_token": None,
        "model_max_length": model_config["max_seq_len"], "clean_up_tokenization_spaces": False,
        "add_bos_token": False, "model_input_names": ["input_ids", "attention_mask"]})
    write_json(out / "special_tokens_map.json", {"eos_token": EOS, "pad_token": "<|pad|>"})
    for filename in ("config.json", "generation_config.json", "tokenizer_config.json", "special_tokens_map.json"):
        files[filename] = sha256(out / filename)

    provenance = {
        "candidate_id": "slm-tavenel-1m-cc0-v2-pilot-s20261022",
        "status": "local candidate; no public upload; source sign-off and manual privacy review pending",
        "weight_license": "MIT",
        "code_license": "MIT",
        "public_release_approved": False,
        "source_review_signed": False,
        "manual_privacy_review_signed": False,
        "training": {
            "run_id": RUN.relative_to(ROOT).as_posix(),
            "steps": checkpoint["step"], "seed": run_manifest["train_config"]["seed"],
            "tokens": run_manifest["train_config"]["max_steps"] * run_manifest["train_config"]["micro_batch_size"] * run_manifest["train_config"]["grad_accum_steps"] * run_manifest["train_config"]["seq_len"],
            "parameters": run_manifest["params"], "config_hash": run_manifest["model_config_hash"],
            "train_config_hash": run_manifest["train_config_hash"], "code": run_manifest["code"],
            "environment": run_manifest["environment"],
            "final_train_loss": 3.363579034805298, "final_dev_loss": 3.43765,
            "sealed_test_scored": False,
        },
        "data": {
            "dataset": "PleIAs/French-Science-Commons", "revision": "bfa114c09d7c28dcd0a8a47fe445c64b4a3c6156",
            "subset": "french_open_science", "license_selection": "CC0 rows matched to CC0 on every HAL API version; unresolved IDs excluded",
            "privacy_source_review_signed": False, "manual_privacy_review_signed": False,
            "audit_report_sha256": sha256(ROOT / "data/corpus-audits/fsc-cc0-hal-license-audit.json"),
            "pipeline_manifest_sha256": sha256(ROOT / "data/processed/fsc-cc0-open-weights-v2/manifest.json"),
            "split_manifest_sha256": sha256(ROOT / "data/processed/fsc-cc0-open-weights-v2/splits/split_manifest.json"),
            "tokenizer_sha256": sha256(TOKENIZER),
            "token_manifest_sha256": sha256(ROOT / "data/tokens/fsc-cc0-open-weights-v2/training/manifest.json"),
            "train_tokens": 26_387_903, "dev_tokens": 1_473_762, "sealed_test_tokens": 3_399_878,
            "training_text_in_bundle": False,
        },
        "checkpoint_sha256": sha256(CHECKPOINT),
        "files_sha256": files,
    }
    write_json(out / "provenance.json", provenance)
    print(json.dumps({"bundle": str(out), "candidate_id": provenance["candidate_id"],
                      "weight_license": provenance["weight_license"], "public_release_approved": False,
        "source_review_signed": False,
        "manual_privacy_review_signed": False,
                      "parameter_count": run_manifest["params"], "checkpoint_sha256": provenance["checkpoint_sha256"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
