"""Package the EXP-0032 25M HPLT-domain checkpoint for Hugging Face.

This exports only model weights, code, metadata, aggregate evaluation records,
and the reproducible experiment report. Source text, token shards, benchmark
prompts, and raw per-example scores are intentionally excluded.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file
import transformers
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from slm.config import ModelConfig
from slm.model import SLM


OUT = ROOT / "exports/slm-tavenel-25m-exp0032-hplt-10pct"
RUNS = {
    "control": ROOT / "runs/EXP-0032-25m-fsc-control-s20261028",
    "treatment": ROOT / "runs/EXP-0032-25m-hplt-10pct-s20261028",
}
TREATMENT_CHECKPOINT = RUNS["treatment"] / "checkpoints/step_00001500.pt"
TOKENIZER = ROOT / "exports/slm-tavenel-25m-exp0009-candidate/tokenizer.json"
BASE_PACKAGE = ROOT / "exports/slm-tavenel-25m-exp0009-candidate"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    # The destination is a dedicated, experiment-specific export directory.
    # Re-running intentionally refreshes its generated files in place.
    if not TREATMENT_CHECKPOINT.is_file() or not TOKENIZER.is_file():
        raise FileNotFoundError("EXP-0032 checkpoint or pinned tokenizer is missing")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "training_record").mkdir(parents=True, exist_ok=True)

    # Keep the exact custom architecture and tokenizer used by the published
    # EXP-0009 parent. The state-dict keys are shared with the native SLM.
    for name in ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "hf_config.py", "hf_model.py"):
        shutil.copyfile(BASE_PACKAGE / name, OUT / name)
    shutil.copyfile(ROOT / "LICENSE", OUT / "LICENSE")
    shutil.copyfile(BASE_PACKAGE / "CODE-LICENSE.txt", OUT / "CODE-LICENSE.txt")
    shutil.copyfile(ROOT / "docs/slm-tavenel-architecture.svg", OUT / "architecture.svg")
    shutil.copyfile(ROOT / "docs/scaling-benchmark-comparison.svg", OUT / "scaling-benchmark-comparison.svg")
    shutil.copyfile(ROOT / "experiments/EXP-0032-hplt-bpb-deltas.svg", OUT / "hplt-bpb-deltas.svg")
    shutil.copyfile(ROOT / "experiments/EXP-0032-25m-hplt-10pct-fresh-seed-2026-10-27.md", OUT / "training_record/EXP-0032-results.md")
    shutil.copyfile(ROOT / "docs/model-cards/slm-tavenel-25m-exp0032-hplt-10pct.md", OUT / "README.md")
    shutil.copyfile(ROOT / "configs/train/exp0032-25m-hplt-10pct-s20261028.yaml", OUT / "training_record/train-config.yaml")
    shutil.copyfile(Path(__file__).resolve(), OUT / "training_record/prepare_exp0032_25m_hf_candidate.py")

    checkpoint = torch.load(TREATMENT_CHECKPOINT, map_location="cpu", weights_only=False)
    if int(checkpoint.get("step", -1)) != 1500:
        raise ValueError("Expected the completed step-1500 treatment checkpoint")
    state = checkpoint["model"]
    if checkpoint["model_config"].get("tie_embeddings"):
        state = {key: value for key, value in state.items() if key != "lm_head.weight"}
    state = {key: value.detach().cpu().contiguous() for key, value in state.items()}
    save_file(state, str(OUT / "model.safetensors"), metadata={"format": "pt"})

    for arm, run in RUNS.items():
        record = OUT / "training_record" / arm
        record.mkdir(parents=True, exist_ok=True)
        for name in ("metrics.jsonl", "eval.jsonl"):
            shutil.copyfile(run / name, record / name)
        manifest = json.loads((run / "run_manifest.json").read_text(encoding="utf-8"))
        # Exclude machine-local paths and environment identifiers; retain the
        # settings needed to understand the run and the source commit status.
        summary = {
            "kind": manifest.get("kind"),
            "code_commit": manifest.get("code", {}).get("commit"),
            "worktree_was_dirty": manifest.get("code", {}).get("dirty"),
            "parameters": manifest.get("params"),
            "model_config": manifest.get("model_config"),
            "train_config": manifest.get("train_config"),
            "initialized_from_checkpoint_sha256": manifest.get("initialized_from", {}).get("sha256"),
            "tokenizer_sha256": manifest.get("tokenizer_sha256"),
            "data_manifest_sha256": manifest.get("data", {}).get("manifest_sha256"),
        }
        write_json(record / "run-manifest-summary.json", summary)

    mix_manifest_path = ROOT / "data/tokens/exp0031-25m-hplt-10pct-fsc-tokenizer/manifest.json"
    mix_manifest = json.loads(mix_manifest_path.read_text(encoding="utf-8"))
    # The manifest is metadata and hashes only; do not copy its token shards.
    write_json(OUT / "training_record/token-mixture-manifest.json", mix_manifest)
    for name, source in (
        ("paired-development-bpb.json", ROOT / "runs/EXP-0032-evaluations/paired-development-bpb.json"),
        ("paired-hplt-test-bpb.json", ROOT / "runs/EXP-0032-evaluations/paired-hplt-test.json"),
    ):
        shutil.copyfile(source, OUT / "training_record" / name)

    checkpoint_hash = digest(TREATMENT_CHECKPOINT)
    tokenizer_hash = digest(TOKENIZER)
    native_model = SLM(ModelConfig(**checkpoint["model_config"]))
    native_model.load_state_dict(checkpoint["model"])
    native_model.eval()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    reference_tokenizer = Tokenizer.from_file(str(TOKENIZER))
    hf_tokenizer = AutoTokenizer.from_pretrained(OUT, trust_remote_code=True, local_files_only=True)
    hf_model = AutoModelForCausalLM.from_pretrained(OUT, trust_remote_code=True, local_files_only=True).eval()
    prompt = "La recherche scientifique. Scientific research."
    token_ids = reference_tokenizer.encode(prompt).ids
    if hf_tokenizer.encode(prompt, add_special_tokens=False) != token_ids:
        raise ValueError("Packaged tokenizer IDs differ from the training tokenizer")
    eos_id = reference_tokenizer.token_to_id("<|endoftext|>")
    input_ids = torch.tensor([[eos_id, *token_ids]], dtype=torch.long)
    with torch.inference_mode():
        native_logits = native_model(input_ids)[0]
        hf_logits = hf_model(input_ids=input_ids).logits
        generated = hf_model.generate(input_ids=input_ids, max_new_tokens=2, do_sample=False)
    max_logit_error = float((native_logits - hf_logits).abs().max())
    if max_logit_error > 1e-6 or generated.shape[1] < input_ids.shape[1] + 1:
        raise ValueError(f"Native/HF inference check failed (max logit error {max_logit_error})")
    parity = {
        "schema": "slm.exp0032_hf_inference_parity.v1",
        "status": "passed",
        "scope": "fixed CPU probe; tokenizer IDs, native/HF logits, and greedy generation",
        "probe_text": prompt,
        "tokenizer_ids_equal": True,
        "native_hf_max_abs_logit_error": max_logit_error,
        "native_hf_mean_abs_logit_error": float((native_logits - hf_logits).abs().mean()),
        "max_abs_tolerance": 1e-6,
        "generation_new_tokens_returned": int(generated.shape[1] - input_ids.shape[1]),
        "parameter_count": sum(parameter.numel() for parameter in hf_model.parameters()),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "native_checkpoint_sha256": checkpoint_hash,
        "tokenizer_sha256": tokenizer_hash,
        "limitations": ["one fixed CPU probe; not a broad numerical or generation-quality evaluation", "does not establish scientific generalization, rights clearance, or privacy approval"],
    }
    write_json(OUT / "inference-parity-report.json", parity)
    provenance = {
        "schema": "slm-tavenel.exp0032-hf-provenance.v1",
        "model_id": "athenista/slm-tavenel-25m-exp0032-hplt-10pct",
        "experiment": "EXP-0032",
        "status": "experimental HPLT-domain-adapted research checkpoint; rights and human privacy reviews unresolved",
        "parameters": 25723648,
        "checkpoint_step": 1500,
        "continuation_tokens": 49152000,
        "training_seed": 20261028,
        "training_code_commit": "5d257a8d0d46ee773a8b6d1656d474b23adde19f",
        "training_worktree_was_dirty": True,
        "parent_model_id": "athenista/slm-tavenel-25m-exp0009",
        "parent_checkpoint_sha256": "ac7cb9c5c418bfadd33b0063f37c7a12bcb667896db64fe08b8bea05a95b5d8d",
        "checkpoint_sha256": checkpoint_hash,
        "tokenizer_sha256": tokenizer_hash,
        "training_data": {
            "mixture": "90% FSC scaling tokens / 10% HPLT-derived candidate-v2 tokens by token count",
            "manifest": "training_record/token-mixture-manifest.json",
            "manifest_sha256": digest(OUT / "training_record/token-mixture-manifest.json"),
            "fsc": "PleIAs/French-Science-Commons, french_open_science, revision bfa114c09d7c28dcd0a8a47fe445c64b4a3c6156; source-declared mixed licenses; inherited rights/privacy gaps remain",
            "hplt": "ashtok897/european-hplt-v1, revision 51edb798468aa71b1000e950d3ab43bb435f72cd; pinned card declares CC0-1.0 inherited from HPLT v3; source-page rights and privacy are not independently cleared",
            "training_text_or_token_shards_redistributed": False,
            "benchmark_files_used_as_training_data": False,
        },
        "evaluation": {
            "aggregate_records": ["training_record/paired-development-bpb.json", "training_record/paired-hplt-test-bpb.json"],
            "raw_benchmark_prompts_or_per_item_predictions_redistributed": False,
            "interpretation": "HPLT held-out BPB improves for this matched seed pair; FSC BPB regresses slightly; no multiple-choice accuracy gain is established",
        },
        "source_reviews": {"rights": "not cleared", "human_privacy": "not completed", "publication_authorization": "not a legal or privacy clearance"},
        "files_sha256": {},
    }
    for path in sorted(OUT.rglob("*")):
        if path.is_file() and path.name != "provenance.json":
            provenance["files_sha256"][path.relative_to(OUT).as_posix()] = digest(path)
    write_json(OUT / "provenance.json", provenance)
    print(json.dumps({"out": str(OUT), "checkpoint_sha256": checkpoint_hash, "tokenizer_sha256": tokenizer_hash, "files": len(provenance["files_sha256"]) + 1, "weight_bytes": (OUT / "model.safetensors").stat().st_size}, indent=2))


if __name__ == "__main__":
    main()
