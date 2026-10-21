"""Record a CPU tokenizer/logit parity check for the local SLM-Tavenel 1M Hub candidate."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import torch
import transformers
from tokenizers import Tokenizer


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from slm.config import ModelConfig  # noqa: E402
from slm.model import SLM  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    bundle = ROOT / "exports/slm-tavenel-1m-candidate"
    run = ROOT / "runs/fsc-scaling/1m-gated-s1234"
    tokenizer_path = ROOT / "data/tokenizer/fsc-pilot-v1-train-only/tokenizer.json"
    checkpoint_path = run / "checkpoints/step_00015000.pt"
    probe_text = "La recherche scientifique. Scientific research."
    reference_tokenizer = Tokenizer.from_file(str(tokenizer_path))
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    native = SLM(ModelConfig(**checkpoint["model_config"]))
    native.load_state_dict(checkpoint["model"])
    native.eval()

    tokenizer = AutoTokenizer.from_pretrained(bundle, trust_remote_code=True, local_files_only=True)
    hf_model = AutoModelForCausalLM.from_pretrained(
        bundle, trust_remote_code=True, local_files_only=True
    ).eval()
    expected_ids = reference_tokenizer.encode(probe_text).ids
    actual_ids = tokenizer.encode(probe_text, add_special_tokens=False)
    if actual_ids != expected_ids:
        raise AssertionError("Hub tokenizer IDs differ from the training tokenizer")
    input_ids = torch.tensor([[reference_tokenizer.token_to_id("<|endoftext|>"), *expected_ids]])
    with torch.inference_mode():
        native_logits = native(input_ids)[0]
        hf_logits = hf_model(input_ids=input_ids).logits
    max_abs = (native_logits - hf_logits).abs().max().item()
    mean_abs = (native_logits - hf_logits).abs().mean().item()
    tolerance = 1e-6
    if max_abs > tolerance:
        raise AssertionError(f"Native/Hub logits exceed tolerance: {max_abs} > {tolerance}")

    with torch.inference_mode():
        generated = hf_model.generate(input_ids=input_ids, max_new_tokens=4, do_sample=False)
    if generated.shape[1] < input_ids.shape[1] + 1:
        raise AssertionError("Hub generate() did not append a continuation token")

    report = {
        "schema": "slm.hf_candidate_parity.v1",
        "status": "passed",
        "scope": "local CPU tokenizer and fixed-input native-versus-Hub inference parity",
        "probe_text": probe_text,
        "probe_token_ids": input_ids[0].tolist(),
        "tokenizer_ids_equal": True,
        "native_hf_max_abs_logit_error": max_abs,
        "native_hf_mean_abs_logit_error": mean_abs,
        "max_abs_tolerance": tolerance,
        "parameter_count": sum(parameter.numel() for parameter in hf_model.parameters()),
        "generation_smoke": {
            "method": "greedy",
            "new_tokens_requested": 4,
            "new_tokens_returned": generated.shape[1] - input_ids.shape[1],
            "continuation_token_ids": generated[0, input_ids.shape[1]:].tolist(),
            "kv_cache": False,
        },
        "versions": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
        "sha256": {
            "native_checkpoint": sha256(checkpoint_path),
            "hub_weights": sha256(bundle / "model.safetensors"),
            "tokenizer": sha256(bundle / "tokenizer.json"),
            "hub_model_code": sha256(bundle / "hf_model.py"),
            "hub_config_code": sha256(bundle / "hf_config.py"),
            "verification_script": sha256(Path(__file__).resolve()),
        },
        "limitations": [
            "single fixed probe on CPU; not a broad numerical conformance suite",
            "does not establish scientific validity, rights clearance, or generation quality",
        ],
    }
    report_path = bundle / "inference-parity-report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tracked_report = ROOT / "docs/model-cards/slm-tavenel-1m-hf-parity-report.json"
    tracked_report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    provenance_path = bundle / "provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["native_hf_parity"] = {
        "status": "passed",
        "report": report_path.name,
        "report_sha256": sha256(report_path),
        "scope": report["scope"],
    }
    provenance["public_release"]["reason"] = (
        "source-rights/privacy review is unfinished; the parity report is a local one-probe smoke check, "
        "not independent release approval"
    )
    provenance["integrity"]["files"][report_path.name] = sha256(report_path)
    provenance["integrity"]["repository_report_sha256"] = sha256(tracked_report)
    provenance_path.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in (
        "status", "tokenizer_ids_equal", "native_hf_max_abs_logit_error", "parameter_count", "generation_smoke"
    )}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
