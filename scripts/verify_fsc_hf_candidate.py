"""Check native versus local Hugging Face inference for an EXP-0009 scale bundle."""

from __future__ import annotations

import argparse
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
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scale", choices=("10m", "25m"))
    args = parser.parse_args()
    spec = {"10m": ("10m-gated-s1234", "step_00015000.pt"), "25m": ("25m-gated-grown-s1234", "step_00015000.pt")}[args.scale]
    run = ROOT / "runs/fsc-scaling" / spec[0]
    checkpoint_path = run / "checkpoints" / spec[1]
    bundle = ROOT / "exports" / ("slm-tavenel-10m-exp0009" if args.scale == "10m" else "slm-tavenel-25m-exp0009-candidate")
    reference_tokenizer = Tokenizer.from_file(str(ROOT / "data/tokenizer/fsc-pilot-v1-train-only/tokenizer.json"))
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    native = SLM(ModelConfig(**checkpoint["model_config"]))
    native.load_state_dict(checkpoint["model"])
    native.eval()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    hf_tokenizer = AutoTokenizer.from_pretrained(bundle, trust_remote_code=True, local_files_only=True)
    hf_model = AutoModelForCausalLM.from_pretrained(bundle, trust_remote_code=True, local_files_only=True).eval()
    prompt = "La recherche scientifique. Scientific research."
    expected_ids = reference_tokenizer.encode(prompt).ids
    actual_ids = hf_tokenizer.encode(prompt, add_special_tokens=False)
    if actual_ids != expected_ids:
        raise AssertionError("Hub tokenizer IDs differ from the training tokenizer")
    eos = reference_tokenizer.token_to_id("<|endoftext|>")
    input_ids = torch.tensor([[eos, *expected_ids]], dtype=torch.long)
    with torch.inference_mode():
        native_logits = native(input_ids)[0]
        hf_logits = hf_model(input_ids=input_ids).logits
    max_abs = float((native_logits - hf_logits).abs().max())
    mean_abs = float((native_logits - hf_logits).abs().mean())
    tolerance = 1e-6
    if max_abs > tolerance:
        raise AssertionError(f"Native/HF max logit error {max_abs} exceeds {tolerance}")
    with torch.inference_mode():
        generated = hf_model.generate(input_ids=input_ids, max_new_tokens=2, do_sample=False)
    report = {
        "schema": "slm.exp0009_hf_inference_parity.v1",
        "status": "passed",
        "scale": args.scale,
        "checkpoint_step": checkpoint["step"],
        "probe_text": prompt,
        "tokenizer_ids_equal": True,
        "native_hf_max_abs_logit_error": max_abs,
        "native_hf_mean_abs_logit_error": mean_abs,
        "max_abs_tolerance": tolerance,
        "generation_smoke_new_tokens": int(generated.shape[1] - input_ids.shape[1]),
        "parameter_count": sum(p.numel() for p in hf_model.parameters()),
        "versions": {"torch": torch.__version__, "transformers": transformers.__version__},
        "sha256": {
            "native_checkpoint": sha256(checkpoint_path),
            "hub_weights": sha256(bundle / "model.safetensors"),
            "tokenizer": sha256(bundle / "tokenizer.json"),
            "hub_model_code": sha256(bundle / "hf_model.py"),
            "hub_config_code": sha256(bundle / "hf_config.py"),
        },
        "limitations": ["one fixed CPU probe; not a broad numerical or quality evaluation", "does not establish data rights, privacy clearance, or scientific generalization"],
    }
    report_path = bundle / "inference-parity-report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    provenance_path = bundle / "provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["inference_parity"] = {"status": "passed", "report": report_path.name, "report_sha256": sha256(report_path), "scope": "CPU fixed-input native-versus-Hugging-Face tokenizer, logits and generation smoke check"}
    provenance["files_sha256"] = {
        path.relative_to(bundle).as_posix(): sha256(path)
        for path in sorted(bundle.rglob("*"))
        if path.is_file() and path != provenance_path
    }
    provenance_path.write_text(json.dumps(provenance, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("status", "scale", "checkpoint_step", "native_hf_max_abs_logit_error", "parameter_count", "generation_smoke_new_tokens")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
