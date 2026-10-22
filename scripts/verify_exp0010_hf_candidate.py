"""Check tokenizer, logits and basic generation for a local EXP-0010 Hub bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
import transformers
from tokenizers import Tokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    args = parser.parse_args()
    bundle, checkpoint_path, tokenizer_path = (p.resolve() for p in (args.bundle, args.checkpoint, args.tokenizer))

    from slm.config import ModelConfig
    from slm.model import SLM

    probe = "La recherche scientifique. Scientific research."
    raw_tokenizer = Tokenizer.from_file(str(tokenizer_path))
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    native = SLM(ModelConfig(**checkpoint["model_config"]))
    native.load_state_dict(checkpoint["model"])
    native.eval()
    tokenizer = AutoTokenizer.from_pretrained(bundle, trust_remote_code=True, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(bundle, trust_remote_code=True, local_files_only=True).eval()

    expected = raw_tokenizer.encode(probe).ids
    actual = tokenizer.encode(probe, add_special_tokens=False)
    if actual != expected:
        raise AssertionError("Bundled tokenizer differs from the training tokenizer")
    eos_id = raw_tokenizer.token_to_id("<|endoftext|>")
    input_ids = torch.tensor([[eos_id, *expected]], dtype=torch.long)
    with torch.inference_mode():
        native_logits = native(input_ids)[0].float()
        bundle_logits = model(input_ids=input_ids).logits.float()
    max_abs = (native_logits - bundle_logits).abs().max().item()
    if max_abs > 1e-6:
        raise AssertionError(f"CPU float32 parity failed: max abs difference {max_abs} > 1e-6")
    with torch.inference_mode():
        generated = model.generate(input_ids=input_ids, max_new_tokens=8, do_sample=False)
    generated_count = generated.shape[1] - input_ids.shape[1]
    if generated_count != 8:
        raise AssertionError(f"Greedy generation returned {generated_count} new tokens, expected 8")

    provenance_path = bundle / "provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    report = {
        "schema": "slm.exp0010_hf_candidate_parity.v1",
        "status": "passed",
        "scope": "fixed-input tokenizer equality, CPU float32 logit parity, short greedy generation",
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "tokenizer_ids_equal": True,
        "max_abs_logit_difference": max_abs,
        "tolerance_cpu_float32_max_abs": 1e-6,
        "generation": {"method": "greedy", "new_tokens_requested": 8,
                       "new_tokens_returned": generated_count, "kv_cache": False},
        "versions": {"torch": torch.__version__, "transformers": transformers.__version__},
        "hashes": {"checkpoint": sha256(checkpoint_path), "weights": sha256(bundle / "model.safetensors"),
                   "tokenizer": sha256(bundle / "tokenizer.json"), "hf_model": sha256(bundle / "hf_model.py"),
                   "hf_config": sha256(bundle / "hf_config.py"), "verifier": sha256(Path(__file__).resolve())},
        "limitations": ["one fixed text probe", "not an evaluation of language quality, source rights, or privacy"],
    }
    report_path = bundle / "inference-parity-report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    provenance["inference_parity"] = {"report": report_path.name, "report_sha256": sha256(report_path),
                                      "status": "passed", "scope": report["scope"]}
    provenance.setdefault("files_sha256", {})[report_path.name] = sha256(report_path)
    provenance_path.write_text(json.dumps(provenance, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("status", "parameter_count", "max_abs_logit_difference", "generation")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
