"""Check tokenizer and native/Transformers inference parity for the CC0 v2 candidate bundle."""

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


def diff(native: torch.nn.Module, hub: torch.nn.Module, input_ids: torch.Tensor, dtype: torch.dtype) -> dict:
    native = native.to(device=input_ids.device, dtype=dtype).eval()
    hub = hub.to(device=input_ids.device, dtype=dtype).eval()
    with torch.inference_mode():
        logits_native = native(input_ids)[0].float()
        logits_hub = hub(input_ids=input_ids).logits.float()
    return {"max_abs": (logits_native - logits_hub).abs().max().item(),
            "mean_abs": (logits_native - logits_hub).abs().mean().item()}


def main() -> int:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    bundle = ROOT / "exports/slm-tavenel-1m-cc0-v2-candidate"
    run = ROOT / "runs/open-weight-candidate/1m-gated-s20261022-v2-pilot"
    tokenizer_path = ROOT / "data/tokenizer/fsc-cc0-open-weights-v2/tokenizer.json"
    checkpoint_path = run / "checkpoints/step_00000805.pt"
    probe = "La recherche scientifique. Scientific research."
    native_tokenizer = Tokenizer.from_file(str(tokenizer_path))
    loaded = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    native = SLM(ModelConfig(**loaded["model_config"]))
    native.load_state_dict(loaded["model"])
    native.eval()
    hub_tokenizer = AutoTokenizer.from_pretrained(bundle, trust_remote_code=True, local_files_only=True)
    hub = AutoModelForCausalLM.from_pretrained(bundle, trust_remote_code=True, local_files_only=True).eval()
    expected = native_tokenizer.encode(probe).ids
    actual = hub_tokenizer.encode(probe, add_special_tokens=False)
    if actual != expected:
        raise AssertionError("Bundled tokenizer differs from the training tokenizer")
    bos_eos = native_tokenizer.token_to_id("<|endoftext|>")
    input_ids = torch.tensor([[bos_eos, *expected]], dtype=torch.long)
    cpu = diff(native, hub, input_ids, torch.float32)
    tolerance_fp32 = 1e-6
    if cpu["max_abs"] > tolerance_fp32:
        raise AssertionError(f"CPU float32 parity failed: {cpu['max_abs']} > {tolerance_fp32}")
    generation_input = input_ids
    with torch.inference_mode():
        generated = hub.generate(input_ids=generation_input, max_new_tokens=8, do_sample=False)
    if generated.shape[1] != input_ids.shape[1] + 8:
        raise AssertionError("Greedy generation did not return the requested continuation length")

    device_results = {"cpu_float32": cpu}
    if torch.cuda.is_available():
        cuda_input = input_ids.cuda()
        gpu = diff(native, hub, cuda_input, torch.float32)
        if gpu["max_abs"] > 1e-5:
            raise AssertionError(f"CUDA float32 parity failed: {gpu['max_abs']} > 1e-5")
        device_results["cuda_float32"] = gpu
        bf16 = diff(native, hub, cuda_input, torch.bfloat16)
        if bf16["max_abs"] > 0.02:
            raise AssertionError(f"CUDA bfloat16 parity failed: {bf16['max_abs']} > 0.02")
        device_results["cuda_bfloat16"] = bf16

    report = {
        "schema": "slm.hf_open_weight_candidate_parity.v2",
        "status": "passed",
        "scope": "fixed-input tokenizer/logit parity plus short greedy generation",
        "tokenizer_ids_equal": True,
        "parameter_count": sum(parameter.numel() for parameter in hub.parameters()),
        "device_results": device_results,
        "tolerances": {"cpu_float32_max_abs": tolerance_fp32, "cuda_float32_max_abs": 1e-5,
                       "cuda_bfloat16_max_abs": 0.02},
        "generation": {"method": "greedy", "new_tokens_requested": 8,
                       "new_tokens_returned": generated.shape[1] - input_ids.shape[1],
                       "kv_cache": False},
        "versions": {"torch": torch.__version__, "transformers": transformers.__version__},
        "hashes": {"checkpoint": sha256(checkpoint_path), "hub_weights": sha256(bundle / "model.safetensors"),
                   "tokenizer": sha256(bundle / "tokenizer.json"), "hf_model": sha256(bundle / "hf_model.py"),
                   "hf_config": sha256(bundle / "hf_config.py"), "script": sha256(Path(__file__).resolve())},
        "limitations": ["one short fixed probe; not broad generation-quality validation",
                        "does not establish data rights, privacy clearance, or scientific replication"],
    }
    path = bundle / "inference-parity-report.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    provenance_path = bundle / "provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["inference_parity"] = {"report": path.name, "report_sha256": sha256(path),
                                      "status": "passed", "scope": report["scope"]}
    provenance["files_sha256"][path.name] = sha256(path)
    provenance_path.write_text(json.dumps(provenance, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("status", "parameter_count", "device_results", "generation")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
