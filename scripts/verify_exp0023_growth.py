"""Verify the EXP-0023 depth expansion preserves fixed-input logits before training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from slm.config import ModelConfig
from slm.model import SLM
from slm.runlog import sha256_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--grown", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    torch.set_num_threads(4)
    source_state = torch.load(args.source, map_location="cpu", weights_only=False)
    grown_state = torch.load(args.grown, map_location="cpu", weights_only=False)
    source = SLM(ModelConfig(**source_state["model_config"]))
    source.load_state_dict(source_state["model"])
    grown = SLM(ModelConfig(**grown_state["model_config"]))
    grown.load_state_dict(grown_state["model"])
    source.eval()
    grown.eval()

    ids = (torch.arange(2 * 128, dtype=torch.long).reshape(2, 128) * 37 + 11) % source.cfg.vocab_size
    with torch.inference_mode():
        source_logits, _ = source(ids)
        grown_logits, _ = grown(ids)
    max_abs_fp32 = float((source_logits - grown_logits).abs().max().item())
    same_fp32 = torch.allclose(source_logits, grown_logits, atol=1e-6, rtol=0)

    # Widening a linear layer changes the GEMM shape. Even with added zero
    # columns, FP32 kernels can change summation order and differ by a few ULPs.
    # Compare the mathematical functions again in float64 so that this benign
    # kernel-rounding delta is not confused with a non-preserving weight map.
    source.double()
    grown.double()
    with torch.inference_mode():
        source_logits64, _ = source(ids)
        grown_logits64, _ = grown(ids)
    max_abs_fp64 = float((source_logits64 - grown_logits64).abs().max().item())
    same_fp64 = torch.allclose(source_logits64, grown_logits64, atol=1e-10, rtol=1e-10)
    record = {
        "kind": "slm.exp0023_growth_function_check",
        "source_checkpoint": str(args.source),
        "source_checkpoint_sha256": sha256_file(args.source),
        "grown_checkpoint": str(args.grown),
        "grown_checkpoint_sha256": sha256_file(args.grown),
        "input_shape": list(ids.shape),
        "input_token_ids_sha256": __import__("hashlib").sha256(ids.numpy().tobytes()).hexdigest(),
        "source_params": source.num_params(),
        "grown_params": grown.num_params(),
        "max_absolute_logit_difference_fp32": max_abs_fp32,
        "allclose_fp32_atol_1e-6_rtol_0": same_fp32,
        "max_absolute_logit_difference_fp64": max_abs_fp64,
        "allclose_fp64_atol_1e-10_rtol_1e-10": same_fp64,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2))
    if not same_fp64:
        raise SystemExit("Growth failed fixed-input float64 function-preservation check")


if __name__ == "__main__":
    main()
