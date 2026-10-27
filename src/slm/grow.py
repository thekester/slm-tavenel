"""Grow a trained model into a larger one that starts by computing the same function.

Two function-preserving operators, combinable:

- **Depth (block expansion).** Each original block is followed by ``copies`` duplicates whose output
  projections (attention ``o_proj`` and MLP ``down_proj``) are zero, so each duplicate initially adds
  nothing to the residual stream: the grown model's output equals the original's. The zeroed
  projections still receive gradients (their input activations are non-zero), so the new blocks
  start learning at the first step. This is the block expansion of LLaMA Pro (Wu et al., 2024,
  arXiv:2401.02415); interleaving the copies keeps each new block next to the block it was copied
  from.
- **MLP width.** New hidden units get freshly initialised ``gate_proj`` / ``up_proj`` rows and zero
  ``down_proj`` columns, so they contribute nothing at first (function preservation as in Net2Net,
  Chen et al., 2016, arXiv:1511.05641, with zero instead of split output weights).

Everything else (embeddings, norms, attention width, vocabulary) must match. Growth is recorded in
the new checkpoint (``grown_from``) so the lineage of a model, and the compute already spent on it,
stays traceable.
"""

from __future__ import annotations

import copy
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from slm.config import ModelConfig
from slm.model import SLM
from slm.runlog import read_json, sha256_file, write_json

FIXED = ("vocab_size", "d_model", "n_heads", "n_kv_heads", "max_seq_len", "rope_theta", "norm_eps",
         "tie_embeddings", "qk_norm", "attn_gate")


def check_compatible(src: ModelConfig, dst: ModelConfig) -> int:
    """Validate a growth from ``src`` to ``dst``; return the number of blocks per original block."""
    for name in FIXED:
        if getattr(src, name) != getattr(dst, name):
            raise ValueError(f"growth cannot change {name}: {getattr(src, name)} -> {getattr(dst, name)}")
    if src.window_pattern or dst.window_pattern:
        raise ValueError("growth with sliding-window layers is not supported")
    if dst.n_layers % src.n_layers:
        raise ValueError("target depth must be a multiple of the source depth")
    if dst.ffn_hidden < src.ffn_hidden:
        raise ValueError("MLP width can only grow")
    return dst.n_layers // src.n_layers


@torch.no_grad()
def grow(src: SLM, dst_cfg: ModelConfig, seed: int = 0) -> SLM:
    factor = check_compatible(src.cfg, dst_cfg)
    torch.manual_seed(seed)
    dst = SLM(dst_cfg)  # fresh initialisation for the parts that are new
    dst.embed.weight.copy_(src.embed.weight)
    if not dst_cfg.tie_embeddings:
        dst.lm_head.weight.copy_(src.lm_head.weight)
    dst.final_norm.weight.copy_(src.final_norm.weight)
    old_h = src.cfg.ffn_hidden
    for i, block in enumerate(src.blocks):
        for c in range(factor):
            target = dst.blocks[i * factor + c]
            state = copy.deepcopy(block.state_dict())
            mlp = {k: v for k, v in state.items() if k.startswith("mlp.")}
            for k in mlp:
                state.pop(k)
            target.load_state_dict(state, strict=False)
            # MLP: copy the original units, keep fresh initialisation for the new ones,
            # and silence the new units through zero output columns.
            target.mlp.gate_proj.weight[:old_h] = block.mlp.gate_proj.weight
            target.mlp.up_proj.weight[:old_h] = block.mlp.up_proj.weight
            target.mlp.down_proj.weight.zero_()
            target.mlp.down_proj.weight[:, :old_h] = block.mlp.down_proj.weight
            if c > 0:  # duplicated block: contributes nothing until it learns
                target.attn.o_proj.weight.zero_()
                target.mlp.down_proj.weight.zero_()
    return dst


def grow_checkpoint(checkpoint: str | Path, dst_cfg: ModelConfig, out: str | Path, seed: int = 0) -> dict[str, Any]:
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    src = SLM(ModelConfig(**state["model_config"]))
    src.load_state_dict(state["model"])
    dst = grow(src, dst_cfg, seed)
    info = {
        "source_checkpoint": str(checkpoint),
        "source_sha256": sha256_file(checkpoint),
        "source_params": src.num_params(),
        "source_train_config": state.get("train_config"),
        "params": dst.num_params(),
        "depth_factor": dst_cfg.n_layers // src.cfg.n_layers,
        "ffn_hidden": [src.cfg.ffn_hidden, dst_cfg.ffn_hidden],
        "method": "block expansion with zero output projections + zero-output MLP widening",
        "seed": seed,
    }
    source_run_manifest = Path(checkpoint).resolve().parent.parent / "run_manifest.json"
    if source_run_manifest.is_file():
        source_manifest = read_json(source_run_manifest)
        source_tokenizer_sha = source_manifest.get("tokenizer_sha256") or source_manifest.get("data", {}).get("tokenizer_sha256")
        if source_tokenizer_sha:
            info["tokenizer_sha256"] = str(source_tokenizer_sha)
    if not info.get("tokenizer_sha256"):
        source_provenance = Path(checkpoint).with_suffix(Path(checkpoint).suffix + ".provenance.json")
        if source_provenance.is_file():
            source_tokenizer_sha = read_json(source_provenance).get("tokenizer_sha256")
            if source_tokenizer_sha:
                info["tokenizer_sha256"] = str(source_tokenizer_sha)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"step": 0, "model": dst.state_dict(), "model_config": asdict(dst_cfg), "grown_from": info}, out)
    write_json(out.with_suffix(out.suffix + ".provenance.json"), info)
    return info
