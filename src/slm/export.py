"""Export checkpoints to Hugging Face format so open weights run in standard tooling.

With global attention and no output gate, the architecture is exactly Llama
(``qk_norm: false``) or Qwen3 (``qk_norm: true``), so weights map one to one and
load in transformers, vLLM or llama.cpp without custom code. Experimental variants
(sliding windows, gated attention) are refused rather than silently approximated.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file

from slm.model import SLM
from slm.runlog import sha256_file, write_json
from slm.tokenizer import EOS


def hf_architecture(model: SLM) -> str:
    c = model.cfg
    if c.attn_gate != "none" or any(c.window_for(i) for i in range(c.n_layers)):
        raise ValueError("gated attention and sliding windows have no standard HF equivalent; export refused")
    return "qwen3" if c.qk_norm else "llama"


def hf_config(model: SLM, eos_id: int, pad_id: int | None) -> dict[str, Any]:
    c = model.cfg
    arch = hf_architecture(model)
    cfg = {
        "architectures": ["Qwen3ForCausalLM" if arch == "qwen3" else "LlamaForCausalLM"],
        "model_type": arch,
        "vocab_size": c.vocab_size,
        "hidden_size": c.d_model,
        "intermediate_size": c.ffn_hidden,
        "num_hidden_layers": c.n_layers,
        "num_attention_heads": c.n_heads,
        "num_key_value_heads": c.n_kv_heads,
        "head_dim": c.head_dim,
        "hidden_act": "silu",
        "max_position_embeddings": c.max_seq_len,
        "rms_norm_eps": c.norm_eps,
        "rope_theta": c.rope_theta,
        "rope_scaling": None,
        "tie_word_embeddings": c.tie_embeddings,
        "attention_bias": False,
        "attention_dropout": 0.0,
        "initializer_range": c.init_std,
        "bos_token_id": eos_id,
        "eos_token_id": eos_id,
        "pad_token_id": pad_id,
        "torch_dtype": "bfloat16",
        "use_cache": True,
    }
    if arch == "llama":
        cfg["mlp_bias"] = False
        cfg["pretraining_tp"] = 1
    else:
        cfg["use_sliding_window"] = False
        cfg["sliding_window"] = None
        cfg["max_window_layers"] = c.n_layers
    return cfg


def hf_state_dict(model: SLM, dtype: torch.dtype = torch.bfloat16) -> dict[str, torch.Tensor]:
    sd: dict[str, torch.Tensor] = {"model.embed_tokens.weight": model.embed.weight}
    for i, b in enumerate(model.blocks):
        p = f"model.layers.{i}."
        sd[p + "input_layernorm.weight"] = b.attn_norm.weight
        sd[p + "post_attention_layernorm.weight"] = b.mlp_norm.weight
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            sd[p + f"self_attn.{name}.weight"] = getattr(b.attn, name).weight
        if b.attn.q_norm is not None:
            sd[p + "self_attn.q_norm.weight"] = b.attn.q_norm.weight
            sd[p + "self_attn.k_norm.weight"] = b.attn.k_norm.weight
        for name in ("gate_proj", "up_proj", "down_proj"):
            sd[p + f"mlp.{name}.weight"] = getattr(b.mlp, name).weight
    sd["model.norm.weight"] = model.final_norm.weight
    if not model.cfg.tie_embeddings:
        sd["lm_head.weight"] = model.lm_head.weight
    return {k: v.detach().to("cpu", dtype).contiguous().clone() for k, v in sd.items()}


def export(model: SLM, tokenizer_file: str | Path, out_dir: str | Path, provenance: dict[str, Any] | None = None,
           dtype: torch.dtype = torch.bfloat16) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_file = Path(tokenizer_file)
    if tokenizer_file.is_dir():
        tokenizer_file = tokenizer_file / "tokenizer.json"
    vocab = json.loads(tokenizer_file.read_text(encoding="utf-8"))["added_tokens"]
    ids = {t["content"]: t["id"] for t in vocab}
    eos_id, pad_id = ids[EOS], ids.get("<|pad|>")

    cfg = hf_config(model, eos_id, pad_id)
    cfg["torch_dtype"] = str(dtype).removeprefix("torch.")
    write_json(out_dir / "config.json", cfg)
    write_json(out_dir / "generation_config.json", {"bos_token_id": eos_id, "eos_token_id": eos_id, "pad_token_id": pad_id})
    save_file(hf_state_dict(model, dtype), str(out_dir / "model.safetensors"), metadata={"format": "pt"})
    shutil.copyfile(tokenizer_file, out_dir / "tokenizer.json")
    write_json(out_dir / "tokenizer_config.json", {
        "tokenizer_class": "PreTrainedTokenizerFast",
        "eos_token": EOS,
        "pad_token": "<|pad|>",
        "bos_token": None,
        "unk_token": None,
        "model_max_length": model.cfg.max_seq_len,
        "clean_up_tokenization_spaces": False,
        "add_bos_token": False,
    })
    write_json(out_dir / "provenance.json", {
        **(provenance or {}),
        "files": {f.name: sha256_file(f) for f in sorted(out_dir.iterdir()) if f.name != "provenance.json"},
    })
    return out_dir
