"""Typed configurations loaded from YAML, with strict key checking and stable hashes.

Every run records the hash of its resolved configuration so that results can be
traced back to the exact settings that produced them.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar

import yaml

T = TypeVar("T")


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a mapping at the top level")
    return data


def from_dict(cls: type[T], data: dict[str, Any]) -> T:
    """Build a dataclass from a dict, rejecting unknown keys (typos must not pass silently)."""
    known = {f.name for f in dataclasses.fields(cls)}  # type: ignore[arg-type]
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"{cls.__name__}: unknown keys {sorted(unknown)}")
    return cls(**data)


def canonical_json(obj: Any) -> str:
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        obj = dataclasses.asdict(obj)
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def config_hash(obj: Any) -> str:
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()[:16]


@dataclass
class ModelConfig:
    """Decoder-only transformer. Defaults describe the 1B target (see docs/ARCHITECTURE.md)."""

    vocab_size: int = 49152
    d_model: int = 2048
    n_layers: int = 16
    n_heads: int = 32
    n_kv_heads: int = 8
    ffn_hidden: int = 8192
    max_seq_len: int = 4096
    rope_theta: float = 100000.0
    norm_eps: float = 1e-6
    tie_embeddings: bool = True
    qk_norm: bool = True
    # Output gate applied after attention: "none", "headwise" (one scalar per head)
    # or "elementwise" (one value per head dimension).
    attn_gate: str = "none"
    # Per-layer sliding window sizes, repeated cyclically over layers. 0 means global
    # attention; an empty list means every layer is global.
    window_pattern: list[int] = field(default_factory=list)
    init_std: float = 0.02

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if self.n_heads % self.n_kv_heads:
            raise ValueError("n_heads must be divisible by n_kv_heads")
        if self.head_dim % 2:
            raise ValueError("head_dim must be even for rotary embeddings")
        if self.attn_gate not in ("none", "headwise", "elementwise"):
            raise ValueError(f"unknown attn_gate {self.attn_gate!r}")
        if any(w < 0 for w in self.window_pattern):
            raise ValueError("window sizes must be >= 0")

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    def window_for(self, layer: int) -> int:
        if not self.window_pattern:
            return 0
        return self.window_pattern[layer % len(self.window_pattern)]


@dataclass
class TrainConfig:
    model: str = "configs/model/slm-tiny.yaml"
    data_dir: str = "data/tokens/smoke"
    out_dir: str = "runs/smoke"
    seed: int = 1234
    seq_len: int = 256
    micro_batch_size: int = 8
    grad_accum_steps: int = 1
    max_steps: int = 200
    # Optimiser: "adamw", or "muon" (Muon for block matrices, AdamW for the rest; see slm.optim)
    optimizer: str = "adamw"
    muon_momentum: float = 0.95
    # Restrict attention to the current document inside packed training windows
    doc_mask: bool = False
    # Initial weights (e.g. a grown checkpoint from `slm grow`); ignored when resuming
    init_from: str = ""
    lr: float = 3e-3
    min_lr: float = 3e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    grad_clip: float = 1.0
    # Schedule: "wsd" (warmup-stable-decay) or "cosine"
    schedule: str = "wsd"
    warmup_steps: int = 20
    decay_fraction: float = 0.2
    # Precision and performance
    dtype: str = "bfloat16"
    compile: bool = False
    deterministic: bool = True
    peak_tflops: float = 0.0  # per device; enables MFU logging when > 0
    # Logging, evaluation, observability, checkpointing
    log_interval: int = 10
    eval_interval: int = 100
    eval_batches: int = 8
    observe_interval: int = 100
    ckpt_interval: int = 100
    keep_checkpoints: int = 3
    resume: bool = True

    def __post_init__(self) -> None:
        if self.optimizer not in ("adamw", "muon"):
            raise ValueError(f"unknown optimizer {self.optimizer!r}")
        if self.schedule not in ("wsd", "cosine"):
            raise ValueError(f"unknown schedule {self.schedule!r}")
        if self.dtype not in ("float32", "bfloat16", "float16"):
            raise ValueError(f"unknown dtype {self.dtype!r}")
        if not 0.0 <= self.decay_fraction <= 1.0:
            raise ValueError("decay_fraction must be in [0, 1]")


def load_model_config(path: str | Path) -> ModelConfig:
    return from_dict(ModelConfig, load_yaml(path))


def load_train_config(path: str | Path, overrides: dict[str, Any] | None = None) -> TrainConfig:
    data = load_yaml(path)
    data.update(overrides or {})
    return from_dict(TrainConfig, data)
