import pytest
import torch

from slm.config import ModelConfig
from slm.model import SLM


def tiny_config(**overrides) -> ModelConfig:
    base = dict(vocab_size=97, d_model=32, n_layers=3, n_heads=4, n_kv_heads=2, ffn_hidden=64,
                max_seq_len=64, rope_theta=10000.0)
    base.update(overrides)
    return ModelConfig(**base)


@pytest.fixture
def tiny_model() -> SLM:
    torch.manual_seed(0)
    return SLM(tiny_config()).eval()
