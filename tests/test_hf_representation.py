from dataclasses import asdict
from pathlib import Path
import shutil

import pytest
import torch

from slm.config import ModelConfig
from slm.hf_config import SlmTavenelConfig
from slm.hf_representation import SlmTavenelForRepresentation
from slm.model import SLM


def model_pair():
    torch.manual_seed(19)
    config = ModelConfig(vocab_size=32, d_model=16, n_layers=3, n_heads=4,
                         n_kv_heads=2, ffn_hidden=32, max_seq_len=32,
                         qk_norm=True, attn_gate="headwise")
    native = SLM(config).eval()
    hf_config = SlmTavenelConfig(**asdict(config), representation_layer=1, representation_max_length=16,
                              auto_map={"AutoConfig": "hf_config.SlmTavenelConfig",
                                        "AutoModel": "hf_representation.SlmTavenelForRepresentation"})
    hf = SlmTavenelForRepresentation(hf_config).eval()
    hf.load_state_dict({k: v for k, v in native.state_dict().items() if k != "lm_head.weight"}, strict=True)
    return native, hf


def test_audited_block_and_masked_pool_match_native():
    native, hf = model_pair()
    ids = torch.randint(1, 32, (2, 8))
    expected = []
    hook = native.blocks[1].register_forward_hook(lambda m, i, o: expected.append(o.detach()))
    with torch.no_grad():
        native(ids)
        actual = hf(ids, output_hidden_states=True)
        torch.testing.assert_close(actual.last_hidden_state, expected[0], atol=1e-6, rtol=0)
        torch.testing.assert_close(actual.pooler_output, expected[0].mean(1), atol=1e-6, rtol=0)
        assert len(actual.hidden_states) == 4
        padded = torch.cat([ids, torch.zeros((2, 3), dtype=torch.long)], dim=1)
        mask = padded.ne(0).long()
        pooled = hf(padded, attention_mask=mask).pooler_output
        torch.testing.assert_close(pooled, actual.pooler_output, atol=1e-6, rtol=0)
    hook.remove()
    assert sum(p.numel() for p in hf.parameters()) == native.num_params()
    assert not hf.can_generate()


def test_standalone_auto_model_reload(tmp_path):
    from transformers import AutoModel
    _, hf = model_pair()
    hf.save_pretrained(tmp_path)
    root = Path(__file__).parents[1]
    for name in ("hf_config.py", "hf_model.py", "hf_representation.py"):
        shutil.copyfile(root / "src/slm" / name, tmp_path / name)
    reloaded = AutoModel.from_pretrained(tmp_path, trust_remote_code=True, local_files_only=True).eval()
    ids = torch.tensor([[1, 2, 3, 4]])
    with torch.no_grad():
        torch.testing.assert_close(reloaded(ids).pooler_output, hf(ids).pooler_output, atol=0, rtol=0)
    assert not reloaded.can_generate()


def test_context_padding_and_layer_are_explicit():
    _, hf = model_pair()
    for ids, mask in ((torch.ones(1, 17, dtype=torch.long), None),
                      (torch.zeros(1, 4, dtype=torch.long), torch.zeros(1, 4)),
                      (torch.ones(1, 4, dtype=torch.long), torch.tensor([[0, 1, 1, 1]]))):
        with pytest.raises(ValueError):
            hf(ids, attention_mask=mask)
    with pytest.raises(ValueError):
        hf(torch.ones(1, 4, dtype=torch.long), representation_layer=3)
