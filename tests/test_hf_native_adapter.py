"""Check the standalone HF implementation against tiny native models, offline."""

from dataclasses import asdict
import shutil
from pathlib import Path

import pytest
import torch

pytest.importorskip("transformers")
from transformers import AutoModelForCausalLM
from slm.config import ModelConfig
from slm.model import SLM
from slm.hf_config import SlmTavenelConfig
from slm.hf_model import SlmTavenelForCausalLM


@pytest.mark.parametrize("gate", ["none", "headwise", "elementwise"])
@pytest.mark.parametrize("tied", [True, False])
@pytest.mark.parametrize("window,qk_norm", [([], True), ([3, 0], False)])
def test_native_logits_and_standalone_reload(tmp_path, gate, tied, window, qk_norm):
    torch.manual_seed(42)
    config = ModelConfig(vocab_size=32, d_model=16, n_heads=4, n_kv_heads=2,
                         n_layers=2, ffn_hidden=24, max_seq_len=16,
                         attn_gate=gate, tie_embeddings=tied, qk_norm=qk_norm, window_pattern=window)
    native = SLM(config).eval()
    hf_config = SlmTavenelConfig(**asdict(config), pad_token_id=0,
                               auto_map={"AutoConfig": "hf_config.SlmTavenelConfig",
                                         "AutoModelForCausalLM": "hf_model.SlmTavenelForCausalLM"})
    model = SlmTavenelForCausalLM(hf_config).eval()
    model.load_state_dict(native.state_dict(), strict=True)
    ids = torch.randint(1, 32, (2, 8))
    with torch.inference_mode():
        expected = native(ids)[0]
        torch.testing.assert_close(model(ids).logits, expected, atol=1e-6, rtol=0)
        padded = torch.cat([torch.zeros((2, 2), dtype=torch.long), ids], dim=1)
        mask = padded.ne(0).long()
        torch.testing.assert_close(model(padded, attention_mask=mask).logits[:, 2:], expected,
                                   atol=1e-6, rtol=0)
    model.save_pretrained(tmp_path)
    for name in ("hf_config.py", "hf_model.py"):
        shutil.copyfile(Path(__file__).resolve().parents[1] / "src/slm" / name, tmp_path / name)
    reloaded = AutoModelForCausalLM.from_pretrained(tmp_path, local_files_only=True, trust_remote_code=True).eval()
    with torch.inference_mode():
        torch.testing.assert_close(reloaded(ids).logits, expected, atol=1e-6, rtol=0)
        assert reloaded.generate(ids, max_new_tokens=2, do_sample=False).shape == (2, 10)
    assert (reloaded.embed.weight is reloaded.lm_head.weight) == tied
