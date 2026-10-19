"""Run the project's evaluations on any Hugging Face causal language model.

Reference models are scored with exactly the same prompts, scoring rules and documents as our own
checkpoints, so differences come from the models, not from the harness. The adapters expose the
small interface the evaluation code relies on: ``model(ids)[0]`` returns logits, ``model.cfg`` has
``max_seq_len``, ``tok.encode(text).ids`` and ``tok.token_to_id(EOS)`` (mapped to the model's own
beginning- or end-of-text token, used as the document prefix).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from slm.tokenizer import EOS


@dataclass
class _Encoding:
    ids: list[int]


@dataclass
class _Cfg:
    max_seq_len: int


class HFTokenizer:
    def __init__(self, tokenizer) -> None:
        self.tok = tokenizer
        prefix = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id
        if prefix is None:
            raise ValueError("tokenizer has neither a beginning- nor an end-of-text token")
        self.prefix_id = prefix

    def encode(self, text: str) -> _Encoding:
        return _Encoding(self.tok.encode(text, add_special_tokens=False))

    def decode(self, ids: list[int]) -> str:
        return self.tok.decode(ids)

    def token_to_id(self, token: str) -> int:
        if token == EOS:
            return self.prefix_id
        return self.tok.convert_tokens_to_ids(token)


class HFModel(torch.nn.Module):
    def __init__(self, model, max_seq_len: int | None = None) -> None:
        super().__init__()
        self.model = model
        limit = getattr(model.config, "max_position_embeddings", 2048)
        self.cfg = _Cfg(min(limit, max_seq_len or limit))

    def forward(self, input_ids: torch.Tensor):
        return self.model(input_ids=input_ids).logits, None

    def num_params(self) -> int:
        return sum(p.numel() for p in self.model.parameters())


def load(name: str, device: torch.device, revision: str | None = None, max_seq_len: int | None = 2048):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name, revision=revision)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(name, revision=revision, torch_dtype=dtype).to(device).eval()
    return HFModel(model, max_seq_len), HFTokenizer(tok)
