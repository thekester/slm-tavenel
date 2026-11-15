"""Hub feature extractor for the native SLM-Tavenel backbone.

Returns pre-final-normalization block features and their mask-weighted mean,
matching EXP-0046's representation metric. It has no language-generation head.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import BaseModelOutputWithPooling

from .hf_config import SlmTavenelConfig
from .hf_model import Block, RMSNorm, rope_tables


class SlmTavenelForRepresentation(PreTrainedModel):
    config_class = SlmTavenelConfig
    base_model_prefix = ""
    _no_split_modules = ["Block"]

    def __init__(self, config):
        super().__init__(config)
        self.embed = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList(Block(config, i) for i in range(config.n_layers))
        # Kept as part of the exact trained backbone, even though the default
        # audited representation is taken before the final norm.
        self.final_norm = RMSNorm(config.d_model, config.norm_eps)
        self._rope = None
        self.post_init()

    def get_input_embeddings(self):
        return self.embed

    def set_input_embeddings(self, value):
        self.embed = value

    def forward(self, input_ids, attention_mask=None, output_hidden_states=False,
                return_dict=None, representation_layer=None, **kwargs):
        if input_ids.ndim != 2 or input_ids.shape[1] < 1:
            raise ValueError("Expected nonempty batch x sequence input_ids")
        limit = min(self.config.max_seq_len, getattr(self.config, "representation_max_length", 64))
        if input_ids.shape[1] > limit:
            raise ValueError(f"Feature extraction is limited to {limit} tokens; tokenize with truncation=True")
        layer = getattr(self.config, "representation_layer", self.config.n_layers-1) if representation_layer is None else representation_layer
        if not 0 <= layer < self.config.n_layers:
            raise ValueError("Representation layer is zero-based and must be inside the backbone")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if attention_mask.shape != input_ids.shape:
            raise ValueError("attention_mask shape differs from input_ids")
        mask = attention_mask.bool()
        if not bool(mask.any(1).all()) or bool(((~mask[:, :-1]) & mask[:, 1:]).any()):
            raise ValueError("Use nonempty, right-padded sequences")
        if self._rope is None or self._rope[0].device != input_ids.device:
            cos, sin = rope_tables(self.config.head_dim, self.config.max_seq_len, self.config.rope_theta)
            self._rope = (cos.to(input_ids.device), sin.to(input_ids.device))
        cos, sin = self._rope
        hidden = self.embed(input_ids)
        states = [hidden] if output_hidden_states else None
        selected = None
        for i, block in enumerate(self.blocks):
            hidden = block(hidden, cos, sin, attention_mask=attention_mask)
            if output_hidden_states:
                states.append(hidden)
            if i == layer:
                selected = hidden
                if not output_hidden_states:
                    break
        pooled = (selected.float()*mask[:, :, None]).sum(1)/mask.sum(1)[:, None]
        result = BaseModelOutputWithPooling(last_hidden_state=selected, pooler_output=pooled,
                                           hidden_states=tuple(states) if states is not None else None)
        return result if return_dict is not False else result.to_tuple()
