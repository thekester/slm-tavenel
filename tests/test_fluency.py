"""Protect stopping and sampling semantics: these affect claimed generation quality."""
from types import SimpleNamespace

import pytest
import torch

from slm.config import load_model_config
from slm.fluency import generate_sample, text_diagnostics
from slm.model import SLM


class Tokenizer:
    def encode(self, text, **kwargs):
        return [1]

    def decode(self, ids, **kwargs):
        return "".join({0: "", 1: "é", 2: " a"}[i] for i in ids)


class Model(torch.nn.Module):
    def __init__(self, eos=0, context=20):
        super().__init__()
        self.logits = torch.nn.Parameter(torch.tensor([-10.0, 2.0, 1.0]))
        self.config = SimpleNamespace(eos_token_id=eos, max_position_embeddings=context)

    def forward(self, input_ids, **kwargs):
        return SimpleNamespace(logits=self.logits.expand(1, input_ids.shape[1], 3))


def test_byte_budget_counts_utf8_and_keeps_raw_ids():
    r = generate_sample(Model(), Tokenizer(), "prompt", max_new_bytes=3)
    assert r["completion"] == "éé"
    assert r["utf8_bytes"] == 4  # Final-token overshoot is visible, not silently trimmed.
    assert r["generated_token_ids"] == [1, 1]
    assert r["stop_reason"] == "byte_budget"


def test_eos_context_and_token_cap_are_distinct():
    assert generate_sample(Model(eos=[1, 2]), Tokenizer(), "p")["stop_reason"] == "eos"
    r = generate_sample(Model(context=3), Tokenizer(), "p")
    assert r["stop_reason"] == "context_limit" and len(r["generated_token_ids"]) == 2
    assert generate_sample(Model(), Tokenizer(), "p", max_new_tokens=1)["stop_reason"] == "token_cap"


def test_sampling_is_seeded_and_rejects_invalid_settings():
    kwargs = dict(temperature=0.8, seed=7, max_new_tokens=10)
    assert generate_sample(Model(), Tokenizer(), "p", **kwargs)["generated_token_ids"] == \
        generate_sample(Model(), Tokenizer(), "p", **kwargs)["generated_token_ids"]
    for bad in ({"temperature": float("nan")}, {"top_p": 0}, {"max_new_bytes": 0}):
        with pytest.raises(ValueError):
            generate_sample(Model(), Tokenizer(), "p", **bad)
    with pytest.raises(ValueError):
        generate_sample(Model(context=1), Tokenizer(), "p")


def test_repetition_is_a_diagnostic_not_an_invented_quality_score():
    assert text_diagnostics("a b c a b c")["repeated_word_trigram_fraction"] == 0.25
    assert text_diagnostics("")["repeated_word_trigram_fraction"] is None


@pytest.mark.parametrize("name,expected", [("control", 592096), ("reallocated", 1166720)])
def test_candidate_parameter_budget(name, expected):
    model = SLM(load_model_config(f"configs/model/slm-1m-v2048-{name}.yaml"))
    assert model.num_params() == expected
    assert model.num_params() <= 1181920
