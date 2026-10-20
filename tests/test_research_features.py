"""Document masking, Muon, gate statistics and the throughput benchmark."""

import copy

import pytest
import torch

from slm import interp
from slm.config import TrainConfig
from slm.model import SLM, document_ids, document_mask
from slm.optim import CombinedOptimizer, Muon, build_optimizer, muon_params, newton_schulz

from conftest import tiny_config

EOS = 0


def test_document_ids():
    x = torch.tensor([[5, 6, EOS, 7, EOS, EOS, 8]])
    assert document_ids(x, EOS).tolist() == [[0, 0, 0, 1, 1, 2, 3]]
    m = document_mask(document_ids(x, EOS))
    assert m.shape == (1, 1, 7, 7) and not m[0, 0, 3, 0] and m[0, 0, 4, 3]


@pytest.mark.parametrize("overrides", [{}, {"attn_gate": "headwise"}, {"window_pattern": [3, 0]}])
def test_packed_document_matches_document_alone(overrides):
    """RoPE is relative, so with document masking a packed document sees exactly what it would alone."""
    torch.manual_seed(0)
    model = SLM(tiny_config(**overrides)).eval()
    doc1 = torch.randint(1, 97, (1, 6))
    doc2 = torch.randint(1, 97, (1, 9))
    packed = torch.cat([doc1, torch.tensor([[EOS]]), doc2], dim=1)
    logits, _ = model(packed, doc_ids=document_ids(packed, EOS))
    alone, _ = model(doc2)
    torch.testing.assert_close(logits[:, 7:], alone, atol=1e-5, rtol=1e-4)
    unmasked, _ = model(packed)
    assert not torch.allclose(unmasked[:, 7:], alone, atol=1e-3)
    model.set_attention_capture(True)
    explicit, _ = model(packed, doc_ids=document_ids(packed, EOS))
    torch.testing.assert_close(explicit, logits, atol=1e-5, rtol=1e-4)


def test_newton_schulz_orthogonalizes():
    torch.manual_seed(0)
    for shape in [(32, 64), (64, 32)]:
        o = newton_schulz(torch.randn(*shape))
        s = torch.linalg.svdvals(o)
        assert s.min() > 0.5 and s.max() < 1.3  # quintic iteration targets singular values near 1


def test_muon_param_split_excludes_embeddings_and_vectors():
    model = SLM(tiny_config())
    muon, other = muon_params(model)
    assert all(p.dim() == 2 for p in muon)
    assert any(p is model.embed.weight for p in other)
    assert not any(p is model.embed.weight for p in muon)
    n_unique = len({id(p) for p in model.parameters()})
    assert len(muon) + len(other) == n_unique


def test_muon_trains_and_state_roundtrips():
    torch.manual_seed(0)
    model = SLM(tiny_config())
    opt = build_optimizer(model, "muon", lr=0.02, weight_decay=0.0, betas=(0.9, 0.95), eps=1e-8)
    assert isinstance(opt, CombinedOptimizer) and isinstance(opt.optimizers[0], Muon)
    x = torch.randint(0, 97, (4, 16))
    first = None
    for _ in range(30):
        _, loss = model(x[:, :-1], x[:, 1:])
        loss.backward()
        opt.step()
        opt.zero_grad()
        first = first if first is not None else loss.item()
    assert loss.item() < 0.5 * first
    for g in opt.param_groups:
        g["lr"] = 0.123
    assert all(g["lr"] == 0.123 for o in opt.optimizers for g in o.param_groups)
    # deepcopy stands in for torch.save/torch.load: Optimizer.load_state_dict keeps references to
    # tensors that already have the right dtype and device, so the two optimisers would share state.
    state = copy.deepcopy(opt.state_dict())
    model2 = SLM(tiny_config())
    model2.load_state_dict(model.state_dict())
    opt2 = build_optimizer(model2, "muon", lr=0.02, weight_decay=0.0, betas=(0.9, 0.95), eps=1e-8)
    opt2.load_state_dict(state)
    for m1, m2 in ((model, opt), (model2, opt2)):
        _, loss = m1(x[:, :-1], x[:, 1:])
        loss.backward()
        m2.step()
        m2.zero_grad()
    for p, q in zip(model.parameters(), model2.parameters()):
        torch.testing.assert_close(p, q)


def test_gate_statistics():
    torch.manual_seed(0)
    model = SLM(tiny_config(attn_gate="headwise")).eval()
    stats = interp.gate_statistics(model, torch.randint(0, 97, (2, 10)))
    assert stats["mean"].shape == (3, 4) and ((stats["mean"] > 0) & (stats["mean"] < 1)).all()
    with pytest.raises(ValueError):
        interp.gate_statistics(SLM(tiny_config()), torch.randint(0, 97, (1, 4)))


def test_benchmark_conversion_and_scoring():
    from tokenizers import pre_tokenizers, trainers

    from slm.benchmarks import convert, multiple_choice
    from slm.tokenizer import SPECIAL_TOKENS, build_empty

    arc = convert("arc_easy", [{"id": "q1", "question": "What is 2+2?", "answerKey": "B",
                                "choices": {"label": ["A", "B"], "text": ["3", "4"]}}])
    assert arc == [{"id": "q1", "context": "Question: What is 2+2?\nAnswer:", "choices": [" 3", " 4"], "label": 1}]
    bele = convert("belebele_fr", [{"link": "u", "question_number": 1, "flores_passage": "Texte.", "question": "Q ?",
                                    **{f"mc_answer{k}": str(k) for k in range(1, 5)}, "correct_answer_num": "3"}])
    assert bele[0]["label"] == 2 and bele[0]["context"].startswith("P: Texte.")

    tok = build_empty()
    tok.train_from_iterator(["Question: What is 2+2? Answer: 3 4"] * 10,
                            trainers.BpeTrainer(vocab_size=80, special_tokens=SPECIAL_TOKENS,
                                                initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    torch.manual_seed(0)
    model = SLM(tiny_config(vocab_size=tok.get_vocab_size()))
    res = multiple_choice(model, tok, arc * 3, torch.device("cpu"))
    assert res["items"] == 3 and res["chance"] == 0.5 and res["acc"] in (0.0, 1.0)
    assert [item["id"] for item in res["per_item"]] == ["q1"] * 3
    assert all(item["label"] == 1 for item in res["per_item"])
    for suffix in ("", "_norm"):
        assert res[f"acc{suffix}"] == sum(item[f"correct{suffix}"] for item in res["per_item"]) / 3
        assert all(item[f"correct{suffix}"] == (item[f"prediction{suffix}"] == item["label"])
                   for item in res["per_item"])

    from slm.benchmarks import batched_choice_logprobs, choice_logprobs

    items = arc + [
        {"id": "q2", "context": "Question: 2?", "choices": [" 3 4 3", " 4"], "label": 0},
        # Truncation must score all retained tokens, even when a choice exceeds the context window.
        {"id": "long", "context": "Question:", "choices": [" 3" * 80, " 4"], "label": 0},
    ]
    batched = batched_choice_logprobs(model, tok, items, torch.device("cpu"))
    single = [choice_logprobs(model, tok, it["context"], it["choices"], torch.device("cpu")) for it in items]
    for b, s in zip(batched, single):
        assert b == pytest.approx(s, abs=1e-4)


@pytest.mark.parametrize("optimizer,doc_mask",[("adamw", False), ("muon", True)])
def test_benchmark_runs(optimizer, doc_mask):
    from slm.train import benchmark

    cfg = TrainConfig(seq_len=32, micro_batch_size=2, optimizer=optimizer, doc_mask=doc_mask, dtype="float32")
    res = benchmark(tiny_config(), cfg, steps=2, warmup=1, mean_doc_len=8)
    assert res["tokens_per_s"] > 0 and res["optimizer"] == optimizer
