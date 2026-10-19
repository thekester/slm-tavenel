"""Intervention-based analyses built on ``slm.observe``.

Each function returns plain tensors or dicts so results can be logged as JSON and
compared across checkpoints. The guiding rule from the project README: an
explanation is supported when intervening on the component it names changes the
behaviour it is supposed to explain.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

import torch

from slm.model import SLM
from slm.observe import ablate_heads, capture, intervene, replace_with

Metric = Callable[[torch.Tensor], torch.Tensor]  # logits -> scalar


def logit_diff_metric(position: int, correct: int, incorrect: int) -> Metric:
    """Logit(correct) - logit(incorrect) at a position, averaged over the batch."""

    def metric(logits: torch.Tensor) -> torch.Tensor:
        lp = logits[:, position].float()
        return (lp[:, correct] - lp[:, incorrect]).mean()

    return metric


def target_logprob_metric(position: int, target: int) -> Metric:
    def metric(logits: torch.Tensor) -> torch.Tensor:
        return logits[:, position].float().log_softmax(-1)[:, target].mean()

    return metric


@torch.no_grad()
def logit_lens(model: SLM, input_ids: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Log-probability of ``targets`` decoded from the residual stream after each layer.

    Returns a tensor of shape (n_layers + 1, B, T): row 0 decodes the embeddings,
    row i decodes the output of block i - 1. The last row equals the model output.
    """
    sites = ["hook_embed"] + [f"blocks.{i}.hook_resid_post" for i in range(model.cfg.n_layers)]
    with capture(model, sites) as acts:
        model(input_ids)
    rows = []
    for name in sites:
        logits = model.lm_head(model.final_norm(acts[name])).float()
        rows.append(logits.log_softmax(-1).gather(-1, targets[..., None]).squeeze(-1))
    return torch.stack(rows)


@torch.no_grad()
def head_ablation_effects(
    model: SLM, input_ids: torch.Tensor, metric: Metric, layers: Iterable[int] | None = None
) -> torch.Tensor:
    """Change in ``metric`` when each attention head is zero-ablated, shape (L, H).

    Negative values mean the head supports the measured behaviour.
    """
    base = metric(model(input_ids)[0])
    layers = list(range(model.cfg.n_layers)) if layers is None else list(layers)
    effects = torch.zeros(model.cfg.n_layers, model.cfg.n_heads)
    for layer in layers:
        for head in range(model.cfg.n_heads):
            with intervene(model, {f"blocks.{layer}.attn.hook_z": ablate_heads([head])}):
                effects[layer, head] = (metric(model(input_ids)[0]) - base).item()
    return effects


@torch.no_grad()
def activation_patching(
    model: SLM,
    clean_ids: torch.Tensor,
    corrupt_ids: torch.Tensor,
    metric: Metric,
    sites: Iterable[str],
) -> dict[str, float]:
    """Normalised effect of restoring each clean activation inside the corrupted run.

    1.0 means patching that site alone fully restores clean behaviour; 0.0 means it
    has no effect. Clean and corrupted inputs must have the same shape.
    """
    sites = list(sites)
    if clean_ids.shape != corrupt_ids.shape:
        raise ValueError("clean and corrupted inputs must have the same shape")
    with capture(model, sites) as clean_acts:
        clean = metric(model(clean_ids)[0]).item()
    corrupt = metric(model(corrupt_ids)[0]).item()
    denom = clean - corrupt
    results = {}
    for site in sites:
        with intervene(model, {site: replace_with(clean_acts[site])}):
            patched = metric(model(corrupt_ids)[0]).item()
        results[site] = (patched - corrupt) / denom if abs(denom) > 1e-12 else float("nan")
    return results


@torch.no_grad()
def attention_statistics(model: SLM, input_ids: torch.Tensor) -> dict[str, torch.Tensor]:
    """Per layer and head: attention mass on the first token (sink score) and mean entropy.

    Query position 0 is excluded since it can only attend to itself. Shapes (L, H).
    """
    was = [b.attn.capture_pattern for b in model.blocks]
    model.set_attention_capture(True)
    try:
        with capture(model, "blocks.*.attn.hook_pattern") as pats:
            model(input_ids)
    finally:
        for b, w in zip(model.blocks, was):
            b.attn.capture_pattern = w
    sink, entropy = [], []
    for i in range(model.cfg.n_layers):
        p = pats[f"blocks.{i}.attn.hook_pattern"][:, :, 1:, :].float()  # B x H x (T-1) x S
        sink.append(p[..., 0].mean(dim=(0, 2)))
        ent = -(p * p.clamp_min(1e-12).log()).sum(-1)
        entropy.append(ent.mean(dim=(0, 2)))
    return {"sink_mass": torch.stack(sink), "entropy": torch.stack(entropy)}


@torch.no_grad()
def gate_statistics(model: SLM, input_ids: torch.Tensor, closed: float = 0.1) -> dict[str, torch.Tensor]:
    """Gated attention only: mean gate value and fraction of tokens with gate < ``closed``,
    per layer and head, shape (L, H). In a gated model a closed gate, not attention to the
    first token, is the signature of a head that outputs nothing."""
    if model.cfg.attn_gate == "none":
        raise ValueError("model has no attention gate")
    with capture(model, "blocks.*.attn.hook_gate") as gates:
        model(input_ids)
    mean, frac = [], []
    for i in range(model.cfg.n_layers):
        g = gates[f"blocks.{i}.attn.hook_gate"].float()  # B x T x H x (1 or D)
        g = g.mean(-1)
        mean.append(g.mean(dim=(0, 1)))
        frac.append((g < closed).float().mean(dim=(0, 1)))
    return {"mean": torch.stack(mean), "closed_frac": torch.stack(frac)}


@torch.no_grad()
def residual_norms(model: SLM, input_ids: torch.Tensor) -> dict[str, torch.Tensor]:
    """RMS and max-abs of the residual stream after each block, shape (L,)."""
    with capture(model, "blocks.*.hook_resid_post") as acts:
        model(input_ids)
    rms, mx = [], []
    for i in range(model.cfg.n_layers):
        a = acts[f"blocks.{i}.hook_resid_post"].float()
        rms.append(a.pow(2).mean().sqrt())
        mx.append(a.abs().max())
    return {"rms": torch.stack(rms), "max_abs": torch.stack(mx)}
