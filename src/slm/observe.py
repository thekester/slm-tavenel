"""Capture and intervene on named activations.

These are the primitives for causal explanations: a claim such as "head 3.5 copies
the subject name" is tested by changing what flows through ``blocks.3.attn.hook_z``
and measuring the change in behaviour, not by looking at attention maps alone.
"""

from __future__ import annotations

import fnmatch
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager

import torch
import torch.nn as nn

from slm.model import SLM

Selector = str | Iterable[str] | Callable[[str], bool]
Intervention = Callable[[torch.Tensor], torch.Tensor]


def resolve_sites(model: SLM, selector: Selector) -> list[str]:
    """Return hook names matching a glob (``blocks.*.hook_resid_post``), a list, or a predicate."""
    names = list(model.hook_points())
    if callable(selector):
        chosen = [n for n in names if selector(n)]
    elif isinstance(selector, str):
        chosen = [n for n in names if fnmatch.fnmatchcase(n, selector)]
    else:
        wanted = list(selector)
        missing = [w for w in wanted if w not in names]
        if missing:
            raise KeyError(f"unknown hook points: {missing}")
        chosen = wanted
    if not chosen:
        raise KeyError(f"no hook point matches {selector!r}")
    if any(n.endswith("hook_pattern") for n in chosen) and not all(
        b.attn.capture_pattern for b in model.blocks
    ):
        raise RuntimeError("attention patterns require model.set_attention_capture(True)")
    return chosen


@contextmanager
def capture(model: SLM, selector: Selector, detach: bool = True, to_cpu: bool = False) -> Iterator[dict[str, torch.Tensor]]:
    """Record activations at the selected sites during forward passes inside the block.

    If the same site fires several times (generation with a KV cache), the last
    value is kept.
    """
    store: dict[str, torch.Tensor] = {}
    handles = []
    points = model.hook_points()
    for name in resolve_sites(model, selector):

        def hook(_m: nn.Module, _inp, out: torch.Tensor, name: str = name) -> None:
            t = out.detach() if detach else out
            store[name] = t.cpu() if to_cpu else t

        handles.append(points[name].register_forward_hook(hook))
    try:
        yield store
    finally:
        for h in handles:
            h.remove()


@contextmanager
def intervene(model: SLM, interventions: dict[str, Intervention]) -> Iterator[None]:
    """Replace activations at sites by ``fn(activation)`` during forward passes inside the block."""
    points = model.hook_points()
    resolve_sites(model, list(interventions))
    handles = [
        points[name].register_forward_hook(lambda _m, _i, out, fn=fn: fn(out))
        for name, fn in interventions.items()
    ]
    try:
        yield
    finally:
        for h in handles:
            h.remove()


# ---- common interventions --------------------------------------------------------------


def zero() -> Intervention:
    return torch.zeros_like


def replace_with(value: torch.Tensor) -> Intervention:
    """Patch in a stored activation (e.g. from a clean run). Shapes must match."""

    def fn(out: torch.Tensor) -> torch.Tensor:
        if out.shape != value.shape:
            raise ValueError(f"patch shape {tuple(value.shape)} != activation shape {tuple(out.shape)}")
        return value.to(out.device, out.dtype)

    return fn


def ablate_heads(heads: Iterable[int], mean: torch.Tensor | None = None) -> Intervention:
    """Ablate heads at an ``attn.hook_z`` site (B x T x H x D), by zero or by a mean activation."""
    heads = list(heads)

    def fn(z: torch.Tensor) -> torch.Tensor:
        z = z.clone()
        if mean is None:
            z[:, :, heads, :] = 0
        else:
            z[:, :, heads, :] = mean[..., heads, :].to(z.device, z.dtype)
        return z

    return fn


def ablate_positions(positions: Iterable[int]) -> Intervention:
    """Zero the activation at given sequence positions (dim 1 for residual-stream sites)."""
    positions = list(positions)

    def fn(x: torch.Tensor) -> torch.Tensor:
        x = x.clone()
        x[:, positions] = 0
        return x

    return fn


def add_direction(direction: torch.Tensor, scale: float) -> Intervention:
    """Steer a residual-stream site by adding ``scale * direction``."""

    def fn(x: torch.Tensor) -> torch.Tensor:
        return x + scale * direction.to(x.device, x.dtype)

    return fn
