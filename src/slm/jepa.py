"""Action-conditioned joint-embedding prediction (JEPA) as an auxiliary objective (EXP-0038).

For a sequence of actions a(1..T) (Othello moves, environment actions), the language model's final
hidden state h(t) summarises the history up to a(t). A small predictor receives h(t) and the next
action a(t+1) and predicts the representation of the next state, h_bar(t+1), produced by an
exponential-moving-average copy of the model under stop-gradient (I-JEPA, Assran et al., 2023;
V-JEPA 2-AC for action conditioning, Assran et al., 2025; LLM-JEPA for language models, Huang,
LeCun & Balestriero, 2025). Targets are layer-normalised and the loss is the mean squared error, so
the objective asks h(t) to carry what is needed to predict the consequence of an action in latent
space, rather than only the next token.

The predictor and action embedding are registered on the model as ``jepa_predictor`` (counted and
reported separately; they are dropped when a checkpoint is loaded for evaluation). The EMA target
lives outside the model and is saved in the training checkpoint.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from slm.model import SLM


class JEPAPredictor(nn.Module):
    def __init__(self, d_model: int, vocab_size: int, hidden: int = 0) -> None:
        super().__init__()
        hidden = hidden or 2 * d_model
        self.action = nn.Embedding(vocab_size, d_model)
        self.net = nn.Sequential(nn.Linear(2 * d_model, hidden), nn.GELU(), nn.Linear(hidden, d_model))
        nn.init.normal_(self.action.weight, 0.0, 0.02)

    def forward(self, h: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([h, self.action(actions)], dim=-1))


def jepa_loss(predictor: JEPAPredictor, h: torch.Tensor, target_h: torch.Tensor, input_ids: torch.Tensor,
              ignore_action: int | None = None) -> torch.Tensor:
    """Predict layer-normalised target(t+1) from (h(t), a(t+1)); positions whose next action is
    ``ignore_action`` (e.g. end-of-game padding) are excluded."""
    actions = input_ids[:, 1:]
    pred = predictor(h[:, :-1], actions).float()
    target = F.layer_norm(target_h[:, 1:].float(), (target_h.size(-1),)).detach()
    per_position = (pred - target).pow(2).mean(-1)
    mask = torch.ones_like(per_position) if ignore_action is None else (actions != ignore_action).float()
    return (per_position * mask).sum() / mask.sum().clamp_min(1)


def make_target(model: SLM) -> SLM:
    """Frozen EMA copy of the language model (without the predictor)."""
    target = SLM(model.cfg).to(next(model.parameters()).device)
    target.load_state_dict(core_state(model))
    target.requires_grad_(False).eval()
    target.keep_hidden = True
    return target


def core_state(model: SLM) -> dict[str, torch.Tensor]:
    return {k: v for k, v in model.state_dict().items() if not k.startswith("jepa_predictor.")}


@torch.no_grad()
def update_target(target: SLM, model: SLM, decay: float) -> None:
    params = dict(model.named_parameters())
    for name, p in target.named_parameters():
        p.lerp_(params[name].detach().to(p.dtype), 1 - decay)
