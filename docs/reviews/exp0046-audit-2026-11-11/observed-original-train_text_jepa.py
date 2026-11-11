#!/usr/bin/env python3
"""Train the 124M decoder on COCO captions, as a text JEPA on good views or next-token control (EXP-0046).

``--mode jepa``: each update takes 256 images and two distinct captions of each (good views of one scene);
sentence embedding = linear projection (256 dims) of the mean final hidden state; loss = VICReg-style
``goodview_loss`` (see slm.goodview), with no next-token term. ``--mode ntp``: the same captions, same
sampling and budget, through next-token cross-entropy only. Muon, WSD schedule, bfloat16; GPU energy logged;
resumable; checkpoints load with ``slm.train.load_model`` (the projection head is dropped).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from slm.config import TrainConfig, load_model_config  # noqa: E402
from slm.energy import GpuEnergyMeter  # noqa: E402
from slm.goodview import GoodViewHead, goodview_loss, masked_mean, parallel_ntp_loss  # noqa: E402
from slm.model import SLM  # noqa: E402
from slm.optim import build_optimizer  # noqa: E402
from slm.train import attention_backend, autocast_context, batch_rng, latest_checkpoint, lr_at, save_checkpoint  # noqa: E402


class CaptionPairs:
    def __init__(self, directory: Path):
        data = np.load(directory / "train.npz")
        self.ids, self.lengths, image = data["ids"], data["lengths"], data["image"]
        order = np.argsort(image, kind="stable")
        self.order = order
        self.starts = np.searchsorted(image[order], np.arange(image.max() + 2))

    def sample(self, rng: np.random.Generator, batch: int) -> tuple[np.ndarray, ...]:
        images = rng.integers(0, len(self.starts) - 1, size=batch)
        a, b = [], []
        for image in images:
            start, stop = self.starts[image], self.starts[image + 1]
            first, second = rng.choice(stop - start, size=2, replace=False)
            a.append(self.order[start + first])
            b.append(self.order[start + second])
        a, b = np.array(a), np.array(b)
        wa, wb = int(self.lengths[a].max()), int(self.lengths[b].max())
        return self.ids[a, :wa], self.lengths[a], self.ids[b, :wb], self.lengths[b]


def chunked_ntp_loss(model: SLM, ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """Same value and gradient as ``parallel_ntp_loss`` on full logits, without storing B x T x vocab logits."""
    targets = ids[:, 1:].clone()
    targets[torch.arange(targets.size(1), device=ids.device)[None, :] >= (lengths[:, None] - 1)] = -100
    return model(ids[:, :-1], targets, return_logits=False)[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("jepa", "ntp"), required=True)
    parser.add_argument("--model", default="configs/model/slm-124m-gpt2match.yaml")
    parser.add_argument("--data", type=Path, default=ROOT / "data/parallel/coco2017-captions-v1")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=9000)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20261109)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    device = torch.device(args.device)
    args.out.mkdir(parents=True, exist_ok=True)
    cfg = TrainConfig(model=args.model, data_dir=str(args.data), out_dir=str(args.out), seed=args.seed, optimizer="muon",
                      lr=3e-3, min_lr=3e-4, weight_decay=0.1, schedule="wsd", warmup_steps=300, decay_fraction=0.2,
                      max_steps=args.steps, dtype="bfloat16", deterministic=False, keep_checkpoints=2)
    torch.manual_seed(args.seed)
    model_cfg = load_model_config(args.model)
    model = SLM(model_cfg).to(device)
    model.activation_checkpointing = True  # recomputation only: same gradients, fits the 8 GB GPU (no dropout)
    if args.mode == "jepa":
        model.jepa_predictor = GoodViewHead(model_cfg.d_model, args.dim).to(device)
        model.keep_hidden = True
    opt = build_optimizer(model, "muon", cfg.lr, cfg.weight_decay, (cfg.beta1, cfg.beta2), cfg.eps,
                          momentum=cfg.muon_momentum, fused=device.type == "cuda")
    start, energy_before = 0, 0.0
    if (resume := latest_checkpoint(args.out)) is not None:
        state = torch.load(resume, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])
        start, energy_before = state["step"], float(state.get("gpu_energy_wh") or 0.0)
    pairs = CaptionPairs(args.data)
    meter = GpuEnergyMeter(device.index or 0) if device.type == "cuda" else None
    metrics = (args.out / "metrics.jsonl").open("a", encoding="utf-8")
    t0, last = time.perf_counter(), start
    for step in range(start, args.steps):
        for group in opt.param_groups:
            group["lr"] = lr_at(step, cfg)
        a, a_len, b, b_len = (torch.from_numpy(x.astype(np.int64)).to(device)
                              for x in pairs.sample(batch_rng(args.seed, step, 0, 0), args.batch))
        stats: dict[str, float] = {}
        with autocast_context(device, cfg.dtype), attention_backend(device, False):
            if args.mode == "jepa":
                model(a)
                z_a = model.jepa_predictor(masked_mean(model.last_hidden, a_len))
                model(b)
                z_b = model.jepa_predictor(masked_mean(model.last_hidden, b_len))
                loss, stats = goodview_loss(z_a, z_b, 1.0, "vicreg")
            else:
                loss = 0.5 * (chunked_ntp_loss(model, a, a_len) + chunked_ntp_loss(model, b, b_len))
            loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip).item()
        opt.step()
        opt.zero_grad(set_to_none=True)
        done = step + 1
        energy = None if meter is None or not meter.available else energy_before + meter.energy_wh()
        if done % 50 == 0 or done == args.steps:
            record = {"time": time.time(), "step": done, "loss": round(loss.item(), 5), "grad_norm": round(grad_norm, 5),
                      "steps_per_s": round((done - last) / (time.perf_counter() - t0), 3), **{k: round(v, 5) for k, v in stats.items()}}
            if device.type == "cuda":
                record["max_mem_gb"] = round(torch.cuda.max_memory_allocated(device) / 1e9, 3)
            if energy is not None:
                record["gpu_energy_wh"] = round(energy, 3)
            metrics.write(json.dumps(record) + "\n")
            metrics.flush()
            t0, last = time.perf_counter(), done
        if done % 500 == 0 or done == args.steps:
            save_checkpoint(args.out, done, model, opt, cfg, 2, extra={"gpu_energy_wh": energy} if energy is not None else None)
    if meter is not None:
        meter.close()


if __name__ == "__main__":
    main()
