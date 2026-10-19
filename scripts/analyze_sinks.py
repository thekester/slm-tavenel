"""Causal tests of attention sinks and massive activations in a trained checkpoint.

Each test is an intervention on a named hook point, scored by the change in next-token loss on
held-out windows (positions 1..T-1). Results go to <run>/analysis/sinks.json.

    python scripts/analyze_sinks.py runs/slm-10m-wiki/checkpoints/step_00015000.pt \
        --data data/tokens/wiki-v0 --windows 32 --seq-len 256
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from slm import interp
from slm.data.shards import TokenShards
from slm.observe import ablate_heads, capture, intervene
from slm.train import load_model


def loss_on(model, x, y) -> float:
    logits, _ = model(x)
    return F.cross_entropy(logits[:, 1:].flatten(0, 1).float(), y[:, 1:].flatten()).item()


def drop_first_key(renormalize: bool):
    def fn(p: torch.Tensor) -> torch.Tensor:
        p = p.clone()
        p[:, :, 1:, 0] = 0.0  # queries after position 0 stop attending to the first token
        if renormalize:
            p[:, :, 1:] = p[:, :, 1:] / p[:, :, 1:].sum(-1, keepdim=True).clamp_min(1e-9)
        return p

    return fn


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--data", required=True)
    ap.add_argument("--windows", type=int, default=32)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    model = load_model(args.checkpoint, device).eval()
    cfg = model.cfg
    x, y = TokenShards(args.data, "val").sample(np.random.default_rng(args.seed), args.windows, args.seq_len)
    x, y = torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)
    results: dict = {"checkpoint": args.checkpoint, "windows": args.windows, "seq_len": args.seq_len, "seed": args.seed}

    with torch.no_grad():
        base = loss_on(model, x, y)
        results["base_loss"] = base

        # 1. Where are the massive activations?
        with capture(model, "blocks.*.hook_resid_post") as acts:
            model(x)
        massive = []
        for i in range(cfg.n_layers):
            a = acts[f"blocks.{i}.hook_resid_post"].float()  # B x T x d
            per_dim = a.abs().amax(dim=(0, 1))
            top = per_dim.topk(3)
            median = a.abs().median().item()
            pos0 = a[:, 0].abs().amax().item()
            rest = a[:, 1:].abs().amax().item()
            massive.append({
                "layer": i,
                "median_abs": round(median, 4),
                "max_abs_pos0": round(pos0, 2),
                "max_abs_other_positions": round(rest, 2),
                "top_dims": top.indices.tolist(),
                "top_values": [round(v, 2) for v in top.values.tolist()],
            })
        results["massive_activations"] = massive
        last = acts[f"blocks.{cfg.n_layers - 1}.hook_resid_post"].float()
        # Dimensions whose position-0 magnitude dwarfs the typical scale anywhere in the stream.
        scale = last[:, 1:].abs().mean().item()
        dims = sorted({d for m in massive for d, v in zip(m["top_dims"], m["top_values"]) if v > 20 * scale})
        results["massive_dims"] = dims

        # 2. Attention to the first token, removed per layer, with and without renormalisation.
        stats = interp.attention_statistics(model, x)
        results["sink_mass"] = [[round(v, 4) for v in row] for row in stats["sink_mass"].tolist()]
        model.set_attention_capture(True)
        per_layer = []
        for i in range(cfg.n_layers):
            site = f"blocks.{i}.attn.hook_pattern"
            row = {"layer": i, "sink_mass": round(stats["sink_mass"][i].mean().item(), 4)}
            for renorm in (False, True):
                with intervene(model, {site: drop_first_key(renorm)}):
                    row["renormalized" if renorm else "removed"] = round(loss_on(model, x, y) - base, 4)
            per_layer.append(row)
        for renorm in (False, True):
            with intervene(model, {f"blocks.{i}.attn.hook_pattern": drop_first_key(renorm) for i in range(cfg.n_layers)}):
                results[f"all_layers_{'renormalized' if renorm else 'removed'}"] = round(loss_on(model, x, y) - base, 4)
        model.set_attention_capture(False)
        results["first_token_attention"] = per_layer

        # 3. Sink heads versus random heads.
        flat = [(stats["sink_mass"][l, h].item(), l, h) for l in range(cfg.n_layers) for h in range(cfg.n_heads)]
        k = max(1, len(flat) // 8)
        sink_heads = [(l, h) for _, l, h in sorted(flat, reverse=True)[:k]]

        def ablate(heads):
            by_layer: dict[int, list[int]] = {}
            for l, h in heads:
                by_layer.setdefault(l, []).append(h)
            with intervene(model, {f"blocks.{l}.attn.hook_z": ablate_heads(hs) for l, hs in by_layer.items()}):
                return loss_on(model, x, y) - base

        rng = random.Random(args.seed)
        others = [(l, h) for _, l, h in flat if (l, h) not in sink_heads]
        random_effects = [ablate(rng.sample(others, k)) for _ in range(10)]
        results["head_ablation"] = {
            "k": k,
            "sink_heads": sink_heads,
            "sink_heads_delta": round(ablate(sink_heads), 4),
            "random_heads_delta_mean": round(float(np.mean(random_effects)), 4),
            "random_heads_delta_max": round(float(np.max(random_effects)), 4),
        }

        # 4. Massive dimensions: replace them with their median at position 0 only, then everywhere.
        def neutralize(dims, positions):
            def fn(a):
                a = a.clone()
                med = a[:, 1:, dims].median(dim=1, keepdim=True).values.median(dim=0, keepdim=True).values
                if positions == "first":
                    a[:, :1, dims] = med
                else:
                    a[:, :, dims] = med
                return a

            return fn

        if dims:
            out = {}
            for where in ("first", "all"):
                sites = {f"blocks.{i}.hook_resid_post": neutralize(dims, where) for i in range(cfg.n_layers - 1)}
                with intervene(model, sites):
                    out[f"{where}_positions"] = round(loss_on(model, x, y) - base, 4)
                    if where == "first":
                        out["sink_mass_after"] = round(interp.attention_statistics(model, x)["sink_mass"].mean().item(), 4)
            rand_dims = random.Random(args.seed).sample([d for d in range(cfg.d_model) if d not in dims], len(dims))
            sites = {f"blocks.{i}.hook_resid_post": neutralize(rand_dims, "first") for i in range(cfg.n_layers - 1)}
            with intervene(model, sites):
                out["random_dims_first_positions"] = round(loss_on(model, x, y) - base, 4)
            out["sink_mass_before"] = round(stats["sink_mass"].mean().item(), 4)
            results["massive_dim_neutralization"] = out

    out_path = Path(args.out) if args.out else Path(args.checkpoint).parent.parent / "analysis" / "sinks.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in results.items() if k not in ("sink_mass",)}, indent=1))


if __name__ == "__main__":
    main()
