"""Pretraining loop: deterministic, resumable, and instrumented.

Reproducibility: batches are drawn from a generator seeded by (seed, step, rank,
micro-step), so a resumed run sees exactly the batches an uninterrupted run would.
Checkpoints include optimiser and RNG state.

Observability: ``metrics.jsonl`` (loss, learning rate, gradient norm, throughput,
memory), ``eval.jsonl`` (validation loss on fixed batches) and
``observability.jsonl`` (per-layer residual norms, attention sink mass and entropy,
weight norms, update-to-weight ratios on a fixed probe batch).

Multi-GPU: launch with ``torchrun``; data parallelism via DDP. FSDP for larger
scale is on the roadmap.
"""

from __future__ import annotations

import contextlib
import math
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from slm import interp
from slm.config import ModelConfig, TrainConfig, config_hash, load_model_config
from slm.data.shards import TokenShards
from slm.model import SLM, document_ids
from slm.optim import build_optimizer
from slm.runlog import JsonlLogger, code_revision, environment_info, read_json, sha256_file, write_json


def lr_at(step: int, cfg: TrainConfig) -> float:
    """Linear warmup, then warmup-stable-decay (linear decay over the final fraction) or cosine."""
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    if cfg.schedule == "cosine":
        progress = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
        return cfg.min_lr + 0.5 * (cfg.lr - cfg.min_lr) * (1 + math.cos(math.pi * min(1.0, progress)))
    decay_steps = int(cfg.max_steps * cfg.decay_fraction)
    decay_start = cfg.max_steps - decay_steps
    if step < decay_start or decay_steps == 0:
        return cfg.lr
    progress = (step - decay_start) / decay_steps
    return cfg.lr + (cfg.min_lr - cfg.lr) * min(1.0, progress)


def param_groups(model: SLM, weight_decay: float) -> list[dict[str, Any]]:
    """Weight decay on matrices only; not on norms, gains or embeddings."""
    decay, no_decay = [], []
    seen = set()
    for name, p in model.named_parameters():
        if not p.requires_grad or id(p) in seen:
            continue
        seen.add(id(p))
        (decay if p.dim() >= 2 and "embed" not in name else no_decay).append(p)
    return [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]


def batch_rng(seed: int, step: int, rank: int, micro: int) -> np.random.Generator:
    return np.random.default_rng([seed, step, rank, micro])


def setup_distributed() -> tuple[int, int, torch.device]:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
        rank, world = dist.get_rank(), dist.get_world_size()
        if torch.cuda.is_available():
            local = int(os.environ.get("LOCAL_RANK", 0))
            torch.cuda.set_device(local)
            return rank, world, torch.device("cuda", local)
        return rank, world, torch.device("cpu")
    return 0, 1, torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_determinism(seed: int, deterministic: bool) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False


def autocast_context(device: torch.device, dtype: str):
    if dtype == "float32":
        return contextlib.nullcontext()
    amp = torch.bfloat16 if dtype == "bfloat16" else torch.float16
    if device.type == "cpu" and amp != torch.bfloat16:
        return contextlib.nullcontext()
    return torch.autocast(device_type=device.type, dtype=amp)


def attention_backend(device: torch.device, deterministic: bool):
    """Fused CUDA attention backward passes are not bitwise deterministic; fall back to math."""
    if deterministic and device.type == "cuda":
        from torch.nn.attention import SDPBackend, sdpa_kernel

        return sdpa_kernel(SDPBackend.MATH)
    return contextlib.nullcontext()


def checkpoint_path(out_dir: Path, step: int) -> Path:
    return out_dir / "checkpoints" / f"step_{step:08d}.pt"


def save_checkpoint(out_dir: Path, step: int, model: SLM, opt: torch.optim.Optimizer, cfg: TrainConfig, keep: int) -> Path:
    path = checkpoint_path(out_dir, step)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "step": step,
        "model": model.state_dict(),
        "optimizer": opt.state_dict(),
        "model_config": model.cfg.__dict__,
        "train_config": cfg.__dict__,
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    tmp.replace(path)
    ckpts = sorted(path.parent.glob("step_*.pt"))
    for old in ckpts[:-keep] if keep > 0 else []:
        old.unlink()
    return path


def latest_checkpoint(out_dir: Path) -> Path | None:
    ckpts = sorted((out_dir / "checkpoints").glob("step_*.pt"))
    return ckpts[-1] if ckpts else None


def load_model(checkpoint: str | Path, device: str | torch.device = "cpu") -> SLM:
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model = SLM(ModelConfig(**state["model_config"])).to(device)
    model.load_state_dict(state["model"])
    return model


@torch.no_grad()
def evaluate_loss(model: SLM, data: TokenShards, cfg: TrainConfig, device: torch.device, eos_id: int | None = None) -> float:
    """Loss on fixed validation windows, with the same attention masking as training.
    Compare runs with different masking through bits per byte on whole documents instead."""
    model.eval()
    losses = []
    for i in range(cfg.eval_batches):
        x, y = data.sample(batch_rng(cfg.seed + 1, 0, 0, i), cfg.micro_batch_size, cfg.seq_len)
        x = torch.from_numpy(x).to(device)
        doc_ids = document_ids(x, eos_id) if eos_id is not None else None
        with autocast_context(device, cfg.dtype):
            _, loss = model(x, torch.from_numpy(y).to(device), doc_ids=doc_ids)
        losses.append(loss.item())
    model.train()
    return float(np.mean(losses))


def matrix_weights(model: SLM) -> dict[str, torch.Tensor]:
    return {n: p.detach().float().clone() for n, p in model.named_parameters() if p.dim() >= 2}


@torch.no_grad()
def update_ratios(model: SLM, before: dict[str, torch.Tensor]) -> dict[str, float]:
    """||W_after - W_before|| / ||W_before|| per matrix: how much one optimiser step moved it."""
    after = matrix_weights(model)
    return {n: round(((w - before[n]).norm() / (before[n].norm() + 1e-12)).item(), 6) for n, w in after.items()}


@torch.no_grad()
def observe(model: SLM, probe: torch.Tensor) -> dict[str, Any]:
    model.eval()
    record: dict[str, Any] = {}
    norms = interp.residual_norms(model, probe)
    record["resid_rms"] = [round(v, 5) for v in norms["rms"].tolist()]
    record["resid_max_abs"] = [round(v, 4) for v in norms["max_abs"].tolist()]
    att = interp.attention_statistics(model, probe)
    record["sink_mass_mean"] = [round(v, 5) for v in att["sink_mass"].mean(1).tolist()]
    record["sink_mass_max"] = [round(v, 5) for v in att["sink_mass"].max(1).values.tolist()]
    record["attn_entropy_mean"] = [round(v, 5) for v in att["entropy"].mean(1).tolist()]
    if model.cfg.attn_gate != "none":
        gates = interp.gate_statistics(model, probe)
        record["gate_mean"] = [round(v, 4) for v in gates["mean"].mean(1).tolist()]
        record["gate_closed_frac"] = [round(v, 4) for v in gates["closed_frac"].mean(1).tolist()]
    record["weight_norm"] = {n: round(w.norm().item(), 4) for n, w in matrix_weights(model).items()}
    model.train()
    return record


def benchmark(model_cfg: ModelConfig, cfg: TrainConfig, steps: int = 30, warmup: int = 5,
              mean_doc_len: int = 600) -> dict[str, Any]:
    """Controlled throughput measurement: random tokens, fixed shapes, warmup excluded.

    Compares architectures and training options on equal footing, independently of data
    loading and of whatever else ran during a real training job. When ``cfg.doc_mask`` is set,
    end-of-text tokens are placed so documents average ``mean_doc_len`` tokens.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.seed)
    model = SLM(model_cfg).to(device)
    opt = build_optimizer(model, cfg.optimizer, cfg.lr, cfg.weight_decay, (cfg.beta1, cfg.beta2), cfg.eps,
                          momentum=cfg.muon_momentum, fused=device.type == "cuda")
    gen = torch.Generator().manual_seed(cfg.seed)
    eos = 0
    times = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for step in range(warmup + steps):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(cfg.grad_accum_steps):
            x = torch.randint(1, model_cfg.vocab_size, (cfg.micro_batch_size, cfg.seq_len + 1), generator=gen)
            if cfg.doc_mask:
                x[torch.rand(x.shape, generator=gen) < 1 / mean_doc_len] = eos
            x = x.to(device)
            doc_ids = document_ids(x[:, :-1], eos) if cfg.doc_mask else None
            with autocast_context(device, cfg.dtype):
                _, loss = model(x[:, :-1], x[:, 1:], return_logits=False, doc_ids=doc_ids)
                (loss / cfg.grad_accum_steps).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        opt.zero_grad(set_to_none=True)
        if device.type == "cuda":
            torch.cuda.synchronize()
        if step >= warmup:
            times.append(time.perf_counter() - t0)
    tokens = cfg.micro_batch_size * cfg.seq_len * cfg.grad_accum_steps
    med = float(np.median(times))
    result = {
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "params": model.num_params(),
        "optimizer": cfg.optimizer,
        "doc_mask": cfg.doc_mask,
        "tokens_per_step": tokens,
        "median_step_s": round(med, 5),
        "p90_step_s": round(float(np.percentile(times, 90)), 5),
        "tokens_per_s": round(tokens / med, 1),
    }
    if device.type == "cuda":
        result["max_mem_gb"] = round(torch.cuda.max_memory_allocated(device) / 1e9, 3)
    if cfg.peak_tflops > 0:
        result["mfu"] = round(model.flops_per_token(cfg.seq_len) * tokens / med / (cfg.peak_tflops * 1e12), 4)
    return result


def train(cfg: TrainConfig) -> dict[str, Any]:
    rank, world, device = setup_distributed()
    master = rank == 0
    set_determinism(cfg.seed, cfg.deterministic)
    out_dir = Path(cfg.out_dir)
    model_cfg = load_model_config(cfg.model)
    if cfg.mir and not 0 <= cfg.mir_mask_token_id < model_cfg.vocab_size:
        raise ValueError(f"MIR mask token id {cfg.mir_mask_token_id} is outside vocab size {model_cfg.vocab_size}")

    train_data = TokenShards(cfg.data_dir, "train")
    val_data = TokenShards(cfg.data_dir, "val")
    if train_data.vocab_size > model_cfg.vocab_size:
        raise ValueError(f"tokenizer vocab {train_data.vocab_size} > model vocab {model_cfg.vocab_size}")

    model = SLM(model_cfg).to(device)
    opt = build_optimizer(model, cfg.optimizer, cfg.lr, cfg.weight_decay, (cfg.beta1, cfg.beta2), cfg.eps,
                          momentum=cfg.muon_momentum, fused=device.type == "cuda")
    eos_id = int(train_data.manifest["eos_id"]) if cfg.doc_mask else None

    resume_from = latest_checkpoint(out_dir) if cfg.resume else None
    init_info = None
    if resume_from is not None and (out_dir / "run_manifest.json").exists():
        init_info = read_json(out_dir / "run_manifest.json").get("initialized_from")
    if cfg.init_from and resume_from is None:
        init = torch.load(cfg.init_from, map_location="cpu", weights_only=False)
        if ModelConfig(**init["model_config"]) != model_cfg:
            raise ValueError(f"{cfg.init_from} was built for a different model configuration")
        model.load_state_dict(init["model"])
        init_info = {"path": cfg.init_from, "sha256": sha256_file(cfg.init_from), "grown_from": init.get("grown_from")}

    start_step = 0
    if resume_from is not None:
        state = torch.load(resume_from, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])
        torch.set_rng_state(state["torch_rng"])
        if state.get("cuda_rng") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        start_step = state["step"]

    run_model: torch.nn.Module = model
    if cfg.compile:
        run_model = torch.compile(run_model)
    if world > 1:
        run_model = DDP(run_model, device_ids=[device.index] if device.type == "cuda" else None)

    if master:
        data_manifest_path = Path(cfg.data_dir) / "manifest.json"
        data_manifest = read_json(data_manifest_path)
        provenance_path = Path(cfg.data_dir) / "provenance.json"
        data_provenance = read_json(provenance_path) if provenance_path.is_file() else None
        write_json(out_dir / "run_manifest.json", {
            "kind": "slm.train",
            "train_config": cfg.__dict__,
            "train_config_hash": config_hash(cfg),
            "model_config": model_cfg.__dict__,
            "model_config_hash": config_hash(model_cfg),
            "params": model.num_params(),
            "params_non_embedding": model.num_params(non_embedding=True),
            "data": {
                "dir": cfg.data_dir,
                "manifest_sha256": sha256_file(data_manifest_path),
                "provenance_path": str(provenance_path) if provenance_path.is_file() else None,
                "provenance": data_provenance,
                "tokenizer_sha256": data_manifest.get("tokenizer_sha256"),
                "splits": {s: {"tokens": v["tokens"], "shards": [x["sha256"] for x in v["shards"]]}
                           for s, v in data_manifest["splits"].items()},
            },
            "world_size": world,
            "resumed_from": str(resume_from) if resume_from else None,
            "initialized_from": init_info,
            "code": code_revision(),
            "environment": environment_info(),
        })
    metrics = JsonlLogger(out_dir / "metrics.jsonl", master)
    evals = JsonlLogger(out_dir / "eval.jsonl", master)
    obs = JsonlLogger(out_dir / "observability.jsonl", master)

    tokens_per_step = cfg.micro_batch_size * cfg.seq_len * cfg.grad_accum_steps * world
    flops_per_step = model.flops_per_token(cfg.seq_len) * tokens_per_step
    if cfg.mir:
        # Both clean and masked objectives contribute a forward/backward pass.
        flops_per_step *= 2
    probe = torch.from_numpy(val_data.sample(batch_rng(cfg.seed + 2, 0, 0, 0), 2, min(cfg.seq_len, 128))[0]).to(device)
    last_loss = float("nan")
    t0 = time.perf_counter()

    for step in range(start_step, cfg.max_steps):
        lr = lr_at(step, cfg)
        for group in opt.param_groups:
            group["lr"] = lr
        loss_sum = 0.0
        clean_loss_sum = 0.0
        mir_loss_sum = 0.0
        for micro in range(cfg.grad_accum_steps):
            rng = batch_rng(cfg.seed, step, rank, micro)
            x, y = train_data.sample(rng, cfg.micro_batch_size, cfg.seq_len)
            x, y = torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)
            sync = contextlib.nullcontext()
            if world > 1 and micro < cfg.grad_accum_steps - 1:
                sync = run_model.no_sync()
            doc_ids = document_ids(x, eos_id) if eos_id is not None else None
            with sync, autocast_context(device, cfg.dtype), attention_backend(device, cfg.deterministic):
                _, loss = run_model(x, y, return_logits=False, doc_ids=doc_ids)
                total_loss = loss
                if cfg.mir:
                    ratios = rng.uniform(cfg.mir_mask_ratio_min, cfg.mir_mask_ratio_max,
                                         size=(cfg.micro_batch_size, 1))
                    mask = rng.random((cfg.micro_batch_size, cfg.seq_len)) < ratios
                    x_masked = x.clone()
                    x_masked[torch.from_numpy(mask).to(device)] = cfg.mir_mask_token_id
                    _, mir_loss = run_model(x_masked, y, return_logits=False, doc_ids=doc_ids)
                    total_loss = loss + cfg.mir_aux_weight * mir_loss
                    mir_loss_sum += mir_loss.item()
                (total_loss / cfg.grad_accum_steps).backward()
            clean_loss_sum += loss.item()
            loss_sum += total_loss.item()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip).item()
        observing = master and cfg.observe_interval and step % cfg.observe_interval == 0
        if observing:
            obs_record = observe(model, probe)
            before = matrix_weights(model)
        opt.step()
        opt.zero_grad(set_to_none=True)
        if observing:
            obs.log(step=step, **obs_record, update_ratio=update_ratios(model, before))
        last_loss = loss_sum / cfg.grad_accum_steps
        done = step + 1

        if master and (done % cfg.log_interval == 0 or done == cfg.max_steps):
            if device.type == "cuda":
                torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            steps_timed = cfg.log_interval if done % cfg.log_interval == 0 else done % cfg.log_interval
            record = {
                "step": done,
                "loss": round(last_loss, 5),
                "lr": lr,
                "grad_norm": round(grad_norm, 5),
                "tokens": done * tokens_per_step,
                "tokens_per_s": round(steps_timed * tokens_per_step / dt, 1),
            }
            record["ntp_loss"] = round(clean_loss_sum / cfg.grad_accum_steps, 5)
            if cfg.mir:
                record["mir_aux_loss"] = round(mir_loss_sum / cfg.grad_accum_steps, 5)
            if cfg.peak_tflops > 0:
                record["mfu"] = round(steps_timed * flops_per_step / dt / (cfg.peak_tflops * 1e12 * world), 4)
            if device.type == "cuda":
                record["max_mem_gb"] = round(torch.cuda.max_memory_allocated(device) / 1e9, 3)
            metrics.log(**record)
            t0 = time.perf_counter()
        if master and cfg.eval_interval and (done % cfg.eval_interval == 0 or done == cfg.max_steps):
            evals.log(step=done, val_loss=round(evaluate_loss(model, val_data, cfg, device, eos_id), 5))
        if master and cfg.ckpt_interval and (done % cfg.ckpt_interval == 0 or done == cfg.max_steps):
            save_checkpoint(out_dir, done, model, opt, cfg, cfg.keep_checkpoints)
        if world > 1:
            dist.barrier()

    if world > 1:
        dist.destroy_process_group()
    return {"final_step": cfg.max_steps, "final_loss": last_loss}
