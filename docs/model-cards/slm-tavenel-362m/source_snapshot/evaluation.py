"""Evaluation: tokenizer-independent likelihood, verified task completion, and cost.

- ``bits_per_byte`` compares models with different tokenizers on the same text.
- ``verified_tasks`` scores generations with a programmatic verifier and reports
  cost per successfully completed task, counting failed attempts.
- ``measure_cost`` reports latency, throughput, peak memory and, when NVML is
  available, GPU energy.

Downstream academic benchmarks are run through lm-evaluation-harness on exported
weights (see ``slm.export``), so that numbers are comparable with other models.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from tokenizers import Tokenizer

from slm.model import SLM
from slm.tokenizer import EOS


@torch.no_grad()
def bits_per_byte(model: SLM, tok: Tokenizer, texts: list[str], device: torch.device, max_len: int | None = None) -> dict[str, float]:
    """Sum of next-token NLL over all tokens, divided by ln(2) times the UTF-8 byte count.

    Each document is prefixed with the end-of-text token so that its first token
    is also predicted; documents longer than the context are scored in
    non-overlapping chunks.
    """
    model.eval()
    eos = tok.token_to_id(EOS)
    max_len = max_len or model.cfg.max_seq_len
    nll, n_tokens, n_bytes = 0.0, 0, 0
    for text in texts:
        ids = [eos] + tok.encode(text).ids
        n_bytes += len(text.encode("utf-8"))
        for start in range(0, len(ids) - 1, max_len):
            chunk = torch.tensor(ids[start : start + max_len + 1], device=device)[None]
            logits, _ = model(chunk[:, :-1])
            lp = logits.float().log_softmax(-1).gather(-1, chunk[:, 1:, None]).squeeze(-1)
            nll -= lp.sum().item()
            n_tokens += lp.numel()
    return {
        "bits_per_byte": nll / math.log(2) / max(1, n_bytes),
        "nll_per_token": nll / max(1, n_tokens),
        "tokens": n_tokens,
        "bytes": n_bytes,
    }


class EnergyMeter:
    """GPU energy via NVML's cumulative counter (Volta and newer); inert if unavailable."""

    def __init__(self, device_index: int = 0) -> None:
        self.handle = None
        try:
            import pynvml

            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
            pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
            self.nvml, self.handle = pynvml, handle
        except Exception:
            self.handle = None

    def read_mj(self) -> int | None:
        if self.handle is None:
            return None
        return self.nvml.nvmlDeviceGetTotalEnergyConsumption(self.handle)


@torch.no_grad()
def measure_cost(model: SLM, tok: Tokenizer, prompts: list[str], max_new_tokens: int, device: torch.device) -> dict[str, Any]:
    model.eval()
    meter = EnergyMeter(device.index or 0) if device.type == "cuda" else None
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize()
    e0 = meter.read_mj() if meter else None
    t_start = time.perf_counter()
    prompt_tokens = new_tokens = 0
    first_token_latencies = []
    for prompt in prompts:
        ids = torch.tensor([tok.encode(prompt).ids], device=device)
        t0 = time.perf_counter()
        model.generate(ids, max_new_tokens=1)
        if device.type == "cuda":
            torch.cuda.synchronize()
        first_token_latencies.append(time.perf_counter() - t0)
        out = model.generate(ids, max_new_tokens=max_new_tokens, eos_id=tok.token_to_id(EOS))
        prompt_tokens += ids.shape[1]
        new_tokens += out.shape[1] - ids.shape[1]
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t_start
    e1 = meter.read_mj() if meter else None
    result: dict[str, Any] = {
        "prompts": len(prompts),
        "prompt_tokens": prompt_tokens,
        "new_tokens": new_tokens,
        "wall_s": round(elapsed, 4),
        "mean_time_to_first_token_s": round(sum(first_token_latencies) / len(first_token_latencies), 5),
        "new_tokens_per_s": round(new_tokens / elapsed, 2),
        "params": model.num_params(),
        "device": str(device),
    }
    if device.type == "cuda":
        result["peak_mem_gb"] = round(torch.cuda.max_memory_allocated(device) / 1e9, 4)
    if e0 is not None and e1 is not None:
        result["energy_j"] = (e1 - e0) / 1000
    return result


@torch.no_grad()
def verified_tasks(
    model: SLM,
    tok: Tokenizer,
    tasks: list[dict[str, Any]],
    verify: Callable[[dict[str, Any], str], bool],
    device: torch.device,
    max_new_tokens: int = 32,
    attempts: int = 1,
    temperature: float = 0.0,
    seed: int = 0,
) -> dict[str, Any]:
    """Success rate with up to ``attempts`` tries per task; cost counts every attempt."""
    model.eval()
    gen = torch.Generator(device=device).manual_seed(seed)
    solved, generated, used_attempts = 0, 0, 0
    t0 = time.perf_counter()
    records = []
    for task in tasks:
        ids = torch.tensor([tok.encode(task["prompt"]).ids], device=device)
        ok = False
        for attempt in range(attempts):
            out = model.generate(ids, max_new_tokens, temperature=temperature, eos_id=tok.token_to_id(EOS), generator=gen)
            completion = tok.decode(out[0, ids.shape[1] :].tolist())
            generated += out.shape[1] - ids.shape[1]
            used_attempts += 1
            if verify(task, completion):
                ok = True
                break
        solved += ok
        records.append({"id": task["id"], "solved": ok, "attempts": attempt + 1})
    elapsed = time.perf_counter() - t0
    return {
        "tasks": len(tasks),
        "solved": solved,
        "success_rate": solved / max(1, len(tasks)),
        "attempts": used_attempts,
        "generated_tokens": generated,
        "tokens_per_solved_task": generated / solved if solved else None,
        "seconds_per_solved_task": elapsed / solved if solved else None,
        "per_task": records,
    }


def read_texts(path: str | Path, field: str = "text", limit: int | None = None) -> list[str]:
    texts = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                texts.append(json.loads(line)[field])
                if limit and len(texts) >= limit:
                    break
    return texts
