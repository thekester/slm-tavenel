"""Auditable free-generation diagnostics; repetition is not a quality score.

Decode one unpadded prompt at a time, without a chat template or hidden prompt
rewrites. Stop at a UTF-8 byte budget (possibly overshot by the final token), an
EOS token, the model's context limit, or a separate token safety cap. This avoids
silently comparing 128 large BPE tokens with 128 much smaller pieces.
"""

from __future__ import annotations

import math
import re
import time
from typing import Any

import torch


def text_diagnostics(text: str) -> dict[str, Any]:
    words = re.findall(r"\w+", text.casefold(), flags=re.UNICODE)
    triples = list(zip(words, words[1:], words[2:]))
    return {
        "utf8_bytes": len(text.encode("utf-8")),
        "words": len(words),
        "repeated_word_trigram_fraction": (
            1 - len(set(triples)) / len(triples) if triples else None
        ),
        "replacement_characters": text.count("\ufffd"),
    }


def sample_token(logits, temperature, top_p, generator):
    if temperature == 0:
        return logits.argmax().reshape(1)
    scores, indices = torch.sort(logits.float() / temperature, descending=True)
    probs = scores.softmax(-1)
    # Keep the token that crosses the nucleus threshold, including the top token.
    remove = probs.cumsum(-1) - probs >= top_p
    scores = scores.masked_fill(remove, -torch.inf)
    return indices[torch.multinomial(scores.softmax(-1), 1, generator=generator)]


@torch.inference_mode()
def generate_sample(model, tokenizer, prompt: str, *, max_new_bytes: int = 320,
                    max_new_tokens: int = 512, temperature: float = 0,
                    top_p: float = 0.9, seed: int = 0) -> dict[str, Any]:
    if not prompt or max_new_bytes <= 0 or max_new_tokens <= 0:
        raise ValueError("A nonempty prompt and positive byte/token budgets are required")
    if not math.isfinite(temperature) or temperature < 0 or not 0 < top_p <= 1:
        raise ValueError("temperature must be finite and >= 0; top_p must be in (0, 1]")
    model.eval()
    device = next(model.parameters()).device
    context_limit = int(model.config.max_position_embeddings)
    ids = tokenizer.encode(prompt, add_special_tokens=True)
    if not ids or len(ids) >= context_limit:
        raise ValueError("Prompt is empty after tokenization or leaves no context for generation")
    eos = model.config.eos_token_id
    eos_ids = set(eos if isinstance(eos, list) else [eos]) - {None}
    generator = torch.Generator(device=device).manual_seed(seed)
    x = torch.tensor([ids], dtype=torch.long, device=device)
    generated, text, reason = [], "", "token_cap"
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    for _ in range(max_new_tokens):
        logits = model(input_ids=x, use_cache=False).logits[0, -1]
        if not torch.isfinite(logits).all():
            raise ValueError("Nonfinite logits; refusing to produce a diagnostic sample")
        token = int(sample_token(logits, temperature, top_p, generator).item())
        generated.append(token)
        text = tokenizer.decode(generated, skip_special_tokens=True,
                                clean_up_tokenization_spaces=False)
        if token in eos_ids:
            reason = "eos"
            break
        if len(text.encode("utf-8")) >= max_new_bytes:
            reason = "byte_budget"
            break
        if len(ids) + len(generated) >= context_limit:
            reason = "context_limit"
            break
        x = torch.cat([x, x.new_tensor([[token]])], dim=1)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    return {
        "prompt": prompt, "prompt_token_ids": ids,
        "generated_token_ids": generated, "completion": text,
        "stop_reason": reason, "context_limit": context_limit,
        "seed": seed, "temperature": temperature, "top_p": top_p,
        "max_new_bytes": max_new_bytes, "max_new_tokens": max_new_tokens,
        "wall_seconds": elapsed, **text_diagnostics(text),
    }
