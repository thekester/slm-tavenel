#!/usr/bin/env python3
"""LFM3-VL-2B with a compiled computer as its calculator (route B, separate stream; see
experiments/AXIS-compiled-computer-qwenputer-2026-11-05.md).

LFM3 writes its native XML tool call (``<tool_call><function=multiply><parameter=a>..``); the call is
answered either by nothing (``--tool none``: the model must answer alone), by host Python (``python``)
or by :class:`slm.computer.CompiledMachine` running ``ADD_PROGRAM`` / ``MUL_PROGRAM`` (``compiled``);
the result goes back as a ``tool`` message and LFM3 writes the final answer. LFM3's weights are not
changed. Needs ``transformers>=5.21`` (run with PYTHONPATH=data/envs/tf5) and the pinned snapshot in
``data/models/lfm3-vl-2b-71d8e745`` (LFM Open License v1.0).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from slm.computer import ADD_PROGRAM, MUL_PROGRAM, CompiledMachine  # noqa: E402

MODEL = ROOT / "data/models/lfm3-vl-2b-71d8e745"
TOOLS = [
    {"type": "function", "function": {"name": name, "description": f"Exact {what} of two non-negative integers.",
                                      "parameters": {"type": "object",
                                                     "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                                                     "required": ["a", "b"]}}}
    for name, what in (("add", "sum"), ("multiply", "product"))
]
CALL = re.compile(r"<tool_call>\s*<function=(\w+)>(.*?)</function>\s*</tool_call>", re.S)
PARAM = re.compile(r"<parameter=(\w+)>\s*(.*?)\s*</parameter>", re.S)


def digits(n: int) -> list[int]:
    return [int(d) for d in reversed(str(n))]


def compiled(name: str, a: int, b: int, machine: CompiledMachine) -> tuple[int, int]:
    """Run the operation on the compiled machine; returns (result, trace tokens)."""
    if name == "add":
        machine.boot(ADD_PROGRAM, digits(a), digits(b), registers={2: max(len(str(a)), len(str(b)))})
    else:
        machine.boot(MUL_PROGRAM, digits(a), digits(b), registers={2: len(str(b)), 13: len(str(a))})
    machine.run()
    return machine.result(), len(machine.trace)


SYSTEM = "You are a helpful assistant."
# Without this sentence LFM3-VL-2B often declines when two tools are declared (observed 2026-11-06).
SYSTEM_TOOLS = SYSTEM + " For any arithmetic on integers, call the provided tools instead of computing by hand."


def answer(model, processor, question: str, tool: str, machine: CompiledMachine, max_new_tokens: int = 200) -> dict:
    system = SYSTEM if tool == "none" else SYSTEM_TOOLS
    messages = [{"role": "system", "content": [{"type": "text", "text": system}]},
                {"role": "user", "content": [{"type": "text", "text": question}]}]
    record = {"question": question, "tool": tool, "calls": []}

    def generate():
        inputs = processor.apply_chat_template(messages, tools=TOOLS if tool != "none" else None,
                                               add_generation_prompt=True, tokenize=True, return_dict=True,
                                               return_tensors="pt", enable_thinking=False)
        with torch.no_grad():
            out = model.generate(**inputs.to(model.device), max_new_tokens=max_new_tokens, do_sample=False)
        # Keep special tokens: <tool_call> is one; drop only the end-of-turn marker.
        text = processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=False)[0]
        return text.split("<|im_end|>")[0]

    text = generate()
    match = CALL.search(text) if tool != "none" else None
    if match:
        name, params = match.group(1), dict(PARAM.findall(match.group(2)))
        call = {"function": name, "arguments": params}
        try:
            a, b = int(params["a"]), int(params["b"])
            if name not in ("add", "multiply") or a < 0 or b < 0:
                raise ValueError(f"unsupported call {name}({a}, {b})")
            if tool == "python":
                result = a + b if name == "add" else a * b
            else:
                result, call["trace_tokens"] = compiled(name, a, b, machine)
            call["result"] = result
        except (KeyError, ValueError) as error:
            result, call["error"] = f"error: {error}", str(error)
        record["calls"].append(call)
        messages.append({"role": "assistant", "content": "",
                         "tool_calls": [{"type": "function", "function": {"name": name, "arguments": params}}]})
        messages.append({"role": "tool", "content": str(result)})
        text = generate()
    record["final"] = text.strip()
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--question", action="append", required=True)
    parser.add_argument("--tool", choices=("none", "python", "compiled"), default="compiled")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(MODEL, trust_remote_code=True,
                                                        dtype=torch.bfloat16).to(args.device).eval()
    machine = CompiledMachine()
    for question in args.question:
        print(json.dumps(answer(model, processor, question, args.tool, machine), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
