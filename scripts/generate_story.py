"""Generate a raw continuation with the standard Llama story checkpoint, on CPU by default."""
from __future__ import annotations

import argparse
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompt", nargs="?", default="Once upon a time, a fox found a key.")
    parser.add_argument("--model", default="data/models/fluency/story-935k-100m-v2")
    parser.add_argument("--revision", help="Full Hub revision when loading a remote model")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--json", action="store_true", help="Include prompt, token IDs and decoding settings")
    args = parser.parse_args()
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be positive")
    torch.set_num_threads(2)
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    model = AutoModelForCausalLM.from_pretrained(args.model, revision=args.revision, dtype=torch.float32).eval()
    torch.manual_seed(args.seed)
    inputs = tokenizer(args.prompt, return_tensors="pt", return_token_type_ids=False)
    if inputs["input_ids"].shape[1] + args.max_new_tokens > 256:
        parser.error("Use at most 256 prompt + continuation tokens: this model was trained with 256-token windows")
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=True,
                                temperature=0.8, top_p=0.9, pad_token_id=tokenizer.pad_token_id)
    new_ids = output[0, inputs["input_ids"].shape[1]:].tolist()
    text = tokenizer.decode(new_ids, skip_special_tokens=True)
    if args.json:
        print(json.dumps({"prompt": args.prompt, "completion": text, "generated_token_ids": new_ids,
                          "seed": args.seed, "temperature": 0.8, "top_p": 0.9,
                          "max_new_tokens": args.max_new_tokens}, ensure_ascii=False, indent=2))
    else:
        print(args.prompt + text)


if __name__ == "__main__":
    main()
