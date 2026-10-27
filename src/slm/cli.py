"""Command line interface: ``python -m slm <command>`` (or ``slm`` once installed)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml


def _device(name: str | None) -> torch.device:
    return torch.device(name or ("cuda" if torch.cuda.is_available() else "cpu"))


def _print(obj) -> None:
    print(json.dumps(obj, indent=2, ensure_ascii=False))


# ---- sources and data ------------------------------------------------------------------


def cmd_sources_audit(args) -> int:
    from slm.data.registry import Registry

    reg = Registry.load(args.registry)
    decisions, errors = reg.audit()
    width = max(len(d.source_id) for d in decisions)
    for d in decisions:
        s = reg[d.source_id]
        state = "USABLE " if d.usable else "BLOCKED"
        print(f"{d.source_id:<{width}}  {s.status:<15} {state}  {s.license}")
        for b in d.blockers if args.verbose else []:
            print(f"{'':<{width}}    - {b}")
        for o in d.obligations if args.verbose else []:
            print(f"{'':<{width}}    * {o}")
    for e in errors:
        print(f"ERROR: {e}", file=sys.stderr)
    return 1 if errors else 0


def cmd_data_synth(args) -> int:
    from slm.data import synthetic

    gen = {"prose": synthetic.prose, "arithmetic": synthetic.arithmetic}[args.kind]
    synthetic.write_jsonl(gen(args.n, seed=args.seed), args.out)
    print(f"wrote {args.n} {args.kind} records to {args.out}")
    return 0


def cmd_data_run(args) -> int:
    from slm.data.pipeline import run

    manifest = run(args.config, allow_unapproved=args.allow_unapproved)
    _print(manifest["stats"])
    return 0


def cmd_data_fetch(args) -> int:
    """Download a pinned revision of a registered Hugging Face dataset to JSONL."""
    from datasets import load_dataset

    from slm.data.registry import Registry
    from slm.runlog import sha256_file, write_json

    src = Registry.load(args.registry)[args.source]
    if not src.revision:
        print(f"source {src.id} has no pinned revision; refusing to fetch", file=sys.stderr)
        return 1
    repo = src.url.removeprefix("https://huggingface.co/datasets/").strip("/")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    files = []
    if args.files:
        # Explicit parquet files spread across a dump avoid the ordering bias of taking the first N rows.
        import pyarrow.parquet as pq
        from huggingface_hub import hf_hub_download

        def records():
            for name in args.files:
                local = hf_hub_download(repo, name, repo_type="dataset", revision=src.revision)
                files.append({"file": name, "sha256": sha256_file(local)})
                for batch in pq.ParquetFile(local).iter_batches(batch_size=1024):
                    yield from batch.to_pylist()
    else:
        def records():
            yield from load_dataset(repo, src.subset, split=args.split, revision=src.revision, streaming=True)

    n = 0
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        for rec in records():
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
            n += 1
            if args.limit and n >= args.limit:
                break
    write_json(out.with_suffix(".fetch.json"), {
        "source": src.id, "repo": repo, "subset": src.subset, "split": None if args.files else args.split,
        "files": files, "revision": src.revision, "records": n, "limit": args.limit, "sha256": sha256_file(out),
    })
    print(f"wrote {n} records to {out}")
    return 0


def cmd_data_encode(args) -> int:
    from slm.data.shards import encode

    manifest = encode(args.tokenizer, args.inputs, args.out, val_fraction=args.val_fraction, shard_tokens=args.shard_tokens)
    _print({s: {k: v for k, v in info.items() if k != "shards"} for s, info in manifest["splits"].items()})
    return 0


def cmd_tokenizer_train(args) -> int:
    from slm import tokenizer

    m = tokenizer.train(args.inputs, args.out, vocab_size=args.vocab_size, max_docs=args.max_docs, seed=args.seed)
    print(f"tokenizer: {m['actual_vocab_size']} tokens, sha256 {m['tokenizer_sha256'][:16]}")
    return 0


# ---- model, training, evaluation ---------------------------------------------------------


def cmd_model_info(args) -> int:
    from slm.config import load_model_config
    from slm.model import SLM

    cfg = load_model_config(args.config)
    with torch.device("meta"):
        model = SLM(cfg)
    n, ne = model.num_params(), model.num_params(non_embedding=True)
    fpt = model.flops_per_token(args.seq_len)
    info = {
        "params": n,
        "params_non_embedding": ne,
        "train_flops_per_token": fpt,
        "head_dim": cfg.head_dim,
        "bf16_weights_gb": round(n * 2 / 1e9, 3),
    }
    if args.tokens:
        info["train_flops_total"] = fpt * args.tokens
        info["tokens_per_param"] = round(args.tokens / n, 1)
    _print(info)
    return 0


def _parse_overrides(pairs: list[str]) -> dict:
    out = {}
    for p in pairs:
        key, _, value = p.partition("=")
        out[key] = yaml.safe_load(value)
    return out


def cmd_train(args) -> int:
    from slm.config import load_train_config
    from slm.train import train

    cfg = load_train_config(args.config, _parse_overrides(args.set))
    _print(train(cfg))
    return 0


def cmd_grow(args) -> int:
    from slm.config import load_model_config
    from slm.grow import grow_checkpoint

    _print(grow_checkpoint(args.checkpoint, load_model_config(args.model), args.out, seed=args.seed))
    return 0


def cmd_bench(args) -> int:
    from slm.config import load_model_config, load_train_config
    from slm.train import benchmark

    cfg = load_train_config(args.config, _parse_overrides(args.set))
    model_cfg = load_model_config(args.model or cfg.model)
    _print(benchmark(model_cfg, cfg, steps=args.steps, warmup=args.warmup))
    return 0


def _load(args):
    from slm import tokenizer
    from slm.train import load_model
    from slm.runlog import assert_tokenizer_compatible, sha256_file

    device = _device(args.device)
    if getattr(args, "hf_model", None):
        from slm import hf_adapter

        model, tok = hf_adapter.load(
            args.hf_model, device, revision=args.hf_revision,
            max_seq_len=getattr(args, "max_seq_len", None),
        )
        return model, tok, device
    if not (args.checkpoint and args.tokenizer):
        raise SystemExit("--checkpoint and --tokenizer are required (or --hf-model for a reference model)")
    tokenizer_path = Path(args.tokenizer)
    tokenizer_file = tokenizer_path if tokenizer_path.is_file() else tokenizer_path / "tokenizer.json"
    tokenizer_hash = sha256_file(tokenizer_file)
    tokenizer_lineage = assert_tokenizer_compatible(args.checkpoint, tokenizer_hash)
    return load_model(args.checkpoint, device), tokenizer.load(args.tokenizer), device


def cmd_eval_bpb(args) -> int:
    from slm.evaluation import bits_per_byte, read_texts

    model, tok, device = _load(args)
    _print(bits_per_byte(model, tok, read_texts(args.data, limit=args.limit), device))
    return 0


def cmd_eval_cost(args) -> int:
    from slm.evaluation import measure_cost

    model, tok, device = _load(args)
    prompts = [json.loads(l)["prompt"] for l in open(args.prompts, encoding="utf-8")] if args.prompts else [
        "The research team", "La bibliothèque", "def fibonacci(n):"]
    _print(measure_cost(model, tok, prompts, args.max_new_tokens, device))
    return 0


def cmd_eval_tasks(args) -> int:
    from slm.data.synthetic import verify_arithmetic
    from slm.evaluation import verified_tasks

    model, tok, device = _load(args)
    tasks = [json.loads(l) for l in open(args.tasks, encoding="utf-8") if l.strip()][: args.limit]
    res = verified_tasks(model, tok, tasks, verify_arithmetic, device, attempts=args.attempts,
                         temperature=args.temperature)
    res.pop("per_task")
    _print(res)
    return 0


def cmd_benchmarks_fetch(args) -> int:
    from slm.benchmarks import BENCHMARKS, fetch

    for name in args.names or list(BENCHMARKS):
        print(fetch(name, args.out))
    return 0


def cmd_eval_mc(args) -> int:
    from slm.benchmarks import load_items, multiple_choice
    from slm.runlog import sha256_file

    model, tok, device = _load(args)
    results = {}
    for path in args.benchmarks:
        results[Path(path).stem] = multiple_choice(
            model, tok, load_items(path, args.limit), device, batch_items=args.batch_items
        )
    _print(results)
    if args.out:
        from slm.runlog import write_json

        write_json(args.out, {"checkpoint": args.checkpoint, "hf_model": args.hf_model,
                              "hf_revision": args.hf_revision,
                              "checkpoint_sha256": sha256_file(args.checkpoint) if args.checkpoint else None,
                              "tokenizer": args.tokenizer,
                              "tokenizer_sha256": (sha256_file(Path(args.tokenizer) if Path(args.tokenizer).is_file()
                                                               else Path(args.tokenizer) / "tokenizer.json")
                                                   if args.tokenizer else None),
                              "model_context_cap": model.cfg.max_seq_len,
                              "batch_items": args.batch_items,
                              "benchmarks": [{"path": path, "sha256": sha256_file(path)} for path in args.benchmarks],
                              "results": results})
    return 0


def cmd_data_contamination(args) -> int:
    """Report overlap between existing training documents and evaluation sets, without rebuilding."""
    from collections import Counter

    from slm.data.contamination import ContaminationIndex

    idx = ContaminationIndex(n=args.n)
    for path in args.eval:
        idx.add_jsonl(path, args.fields, "id")
    flagged, per_bench, n = 0, Counter(), 0
    with open(args.docs, encoding="utf-8") as f:
        for line in f:
            n += 1
            hits = idx.matches(json.loads(line)["text"])
            if hits:
                flagged += 1
                per_bench.update({k.split(":", 1)[0] for k in hits})
    _print({"documents": n, "flagged": flagged, "per_benchmark_documents": dict(per_bench), "ngram": args.n})
    return 0


def cmd_generate(args) -> int:
    from slm.tokenizer import EOS

    model, tok, device = _load(args)
    gen = torch.Generator(device=device).manual_seed(args.seed)
    for prompt in args.prompt:
        ids = torch.tensor([tok.encode(prompt).ids], device=device)
        out = model.generate(ids, args.max_new_tokens, temperature=args.temperature, top_k=args.top_k,
                             eos_id=tok.token_to_id(EOS), generator=gen)
        print(f"--- {prompt!r}\n{tok.decode(out[0].tolist())}\n")
    return 0


def cmd_interp_attention(args) -> int:
    from slm import interp

    model, tok, device = _load(args)
    ids = torch.tensor([tok.encode(args.text).ids], device=device)
    stats = interp.attention_statistics(model, ids)
    _print({k: [[round(x, 4) for x in row] for row in v.tolist()] for k, v in stats.items()})
    return 0


def cmd_export(args) -> int:
    from slm.export import export
    from slm.runlog import read_json, sha256_file

    model, _, _ = _load(args)
    run_dir = Path(args.checkpoint).parent.parent
    provenance = {"checkpoint": str(args.checkpoint), "checkpoint_sha256": sha256_file(args.checkpoint)}
    if (run_dir / "run_manifest.json").exists():
        provenance["run_manifest"] = read_json(run_dir / "run_manifest.json")
    print(export(model, args.tokenizer, args.out, provenance))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="slm", description="SLM-1B Challenge tooling")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("sources", help="source registry").add_subparsers(dest="sub", required=True)
    a = s.add_parser("audit", help="evaluate every source against the license policy")
    a.add_argument("--registry", default="configs/data/sources.yaml")
    a.add_argument("-v", "--verbose", action="store_true")
    a.set_defaults(fn=cmd_sources_audit)

    d = sub.add_parser("data", help="data preparation").add_subparsers(dest="sub", required=True)
    a = d.add_parser("synth", help="generate deterministic synthetic data")
    a.add_argument("--kind", choices=["prose", "arithmetic"], default="prose")
    a.add_argument("--n", type=int, default=2000)
    a.add_argument("--seed", type=int, default=0)
    a.add_argument("--out", required=True)
    a.set_defaults(fn=cmd_data_synth)
    a = d.add_parser("run", help="run a pipeline configuration")
    a.add_argument("config")
    a.add_argument("--allow-unapproved", action="store_true")
    a.set_defaults(fn=cmd_data_run)
    a = d.add_parser("fetch", help="download a pinned Hugging Face dataset revision")
    a.add_argument("--source", required=True)
    a.add_argument("--registry", default="configs/data/sources.yaml")
    a.add_argument("--split", default="train")
    a.add_argument("--limit", type=int, default=None)
    a.add_argument("--files", nargs="*", default=None, help="parquet files of the repo to read, in order")
    a.add_argument("--out", required=True)
    a.set_defaults(fn=cmd_data_fetch)
    a = d.add_parser("encode", help="tokenize documents into shards")
    a.add_argument("--tokenizer", required=True)
    a.add_argument("--inputs", nargs="+", required=True)
    a.add_argument("--out", required=True)
    a.add_argument("--val-fraction", type=float, default=0.01)
    a.add_argument("--shard-tokens", type=int, default=100_000_000)
    a.set_defaults(fn=cmd_data_encode)

    t = sub.add_parser("tokenizer", help="tokenizer").add_subparsers(dest="sub", required=True)
    a = t.add_parser("train")
    a.add_argument("--inputs", nargs="+", required=True)
    a.add_argument("--out", required=True)
    a.add_argument("--vocab-size", type=int, default=49152)
    a.add_argument("--max-docs", type=int, default=None)
    a.add_argument("--seed", type=int, default=0)
    a.set_defaults(fn=cmd_tokenizer_train)

    m = sub.add_parser("model", help="model utilities").add_subparsers(dest="sub", required=True)
    a = m.add_parser("info", help="parameter count and compute estimate")
    a.add_argument("config")
    a.add_argument("--seq-len", type=int, default=4096)
    a.add_argument("--tokens", type=float, default=None, help="planned training tokens")
    a.set_defaults(fn=cmd_model_info)

    a = sub.add_parser("train", help="pretrain from a configuration")
    a.add_argument("config")
    a.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    a.set_defaults(fn=cmd_train)

    a = sub.add_parser("grow", help="grow a trained checkpoint into a larger model (function preserving)")
    a.add_argument("--checkpoint", required=True)
    a.add_argument("--model", required=True, help="target model configuration")
    a.add_argument("--out", required=True)
    a.add_argument("--seed", type=int, default=0)
    a.set_defaults(fn=cmd_grow)

    a = sub.add_parser("bench", help="controlled training throughput on random tokens")
    a.add_argument("config", help="training configuration (shapes, optimizer, precision)")
    a.add_argument("--model", default=None, help="model configuration, overriding the one in the training config")
    a.add_argument("--steps", type=int, default=30)
    a.add_argument("--warmup", type=int, default=5)
    a.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    a.set_defaults(fn=cmd_bench)

    def with_model(parser, allow_hf=False):
        parser.add_argument("--checkpoint", default=None, required=not allow_hf)
        parser.add_argument("--tokenizer", default=None, required=not allow_hf)
        if allow_hf:
            parser.add_argument("--hf-model", default=None, help="evaluate a Hugging Face reference model instead")
            parser.add_argument("--hf-revision", default=None)
        parser.add_argument("--device", default=None)
        return parser

    e = sub.add_parser("eval", help="evaluation").add_subparsers(dest="sub", required=True)
    a = with_model(e.add_parser("bpb", help="bits per byte on JSONL text"), allow_hf=True)
    a.add_argument("--data", required=True)
    a.add_argument("--limit", type=int, default=None)
    a.set_defaults(fn=cmd_eval_bpb)
    a = with_model(e.add_parser("cost", help="latency, throughput, memory, energy"))
    a.add_argument("--prompts", default=None)
    a.add_argument("--max-new-tokens", type=int, default=64)
    a.set_defaults(fn=cmd_eval_cost)
    a = with_model(e.add_parser("tasks", help="verified task completion"))
    a.add_argument("--tasks", required=True)
    a.add_argument("--limit", type=int, default=None)
    a.add_argument("--attempts", type=int, default=1)
    a.add_argument("--temperature", type=float, default=0.0)
    a.set_defaults(fn=cmd_eval_tasks)

    a = with_model(e.add_parser("mc", help="multiple-choice benchmarks by log-likelihood"), allow_hf=True)
    a.add_argument("--benchmarks", nargs="+", required=True)
    a.add_argument("--limit", type=int, default=None)
    a.add_argument("--batch-items", type=int, default=16,
                   help="items scored per forward batch; reduce this to lower evaluation GPU memory")
    a.add_argument("--max-seq-len", type=int, default=None,
                   help="cap reference-model context for matched evaluation")
    a.add_argument("--out", default=None)
    a.set_defaults(fn=cmd_eval_mc)
    a = e.add_parser("fetch-benchmarks", help="download pinned benchmark revisions")
    a.add_argument("names", nargs="*")
    a.add_argument("--out", default="data/benchmarks")
    a.set_defaults(fn=cmd_benchmarks_fetch)
    a = d.add_parser("contamination", help="n-gram overlap between documents and evaluation sets")
    a.add_argument("--docs", required=True)
    a.add_argument("--eval", nargs="+", required=True)
    a.add_argument("--fields", nargs="+", default=["context", "choices"])
    a.add_argument("--n", type=int, default=13)
    a.set_defaults(fn=cmd_data_contamination)

    a = with_model(sub.add_parser("generate", help="sample continuations of prompts"))
    a.add_argument("--prompt", nargs="+", required=True)
    a.add_argument("--max-new-tokens", type=int, default=80)
    a.add_argument("--temperature", type=float, default=0.8)
    a.add_argument("--top-k", type=int, default=50)
    a.add_argument("--seed", type=int, default=0)
    a.set_defaults(fn=cmd_generate)

    i = sub.add_parser("interp", help="interpretability").add_subparsers(dest="sub", required=True)
    a = with_model(i.add_parser("attention", help="attention sink mass and entropy per head"))
    a.add_argument("--text", required=True)
    a.set_defaults(fn=cmd_interp_attention)

    a = with_model(sub.add_parser("export", help="export to Hugging Face format"))
    a.add_argument("--out", required=True)
    a.set_defaults(fn=cmd_export)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
