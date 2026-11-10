"""Assemble the story-model evidence and a local Hugging Face package; never upload anything.

Inputs are the completed EXP-0045 run, its untouched export, and existing local evaluation files.
All verification and inference run on CPU. Real training stories and optimizer checkpoints are excluded.
The source's candidate status is preserved. Public release requires the outstanding publication decision.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import shutil
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from slm.data.pii import redact
from slm.evaluation import bits_per_byte, read_texts
from slm.export import hf_state_dict
from slm.hf_adapter import HFModel, HFTokenizer
from slm.interp import head_ablation_effects, target_logprob_metric
from slm.runlog import environment_info, sha256_file
from slm.train import load_model

RUN = ROOT / "runs/EXP-0045/story-935k-100m-v2-s421201"
EXPORT = ROOT / "data/models/fluency/story-935k-100m-v2"
CORPUS = ROOT / "data/processed/exp0045-simple-stories-v2"
COMPARISON = ROOT / "runs/EXP-0045/compare-320-bytes-100m-v2"
EVAL = ROOT / "runs/EXP-0045/evaluations-100m-v2"
DOCS = ROOT / "docs/model-cards/slm-tavenel-story-935k"
PROMPTS = ["Once upon a time, a fox found a key.",
           "Lily was afraid of the dark, so her brother gave her a small lamp."]


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def source_audit(record: Path) -> dict:
    """Check source-group / text overlap and count structured-identifier flags; do not copy story text."""
    groups = {"train": set(), "dev": set()}
    trace = CORPUS / "provenance.jsonl"
    count = 0
    with trace.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            groups[row["split"]].add(row["completion_group"])
            count += 1
    split_report, texts = {}, {}
    for split in ("train", "dev"):
        hashes, normalized, flags = set(), set(), {"email": 0, "ip": 0, "phone": 0, "iban": 0}
        documents = 0
        for line in (CORPUS / f"{split}.jsonl").read_text(encoding="utf-8").splitlines():
            text = json.loads(line)["text"]
            hashes.add(hashlib.sha256(text.encode()).hexdigest())
            normalized.add(hashlib.sha256(" ".join(text.casefold().split()).encode()).hexdigest())
            _, found = redact(text)
            for kind, n in found.items():
                flags[kind] += n
            documents += 1
        texts[split] = hashes, normalized
        split_report[split] = {"documents": documents, "unique_exact_texts": len(hashes),
                               "unique_normalized_texts": len(normalized), "structured_identifier_hits": flags}
    report = {"kind": "slm.story_source_audit", "documents": count, "splits": split_report,
              "shared_completion_groups": len(groups["train"] & groups["dev"]),
              "shared_exact_texts": len(texts["train"][0] & texts["dev"][0]),
              "shared_case_whitespace_normalized_texts": len(texts["train"][1] & texts["dev"][1]),
              "limits": ["No semantic or near-duplicate independence claim", "Regex flags are not a privacy clearance",
                         "Source remains candidate; producer terms are documented separately"]}
    if report["shared_completion_groups"]:
        raise ValueError("Completion-group leakage in the release data")
    # Deterministic compressed source index: IDs, source rows, groups and hashes, never training text.
    with gzip.GzipFile(filename="", mode="wb", fileobj=(record / "document-index.jsonl.gz").open("wb"), mtime=0) as stream:
        stream.write(trace.read_bytes())
    write(record / "source-audit.json", report)
    return report


@torch.inference_mode()
def verify_and_measure(record: Path) -> dict:
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import transformers

    torch.set_num_threads(2)
    provenance = load(EXPORT / "provenance.json")
    for name, expected in provenance["files"].items():
        if sha256_file(EXPORT / name) != expected:
            raise ValueError(f"Export file has changed: {name}")
    checkpoint = ROOT / provenance["checkpoint"]
    if sha256_file(checkpoint) != provenance["checkpoint_sha256"]:
        raise ValueError("Native checkpoint hash mismatch")
    tokenizer = AutoTokenizer.from_pretrained(EXPORT, local_files_only=True)
    hf = AutoModelForCausalLM.from_pretrained(EXPORT, local_files_only=True, dtype=torch.float32).eval()
    native = load_model(checkpoint, "cpu").eval()
    expected, actual = hf_state_dict(native, torch.bfloat16), load_file(EXPORT / "model.safetensors")
    if set(expected) != set(actual) or any(not torch.equal(expected[k], actual[k]) for k in expected):
        raise ValueError("Published tensors do not match the documented native-to-bfloat16 mapping")
    if sum(p.numel() for p in hf.parameters()) != 935040:
        raise ValueError("Model parameter budget changed")
    ids = [tokenizer(p, return_tensors="pt", return_token_type_ids=False)["input_ids"] for p in PROMPTS]
    fp32_diff = [float((native(x)[0] - hf(x).logits).abs().max()) for x in ids]
    for parameter in native.parameters():
        parameter.data = parameter.data.to(torch.bfloat16).float()
    rounded_diff = [float((native(x)[0] - hf(x).logits).abs().max()) for x in ids]
    if max(rounded_diff) > 1e-4:
        raise ValueError(f"Rounded-weight export logit parity failed: {rounded_diff}")
    cache_identical = []
    examples, ablations = [], []
    for prompt, x in zip(PROMPTS, ids):
        without = hf.generate(x, max_new_tokens=32, do_sample=False, use_cache=False, pad_token_id=1)
        cached = hf.generate(x, max_new_tokens=32, do_sample=False, use_cache=True, pad_token_id=1)
        cache_identical.append(torch.equal(without, cached))
        torch.manual_seed(0)
        sampled = hf.generate(x, max_new_tokens=160, do_sample=True, temperature=0.8, top_p=0.9, pad_token_id=1)
        generated = sampled[0, x.shape[1]:].tolist()
        examples.append({"prompt": prompt, "seed": 0, "temperature": 0.8, "top_p": 0.9,
                         "max_new_tokens": 160, "prompt_token_ids": x[0].tolist(),
                         "generated_token_ids": generated, "completion": tokenizer.decode(generated, skip_special_tokens=True)})
        target = int(native(x)[0][0, -1].argmax())
        effects = head_ablation_effects(native, x, target_logprob_metric(-1, target))
        ablations.append({"prompt": prompt, "target_token_id": target, "target_piece": tokenizer.decode([target]),
                          "delta_log_probability_by_layer_head": effects.tolist()})
    if not all(cache_identical):
        raise ValueError("Cached and uncached greedy decoding diverged on the fixed probes")
    write(record / "examples.json", {"selection": "Both fixed user-provided prompts; one sample each, seed 0, no reranking",
                                    "examples": examples})
    write(record / "head-ablations.json", {"kind": "slm.head_zero_ablation", "model": "native checkpoint rounded to exported bfloat16 weights, then CPU float32",
                                         "metric": "change in log probability of the baseline top next token",
                                         "limits": "Two fixed prompts; causal effect on one next-token score, not a semantic explanation of a story",
                                         "probes": ablations})
    print("Verified tensor mapping, rounded-weight logit parity, and cache equivalence", flush=True)
    # Reuse the repository's existing BPB evaluator to score the exact exported artifact.
    dev = read_texts(CORPUS / "dev.jsonl")
    scored = bits_per_byte(HFModel(hf, 256), HFTokenizer(tokenizer), dev, torch.device("cpu"), max_len=256)
    write(record / "exported-dev-bpb.json", {"kind": "slm.story_export_bpb", "context": 256,
                                           "documents": len(dev), "model_sha256": sha256_file(EXPORT / "model.safetensors"),
                                           "data_sha256": sha256_file(CORPUS / "dev.jsonl"), **scored})
    # Existing operational measurements: one warmup, then three timed CPU repetitions.
    x = tokenizer("A little fox found a small key. " * 30, return_tensors="pt")["input_ids"][:, :128]
    options = dict(max_new_tokens=64, min_new_tokens=64, do_sample=False, use_cache=True, pad_token_id=1)
    hf.generate(x, **options)
    prefill, totals = [], []
    for _ in range(3):
        t0 = time.perf_counter(); hf(x, use_cache=True); prefill.append((time.perf_counter() - t0) * 1000)
        t0 = time.perf_counter(); output = hf.generate(x, **options); totals.append(time.perf_counter() - t0)
        if output.shape[1] - x.shape[1] != 64:
            raise ValueError("Timing did not generate the declared number of tokens")
    try:
        import psutil
        rss = psutil.Process().memory_info().rss
    except ImportError:
        rss = None
    checks = {"kind": "slm.story_release_verification", "parameters": 935040, "tensor_mapping_exact": True,
              "max_abs_logit_diff_original_fp32_vs_export": fp32_diff,
              "max_abs_logit_diff_bfloat16_rounded_native_vs_export": rounded_diff,
              "cached_uncached_greedy_identical": cache_identical,
              "environment": {**environment_info(), "transformers": transformers.__version__},
              "inference": {"device": "cpu", "threads": 2, "loaded_dtype": "float32", "batch": 1,
                            "prompt_tokens": 128, "new_tokens": 64, "warmup": 1, "timed_runs": 3,
                            "prefill_ms_median": statistics.median(prefill), "generate_seconds_median": statistics.median(totals),
                            "generated_tokens_per_second": 64 / statistics.median(totals),
                            "process_rss_bytes_after_checks": rss, "rss_note": "Includes Python, both model copies and benchmark allocations; not isolated model memory or peak RAM",
                            "limits": "One CPU machine/session; cached greedy generation; excludes model loading"}}
    write(record / "verification.json", checks)
    print("Exported artifact BPB:", scored["bits_per_byte"], flush=True)
    return checks


def figures(record: Path, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "svg.hashsalt": "story935k"})
    out.mkdir(parents=True, exist_ok=True)

    def save(fig, name):
        fig.savefig(out / f"{name}.png", dpi=170, bbox_inches="tight")
        fig.savefig(out / f"{name}.svg", bbox_inches="tight")
        plt.close(fig)

    metrics = [json.loads(s) for s in (record / "metrics.jsonl").read_text().splitlines()]
    val = [json.loads(s) for s in (record / "eval.jsonl").read_text().splitlines()]
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot([r["step"] for r in metrics], [r["loss"] for r in metrics], lw=1, alpha=.6, label="Training minibatch loss")
    ax.plot([r["step"] for r in val], [r["val_loss"] for r in val], "o-", ms=3, label="Sampled validation loss")
    ax.set(xlabel="Optimizer updates", ylabel="Next-token cross-entropy (nats)", title="SLM-Tavenel Story 935K: one run, seed 421201")
    ax.legend(frameon=False); save(fig, "training")

    bpb = load(record / "exported-dev-bpb.json")["bits_per_byte"]
    tiny = load(record / "tiny1m-dev-bpb.json")["bits_per_byte"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    axes[0].barh(["SLM-Tavenel Story 935K\n(exported weights)", "TinyStories tiny1m"], [bpb, tiny], color=["#2563eb", "#94a3b8"])
    for i, n in enumerate([bpb, tiny]): axes[0].text(n+.02, i, f"{n:.3f}", va="center")
    axes[0].set(xlim=(0, 1.9), xlabel="Bits per byte, lower is better", title="Local SimpleStories dev\n3,185 texts; 256-token chunks")
    context = load(record / "native-bpb-by-context.json")
    ns = [context[str(n)]["bits_per_byte"] for n in (256, 512, 1024)]
    axes[1].bar(["256", "512", "1,024"], ns, color=["#2563eb", "#93c5fd", "#bfdbfe"])
    axes[1].set(xlabel="Scoring chunk length", ylabel="Bits per byte", title="Context diagnostic\nNative Story 935K checkpoint")
    fig.text(.5, -.06, "Exploratory domain comparison. Training corpora differ; no confidence interval or narrative-quality ranking.", ha="center", fontsize=9)
    fig.tight_layout(); save(fig, "benchmarks")

    obs = [json.loads(s) for s in (record / "observability.jsonl").read_text().splitlines()][-1]
    fig, axes = plt.subplots(1, 3, figsize=(10, 3))
    for ax, key, label in zip(axes, ("attn_entropy_mean", "sink_mass_mean", "resid_rms"), ("Attention entropy (nats)", "Attention on first token", "Residual RMS")):
        ax.bar(range(1, 5), obs[key], color="#2563eb"); ax.set(xlabel="Layer", title=label, xticks=range(1, 5))
    fig.suptitle(f"Logged training probe, update {obs['step']}; descriptive diagnostics")
    fig.tight_layout(); save(fig, "observability")

    probes = load(record / "head-ablations.json")["probes"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    scale = max(abs(v) for p in probes for row in p["delta_log_probability_by_layer_head"] for v in row)
    for i, (ax, probe) in enumerate(zip(axes, probes)):
        im = ax.imshow(probe["delta_log_probability_by_layer_head"], cmap="RdBu", vmin=-scale, vmax=scale)
        ax.set(xlabel="Attention head", ylabel="Layer", xticks=range(4), yticks=range(4),
               title=f"Fixed prompt {i+1}\nBaseline target: token {probe['target_token_id']}")
    fig.colorbar(im, ax=list(axes), label="Change in target log probability (nats)", shrink=.8)
    save(fig, "head-ablations")

    fig, ax = plt.subplots(figsize=(11, 4.3)); ax.set(xlim=(0, 12), ylim=(0, 4.5)); ax.axis("off")
    boxes = [(0, 1.3, 2, 1.5, "Byte-level BPE\n1,024 tokens\nEnglish story text"),
             (2.6, 1.3, 2, 1.5, "Tied embedding\n1,024 x 128\n131,072 weights"),
             (5.2, .65, 3.4, 2.8, "4 decoder blocks\nRMSNorm -> GQA (RoPE)\n4 Q / 1 KV heads, head dim 32\n+ residual connection\nRMSNorm -> SwiGLU 416\n+ residual connection"),
             (9.2, 1.3, 2.6, 1.5, "Final RMSNorm\nShared output matrix\nNext-token logits")]
    for x, y, w, h, text in boxes:
        ax.add_patch(FancyBboxPatch((x+.05, y), w-.1, h, boxstyle="round,pad=0.05", facecolor="#eff6ff", edgecolor="#2563eb"))
        ax.text(x+w/2, y+h/2, text, ha="center", va="center", fontsize=10)
    for a, b in ((2, 2.6), (4.6, 5.2), (8.6, 9.2)):
        ax.annotate("", (b, 2.05), (a, 2.05), arrowprops={"arrowstyle": "->", "color": "#1e3a8a", "lw": 1.5})
    ax.text(6, 4.1, "SLM-Tavenel Story 935K | 935,040 unique parameters | standard Llama export", ha="center", weight="bold", fontsize=13)
    ax.text(6, .1, "Training window: 256 tokens. Configured positional limit: 1,024; quality degrades beyond training length.", ha="center", fontsize=10)
    save(fig, "architecture")


def render_card(record: Path) -> None:
    path = DOCS / "README.md"
    text = path.read_text(encoding="utf-8")
    exported = load(record / "exported-dev-bpb.json")
    tiny = load(record / "tiny1m-dev-bpb.json")
    table = ("| Model | Unique parameters | Scoring context | Dev BPB |\n"
             "|---|---:|---:|---:|\n"
             f"| **SLM-Tavenel Story 935K, exported weights** | 935,040 | 256 | **{exported['bits_per_byte']:.4f}** |\n"
             f"| [TinyStories tiny1m](https://huggingface.co/shibatch/tiny1m/tree/fdacad8ff9dc27bd72c77d3684075a494be1ba11) | 935,040 | 256 | {tiny['bits_per_byte']:.4f} |")
    examples = []
    for i, row in enumerate(load(record / "examples.json")["examples"], 1):
        examples.append(f"**Prompt {i}: {row['prompt']}**\n\n" + "\n".join("> " + line for line in row["completion"].splitlines()))
    info = load(record / "verification.json")["inference"]
    inference = ("| CPU, two threads; batch 1 | Measured value |\n|---|---:|\n"
                 f"| Prefill, 128 tokens | {info['prefill_ms_median']:.2f} ms |\n"
                 f"| 64 generated tokens, total | {info['generate_seconds_median']:.3f} s |\n"
                 f"| Cached greedy generation | {info['generated_tokens_per_second']:.1f} tokens/s |\n\n"
                 "Median of three runs after one warmup, CPU float32. The generated sequences are forced to 64 new tokens for timing. "
                 "This is one machine/session and excludes loading; it is not a hardware-independent speed claim.")
    for marker, content in (("benchmark-table", table), ("examples", "\n\n".join(examples)), ("inference", inference)):
        pattern = rf"<!-- generated:{marker} -->.*?<!-- /generated:{marker} -->"
        text, count = re.subn(pattern, lambda _: f"<!-- generated:{marker} -->\n{content}\n<!-- /generated:{marker} -->", text, flags=re.S)
        if count != 1:
            raise ValueError(f"Expected one {marker} block in the model card")
    path.write_text(text, encoding="utf-8", newline="\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "data/models/fluency/slm-tavenel-story-935k-release")
    parser.add_argument("--reuse-measurements", action="store_true", help="Only rebuild figures/package from existing verified measurements")
    args = parser.parse_args()
    DOCS.mkdir(parents=True, exist_ok=True)
    record = DOCS / "training_record"; record.mkdir(exist_ok=True)
    for name in ("run_manifest.json", "metrics.jsonl", "eval.jsonl", "observability.jsonl", "energy_meter_00000000.json"):
        copy(RUN / name, record / name)
    copy(EXPORT / "provenance.json", record / "source-export-provenance.json")
    copy(CORPUS / "manifest.json", record / "corpus-manifest.json")
    copy(ROOT / "data/tokens/exp0045-simple-stories-v2/manifest.json", record / "token-shards-manifest.json")
    copy(ROOT / "data/processed/exp0045-tokenizer-v2-s421201/manifest.json", record / "tokenizer-manifest.json")
    copy(EVAL / "story-935k-dev-bpb-by-context.json", record / "native-bpb-by-context.json")
    copy(EVAL / "tiny1m-dev-bpb.json", record / "tiny1m-dev-bpb.json")
    for name in ("manifest.json", "generations.jsonl", "blind_review.csv", "review_key.json"):
        copy(COMPARISON / name, record / "generation-comparison" / name)
    copy(ROOT / "runs/PR1-fixes-2026-11-10/snapshots-story100m.json", record / "generation-comparison/snapshots.json")
    if not args.reuse_measurements or not (record / "source-audit.json").exists():
        source_audit(record)
    if not args.reuse_measurements:
        verify_and_measure(record)
    figures(record, DOCS / "figures")
    render_card(record)
    args.out.mkdir(parents=True, exist_ok=True)
    for path in EXPORT.iterdir():
        if path.is_file(): copy(path, args.out / path.name)
    for path in DOCS.rglob("*"):
        if path.is_file() and path.name != "package-manifest.json": copy(path, args.out / path.relative_to(DOCS))
    copy(ROOT / "LICENSE", args.out / "LICENSE")
    write(DOCS / "package-manifest.json", {"kind": "slm.story_hf_package", "repo_id": "athenista/slm-tavenel-story-935k",
                                         "visibility": "private_pending_publication_decision", "weights_in_git": False,
                                         "files": {str(p.relative_to(args.out)).replace('\\', '/'): sha256_file(p) for p in sorted(args.out.rglob('*')) if p.is_file() and p.name != "package-manifest.json"}})
    copy(DOCS / "package-manifest.json", args.out / "package-manifest.json")
    print("Prepared package:", args.out, flush=True)


if __name__ == "__main__":
    main()
