"""Build local, auditable story-mixture pilots; never train on official validation.

The published EXP-0045 tokenizer is frozen to isolate the data intervention.
TinyStories official validation and the old SimpleStories dev groups are excluded.
New dev/test documents come from source train partitions. Test stays JSONL-only,
outside every trainer manifest, and is not scored by the experiment runner.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import shutil
import sys
import unicodedata

import torch  # Before pyarrow on Windows.

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from slm.data.shards import encode_fixed_splits
from slm.runlog import code_revision, sha256_file, write_json

SIMPLE_REV = "e63b8adc3b1a1bdc7cac5b500d150b71346b0628"
TINY_REV = "f54c09fd23315a6f9c86f9dc80f725de7d8f9c64"
TINY_TRAIN = "data/train-00000-of-00004-2d5a1467fff1081b.parquet"
TINY_VAL = "data/validation-00000-of-00001-869c898b519ad725.parquet"
SEED = 421207
TOKENIZER = ROOT / "data/processed/exp0045-tokenizer-v2-s421201/tokenizer.json"
DATA = ROOT / "data/processed/exp0047-story-mix"
TOKENS = ROOT / "data/tokens/exp0047-story-mix"


def normalized_hash(text: str) -> str:
    text = " ".join(unicodedata.normalize("NFC", text).casefold().split())
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def completion_group(gid: str) -> str:
    head, sep, tail = gid.rpartition("-")
    return head if sep and head and tail.isdigit() else gid


def assigned_split(group: str) -> str:
    # Reuses the old SimpleStories hash buckets: its train groups (>=5) cannot
    # enter the new dev/test (<5), even if other stories share the completion.
    bucket = int.from_bytes(hashlib.sha256(group.encode()).digest()[:8], "big") % 100
    return "dev" if bucket < 2 else "test" if bucket < 5 else "train"


class Reservoir:
    def __init__(self, capacity: int, seed: int):
        self.capacity, self.seen = capacity, 0
        self.rng, self.items = random.Random(seed), []

    def add(self, item: dict) -> None:
        self.seen += 1
        if len(self.items) < self.capacity:
            self.items.append(item)
        else:
            index = self.rng.randrange(self.seen)
            if index < self.capacity:
                self.items[index] = item


def build_partitions(train_docs: int, dev_docs: int, test_docs: int) -> dict:
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    DATA.mkdir(parents=True, exist_ok=True)
    old = ROOT / "data/processed/exp0045-simple-stories-v2"
    excluded_groups = set()
    with (old / "provenance.jsonl").open(encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec["split"] == "dev":
                excluded_groups.add(rec["completion_group"])
    excluded_hashes, old_train_hashes = set(), set()
    for split in ("train", "dev"):
        with (old / f"{split}.jsonl").open(encoding="utf-8") as f:
            dest = old_train_hashes if split == "train" else excluded_hashes
            dest.update(normalized_hash(json.loads(line)["text"]) for line in f)
    validation = Path(hf_hub_download("roneneldan/TinyStories", TINY_VAL, repo_type="dataset",
                                     revision=TINY_REV, local_dir=ROOT / "data/raw/tinystories-f54c09fd"))
    for batch in pq.ParquetFile(validation).iter_batches(columns=["text"]):
        excluded_hashes.update(normalized_hash(t) for t in batch.column(0).to_pylist())
    sources = [
        ("simple", "SimpleStories/SimpleStories", SIMPLE_REV,
         [f"data/train-{i:05d}-of-00007.parquet" for i in (0, 1)], "story"),
        ("tiny", "roneneldan/TinyStories", TINY_REV, [TINY_TRAIN], "text"),
    ]
    counts, seen, files, output = Counter(), set(), [], {}
    for source_i, (name, repo, revision, names, text_key) in enumerate(sources):
        capacities = {"train": train_docs, "dev": dev_docs, "test": test_docs}
        reservoirs = {s: Reservoir(n, SEED + source_i * 10 + i) for i, (s, n) in enumerate(capacities.items())}
        for filename in names:
            local_dir = ROOT / ("data/raw/simple-stories-e63b8adc" if name == "simple" else "data/raw/tinystories-f54c09fd")
            path = Path(hf_hub_download(repo, filename, repo_type="dataset", revision=revision, local_dir=local_dir))
            files.append({"repo": repo, "revision": revision, "file": filename, "sha256": sha256_file(path)})
            columns = [text_key, "generation_id"] if name == "simple" else [text_key]
            row_index = -1
            for batch in pq.ParquetFile(path).iter_batches(columns=columns, batch_size=4096):
                for rec in batch.to_pylist():
                    row_index += 1
                    text = rec.get(text_key)
                    if not isinstance(text, str) or len(text.strip()) < 80:
                        counts[f"{name}_invalid"] += 1
                        continue
                    digest = normalized_hash(text)
                    gid = str(rec.get("generation_id") or "")
                    if name == "simple" and not gid:
                        counts["simple_missing_group"] += 1
                        continue
                    group = completion_group(gid) if name == "simple" else digest
                    if digest in excluded_hashes or (name == "simple" and group in excluded_groups):
                        counts[f"{name}_old_dev_or_official_validation_excluded"] += 1
                        continue
                    split = assigned_split(group)
                    if split != "train" and digest in old_train_hashes:
                        counts[f"{name}_old_train_excluded_from_new_holdout"] += 1
                        continue
                    if digest in seen:
                        counts[f"{name}_duplicate_excluded"] += 1
                        continue
                    seen.add(digest)
                    reservoirs[split].add({"id": f"{name}:{filename}:{row_index}", "text": text,
                                           "normalized_sha256": digest, "source": name,
                                           "source_repo": repo, "source_revision": revision,
                                           "source_file": filename, "source_row": row_index,
                                           "group": group, "split": split})
        output[name] = {}
        for split, reservoir in reservoirs.items():
            if len(reservoir.items) != capacities[split]:
                raise ValueError(f"Not enough {name}/{split}: {len(reservoir.items)}")
            random.Random(SEED).shuffle(reservoir.items)
            path = DATA / f"{name}-{split}.jsonl"
            with path.open("w", encoding="utf-8", newline="\n") as f:
                for item in reservoir.items:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
            output[name][split] = {"path": str(path.relative_to(ROOT)), "documents": len(reservoir.items),
                                   "eligible_documents": reservoir.seen, "sha256": sha256_file(path)}
        print(f"partitioned {name}: {output[name]}", flush=True)
    manifest = {"kind": "slm.exp0047.story_partitions", "seed": SEED, "code": code_revision(),
                "builder_sha256": sha256_file(__file__), "source_files": files, "partitions": output,
                "exclusions": dict(counts), "official_validation_sha256": sha256_file(validation),
                "old_dev_provenance_sha256": sha256_file(old / "provenance.jsonl"),
                "old_train_sha256": sha256_file(old / "train.jsonl"),
                "holdout_policy": "source train only; completion-group split for SimpleStories, normalized-content hash for TinyStories; old train excluded from new holdouts; old dev and official validation excluded everywhere",
                "limitations": ["exact/NFC/case/whitespace exclusion, not semantic decontamination",
                                "tiny1m training exposure to source-train holdouts unknown; development comparisons only",
                                "SimpleStories candidate and TinyStories candidate training sources; local research, no automatic publication",
                                "single seed pilots; source completion families need not be statistically independent"]}
    write_json(DATA / "manifest.json", manifest)
    return manifest


def encode_mixtures(manifest: dict, pool_tokens: int) -> None:
    import numpy as np

    pools = {}
    for name in ("simple", "tiny"):
        out = TOKENS / f"pool-{name}"
        encode_fixed_splits(TOKENIZER, {"train": [DATA / f"{name}-train.jsonl"],
                                       "dev": [DATA / f"{name}-dev.jsonl"]}, out)
        pools[name] = out
    val_bytes = b"".join((pools[name] / "val_00000.bin").read_bytes() for name in pools)
    for percent in (0, 30, 60):
        out = TOKENS / f"tiny{percent:02d}"
        out.mkdir(parents=True, exist_ok=True)
        shards = []
        for name, fraction in (("simple", (100-percent)/100), ("tiny", percent/100)):
            n = round(pool_tokens * fraction)
            if not n:
                continue
            source = np.memmap(pools[name] / "train_00000.bin", dtype=np.uint16, mode="r")
            if len(source) < n:
                raise ValueError(f"{name} has {len(source)} tokens; need {n}")
            filename = f"train_{name}.bin"
            source[:n].tofile(out / filename)
            shards.append({"file": filename, "tokens": n, "sha256": sha256_file(out / filename), "source": name})
        (out / "val_00000.bin").write_bytes(val_bytes)
        token_manifest = {"kind": "slm.exp0047.fixed_mixture", "dtype": "uint16", "vocab_size": 1024,
                          "eos_id": 0, "tokenizer_sha256": sha256_file(TOKENIZER),
                          "target_tiny_token_fraction": percent/100,
                          "expected_tiny_window_fraction": sum(max(s["tokens"]-257, 0) for s in shards if s["source"] == "tiny") / sum(s["tokens"]-257 for s in shards),
                          "pool_tokens": pool_tokens, "inputs_manifest_sha256": sha256_file(DATA / "manifest.json"),
                          "splits": {"train": {"tokens": pool_tokens, "shards": shards},
                                     "val": {"tokens": len(val_bytes)//2, "shards": [{"file": "val_00000.bin", "tokens": len(val_bytes)//2, "sha256": sha256_file(out / "val_00000.bin")}]}}}
        write_json(out / "manifest.json", token_manifest)
        write_json(out / "provenance.json", {"partitions_manifest": manifest,
                    "pool_manifests": {name: {"path": str(p / "manifest.json"), "sha256": sha256_file(p / "manifest.json")} for name, p in pools.items()},
                    "mixture": "prefix of deterministically shuffled train-doc token streams; last document can be truncated; random windows sampled proportional to eligible token positions",
                    "test": "not encoded or present in any trainer manifest"})
        print(f"encoded tiny{percent:02d}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-docs", type=int, default=150000)
    parser.add_argument("--dev-docs", type=int, default=512)
    parser.add_argument("--test-docs", type=int, default=1000)
    parser.add_argument("--pool-tokens", type=int, default=32000000)
    args = parser.parse_args()
    torch.set_num_threads(2)
    if (DATA / "complete.json").exists():
        record = json.loads((DATA / "complete.json").read_text())
        if record["args"] != vars(args) or record["builder_sha256"] != sha256_file(__file__):
            raise ValueError("Completed corpus has different settings/code; use a new experiment")
        print("EXP-0047 corpus already complete", flush=True)
        return
    manifest = build_partitions(args.train_docs, args.dev_docs, args.test_docs)
    encode_mixtures(manifest, args.pool_tokens)
    write_json(DATA / "complete.json", {"args": vars(args), "builder_sha256": sha256_file(__file__),
                                        "manifest_sha256": sha256_file(DATA / "manifest.json")})


if __name__ == "__main__":
    main()
