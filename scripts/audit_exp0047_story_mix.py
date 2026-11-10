"""Verify built EXP-0047 split isolation and trainer inputs before any training."""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from build_exp0047_story_mix import DATA, TOKENS, TOKENIZER, normalized_hash, assigned_split
from slm import tokenizer
from slm.runlog import sha256_file, write_json


def audit() -> dict:
    import pyarrow.parquet as pq

    manifest = json.loads((DATA / "manifest.json").read_text())
    completed = json.loads((DATA / "complete.json").read_text())
    if completed["manifest_sha256"] != sha256_file(DATA / "manifest.json"):
        raise ValueError("Corpus manifest changed")
    if completed["builder_sha256"] != sha256_file(ROOT / "scripts/build_exp0047_story_mix.py"):
        raise ValueError("Corpus builder changed")
    tok = tokenizer.load(TOKENIZER)
    if tok.get_vocab_size() != 1024 or tok.token_to_id(tokenizer.EOS) != 0:
        raise ValueError("Tokenizer shape/EOS mismatch")
    old = ROOT / "data/processed/exp0045-simple-stories-v2"
    blocked = {normalized_hash(json.loads(line)["text"]) for line in (old / "dev.jsonl").read_text(encoding="utf-8").splitlines()}
    validation = ROOT / "data/raw/tinystories-f54c09fd/data/validation-00000-of-00001-869c898b519ad725.parquet"
    blocked.update(normalized_hash(t) for t in pq.read_table(validation, columns=["text"]).column(0).to_pylist())
    old_train = {normalized_hash(json.loads(line)["text"]) for line in (old / "train.jsonl").read_text(encoding="utf-8").splitlines()}
    old_dev_groups = {r["completion_group"] for line in (old / "provenance.jsonl").read_text(encoding="utf-8").splitlines()
                      if (r := json.loads(line))["split"] == "dev"}
    all_hashes, groups, counts = set(), {}, {}
    for source in ("simple", "tiny"):
        for split in ("train", "dev", "test"):
            path = DATA / f"{source}-{split}.jsonl"
            expected = manifest["partitions"][source][split]
            if sha256_file(path) != expected["sha256"]:
                raise ValueError(f"Partition changed: {path}")
            count = 0
            with path.open(encoding="utf-8") as f:
                for line in f:
                    record = json.loads(line)
                    digest, group = normalized_hash(record["text"]), record["group"]
                    if digest != record["normalized_sha256"] or digest in blocked or digest in all_hashes:
                        raise ValueError("Duplicate, blocked text, or text hash mismatch")
                    if assigned_split(group) != split or (split != "train" and digest in old_train):
                        raise ValueError("Holdout leakage or split mismatch")
                    if source == "simple" and group in old_dev_groups:
                        raise ValueError("Old dev completion leaked")
                    if groups.setdefault((source, group), split) != split:
                        raise ValueError("Completion/content group spans partitions")
                    all_hashes.add(digest)
                    count += 1
            if count != expected["documents"]:
                raise ValueError("Document count mismatch")
            counts[f"{source}_{split}"] = count
    for percent in (0, 30, 60):
        folder = TOKENS / f"tiny{percent:02d}"
        token_manifest = json.loads((folder / "manifest.json").read_text())
        if token_manifest["tokenizer_sha256"] != sha256_file(TOKENIZER):
            raise ValueError("Tokenizer lineage mismatch")
        if set(token_manifest["splits"]) != {"train", "val"}:
            raise ValueError("Reserved test entered trainer manifest")
        for part in token_manifest["splits"].values():
            for shard in part["shards"]:
                if sha256_file(folder / shard["file"]) != shard["sha256"]:
                    raise ValueError("Token shard changed")
        train = token_manifest["splits"]["train"]
        if sum(s["tokens"] for s in train["shards"]) != 32000000:
            raise ValueError("Training pool size mismatch")
        if sum(s["tokens"] for s in train["shards"] if s["source"] == "tiny") != 32000000 * percent // 100:
            raise ValueError("Mixture ratio mismatch")
    result = {"passed": True, "counts": counts, "normalized_duplicates": 0, "blocked_text_matches": 0,
              "group_split_leakage": 0, "old_train_matches_in_new_holdouts": 0,
              "manifest_sha256": sha256_file(DATA / "manifest.json"), "audit_script_sha256": sha256_file(__file__),
              "limits": "Hash/group audit only; no semantic privacy or rights clearance. Test read only for integrity audit, never scored or selected on."}
    write_json(DATA / "audit.json", result)
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    audit()
