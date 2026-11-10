#!/usr/bin/env python3
"""Tokenized COCO 2017 captions for EXP-0046: captions of one image are good views of the same scene.

Source: COCO 2017 annotations (captions_train2017.json, captions_val2017.json; cocodataset.org; annotations
CC BY 4.0; images remain under their Flickr licences and are used locally for evaluation only, never
redistributed). Pinned SmolLM2 tokenizer, truncation to 48 tokens, right-padding with the end-of-text id.
train.npz: ids (captions x 48), lengths, image (index of the caption's image). Only images with at least two
captions are kept. val.json: for each val2017 image, its id, file name and first caption (lowest annotation id).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from slm.runlog import sha256_file  # noqa: E402

RAW = ROOT / "data/raw/coco2017/annotations"
TOKENIZER = ROOT / "data/tokenizers/smollm2-f8027fd0/tokenizer.json"
MAX_TOKENS = 48


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "data/parallel/coco2017-captions-v1")
    args = parser.parse_args()
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(TOKENIZER))
    eos = tokenizer.token_to_id("<|endoftext|>")
    args.out.mkdir(parents=True, exist_ok=True)
    source = RAW / "captions_train2017.json"
    annotations = json.loads(source.read_text(encoding="utf-8"))["annotations"]
    by_image: dict[int, list[str]] = defaultdict(list)
    for item in sorted(annotations, key=lambda a: a["id"]):
        by_image[item["image_id"]].append(item["caption"].strip())
    images = sorted(i for i, c in by_image.items() if len(c) >= 2)
    texts, owner = [], []
    for index, image in enumerate(images):
        for caption in by_image[image]:
            texts.append(caption)
            owner.append(index)
    encoded = [e.ids[:MAX_TOKENS] for e in tokenizer.encode_batch(texts)]
    ids = np.full((len(encoded), MAX_TOKENS), eos, dtype=np.int32)
    for i, row in enumerate(encoded):
        ids[i, : len(row)] = row
    lengths = np.array([max(1, len(r)) for r in encoded], dtype=np.int32)
    np.savez(args.out / "train.npz", ids=ids, lengths=lengths, image=np.array(owner, dtype=np.int32))

    val_source = RAW / "captions_val2017.json"
    val = json.loads(val_source.read_text(encoding="utf-8"))
    names = {im["id"]: im["file_name"] for im in val["images"]}
    first: dict[int, str] = {}
    for item in sorted(val["annotations"], key=lambda a: a["id"]):
        first.setdefault(item["image_id"], item["caption"].strip())
    rows = [{"image_id": i, "file_name": names[i], "caption": first[i]} for i in sorted(first)]
    (args.out / "val.json").write_text(json.dumps(rows, ensure_ascii=False) + "\n", encoding="utf-8")

    manifest = {
        "schema": "slm-tavenel.coco_caption_pairs.v1",
        "source": {"site": "https://cocodataset.org", "captions_train_sha256": sha256_file(source),
                   "captions_val_sha256": sha256_file(val_source),
                   "license": "annotations CC BY 4.0; images under their Flickr licences (evaluation only, not redistributed)"},
        "tokenizer_sha256": sha256_file(TOKENIZER), "pad_id": int(eos), "max_tokens": MAX_TOKENS,
        "train_images": len(images), "train_captions": len(texts), "val_images": len(rows),
        "mean_caption_tokens": round(float(lengths.mean()), 2), "train_npz_sha256": sha256_file(args.out / "train.npz"),
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
