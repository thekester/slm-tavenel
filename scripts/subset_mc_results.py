"""Create a traceable subset of an item-level multiple-choice evaluation output."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--names", nargs="+", required=True)
    parser.add_argument("--split", action="append", default=[], metavar="NAME=SPLIT",
                        help="optional benchmark split metadata, e.g. arc_challenge=test")
    parser.add_argument("--model-params", type=int,
                        help="explicit parameter count when the source evaluation omitted it")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(args.input.read_text(encoding="utf-8"))
    by_name = {Path(row["path"]).stem: row for row in source.get("benchmarks", [])}
    missing = set(args.names) - set(by_name)
    if missing:
        raise ValueError(f"Requested benchmarks absent from input: {sorted(missing)}")
    results = {name: source["results"][name] for name in args.names}
    if set(results) != set(args.names):
        raise ValueError("Benchmark metadata and result names differ")
    split_by_name = {}
    for value in args.split:
        if "=" not in value:
            parser.error("--split must have NAME=SPLIT form")
        name, split = value.split("=", 1)
        split_by_name[name] = split
    benchmark_rows = []
    for name in args.names:
        item = dict(by_name[name])
        if name in split_by_name:
            item["split"] = split_by_name[name]
        benchmark_rows.append(item)
    output = {"source_evaluation": str(args.input), "source_sha256": sha256(args.input),
              "benchmark_names": args.names,
              "benchmarks": benchmark_rows, "results": results}
    params = args.model_params if args.model_params is not None else source.get("model_params")
    if params is not None:
        output["model_params"] = params
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "source_sha256": output["source_sha256"],
                      "benchmarks": {name: results[name].get("items") for name in args.names}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
