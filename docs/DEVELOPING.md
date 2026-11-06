# Developing SLM-Tavenel

This guide runs a small synthetic-data workflow from the repository root. It does not download or process the corpora used for the published models.

## Requirements

- Python 3.10 or newer
- PyTorch 2.3 or newer
- A GPU is optional for the smoke workflow

Install the development, Transformers, and energy-monitoring extras:

```bash
pip install -e ".[dev,hf,energy]"
```

## Synthetic smoke workflow

The following commands create toy prose and arithmetic data, process it, train a tokenizer and a tiny model, and export a checkpoint. The generated artifacts go under `data/` and `runs/`, which are ignored by Git.

```bash
python -m slm sources audit -v
python -m slm data synth --kind prose --n 3000 --out data/raw/smoke/prose.jsonl
python -m slm data synth --kind arithmetic --n 200 --seed 1 --out data/raw/smoke/arith-eval.jsonl
python -m slm data run configs/data/pipeline-smoke.yaml
python -m slm tokenizer train --inputs data/processed/smoke/documents.jsonl --out data/tokenizer/smoke --vocab-size 4096
python -m slm data encode --tokenizer data/tokenizer/smoke --inputs data/processed/smoke/documents.jsonl --out data/tokens/smoke --val-fraction 0.05
python -m slm train configs/train/smoke.yaml
python -m slm eval bpb --checkpoint runs/smoke/checkpoints/step_00000300.pt --tokenizer data/tokenizer/smoke --data data/processed/smoke/documents.jsonl --limit 50
python -m slm eval cost --checkpoint runs/smoke/checkpoints/step_00000300.pt --tokenizer data/tokenizer/smoke
python -m slm export --checkpoint runs/smoke/checkpoints/step_00000300.pt --tokenizer data/tokenizer/smoke --out exports/smoke-hf
```

For other commands, run `python -m slm --help`. Training configurations accept overrides, for example:

```bash
python -m slm train configs/train/smoke.yaml --set max_steps=100 seed=7
```

The example configuration is intentionally tiny and synthetic. It is not the training recipe for any published model. The exact source revisions, filters, tokenization, training settings, checkpoints, and evaluation protocols are documented separately in each model card and experiment report. The source corpus files are not bundled with the repository.

## Code map

| Path | Purpose |
|---|---|
| `src/slm/model.py` | Transformer, hook points, KV cache, and generation |
| `src/slm/observe.py`, `src/slm/interp.py` | Activation capture, interventions, and causal analyses |
| `src/slm/data/` | Source registry, filters, deduplication, privacy checks, decontamination, and data pipeline |
| `src/slm/tokenizer.py` | Byte-level BPE training and provenance manifest |
| `src/slm/train.py` | Resumable training loop and instrumentation |
| `src/slm/evaluation.py` | Bits per byte, verified tasks, cost, and energy |
| `src/slm/export.py` | Hugging Face export and provenance |
| `configs/` | Model, training, and data-pipeline configurations |

## Reproducibility notes

The synthetic workflow is useful for checking local setup and CLI wiring; it does not reproduce published results. Reproducing a reported run requires its pinned code revision, data manifest, tokenizer, configuration, and recorded environment. Benchmark prompts and evaluation records are governed by their own redistribution terms and are not all bundled here.
