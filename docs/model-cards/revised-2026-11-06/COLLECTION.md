# Hugging Face collection draft: "SLM-Tavenel: small English/French research models"

Not created. To create it later (owner `athenista`), use the Hub UI or `HfApi().create_collection(...)`
then `add_collection_item(...)` with the notes below (each note is limited to 500 characters on the Hub).

**Description (≤ 150 characters):** Experimental English/French base models trained from scratch on one
8 GB GPU, with full data provenance, observability logs and same-harness benchmarks.

| Order | Item | Note to attach |
|---:|---|---|
| 1 | `athenista/slm-tavenel-124m` | 124M, trained from scratch on FineWeb-Edu + FineWeb2-HQ (983M tokens, Muon). Same-harness comparison with GPT-2 Small: better ARC-Easy, worse HellaSwag, preregistered overall-win criterion not met. |
| 2 | `athenista/slm-tavenel-25m-hplt-10pct` | 25M continued for 1,500 steps on a 90/10 FSC/HPLT mixture (EXP-0032): lower held-out HPLT BPB than a matched FSC-only control, no benchmark change. |
| 3 | `athenista/slm-tavenel-25m` | 25M grown from the trained 10M and trained 491.5M more tokens on French-Science-Commons: lower BPB than 10M, no resolved benchmark gain. |
| 4 | `athenista/slm-tavenel-10m` | 10M trained from scratch on French-Science-Commons (491.5M tokens); parent of the 25M. |
| 5 | `athenista/slm-tavenel-1m` | 1.18M on a CC0 science subset, with a preregistered ten-seed gated-vs-ungated study (gate lowers BPB by 0.0275, 95% CI [0.0168, 0.0395]); all 20 runs published. |

Lineage: 1M (independent) · 10M → grown 25M → 25M HPLT 10% · 124M (independent, different tokenizer).
Common to all: custom architecture (`trust_remote_code=True`), no KV cache, no quantized weights, no
safety evaluation, MIT for weights and code, data sources under their own terms.
