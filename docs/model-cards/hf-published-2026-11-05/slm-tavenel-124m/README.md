---

language:

- en

- fr

license: mit

library_name: transformers

pipeline_tag: text-generation

datasets:

- HuggingFaceFW/fineweb-edu

- epfml/FineWeb2-HQ

tags:

- slm-tavenel

- research-artifact

- base-model

- experimental

- 124m

---



# SLM-Tavenel 124M



**Experimental English/French base model, trained from scratch.** This repository publishes the checkpoint evaluated in EXP-0036. It is not instruction tuned and is not a chat assistant. The model is compared with GPT-2 Small using the same six-task evaluator and pinned benchmark items. Results are mixed; the preregistered criterion for an overall win was not met.



[![Architecture of SLM-Tavenel 124M](architecture.png)](architecture.svg)



## Model facts



| Property | Value |
|---|---|
| Parameters | 124,038,144 total; 95,726,592 excluding token embeddings |
| Layers / width | 27 / 576 |
| Attention | 9 query heads, 3 key/value heads, QK normalization, RoPE, head-wise sigmoid attention gate |
| MLP | SwiGLU, width 1,536 |
| Context | 2,048 tokens |
| Tokenizer | SmolLM2 tokenizer, 49,152 tokens, tied input/output embeddings |
| Training | From scratch, Muon, seed 20261031, 15,000 updates, 983,040,000 token positions |
| Precision in this package | float32 safetensors |



## Results against GPT-2 Small



Both checkpoints were scored by the SLM-Tavenel evaluator on the same pinned six benchmark files. Accuracy is raw multiple-choice accuracy. The 95% intervals are paired bootstrap intervals over items (10,000 draws); McNemar tests are also in the experiment record. They measure item uncertainty for these checkpoints, not run-to-run training variance.



[![Paired benchmark accuracy differences](gpt2-paired-accuracy.png)](gpt2-paired-accuracy.svg)



| Task | Items | GPT-2 124M | SLM-Tavenel 124M | Difference, 95% paired interval |
|---|---:|---:|---:|---|
| ARC-Easy | 2,376 | 0.447 | 0.484 | +0.036 [ +0.018, +0.055 ] |
| ARC-Challenge | 1,172 | 0.185 | 0.206 | +0.021 [ 0.000, +0.044 ] |
| HellaSwag | 10,042 | 0.292 | 0.277 | −0.015 [ −0.021, −0.010 ] |
| Belebele EN | 900 | 0.310 | 0.316 | +0.006 [ −0.021, +0.032 ] |
| Belebele FR | 900 | 0.318 | 0.311 | −0.007 [ −0.038, +0.024 ] |
| WinoGrande | 1,267 | 0.517 | 0.519 | +0.002 [ −0.030, +0.036 ] |
| **Six-task mean** | — | **0.345** | **0.352** | **+0.007 [ −0.003, +0.017 ]** |



The preregistered criterion required higher HellaSwag accuracy and a higher six-task mean. It was **not met**: the six-task mean interval includes zero and HellaSwag is lower. ARC-Easy is higher; other task intervals include zero. The length-normalized HellaSwag scores are 0.316 for GPT-2 and 0.289 for SLM-Tavenel. This is a descriptive comparison, not evidence that SLM-Tavenel is generally better than GPT-2.



Held-out bits per byte on this model's corpus-v2 test split is 1.067 overall (English 1.085, French 1.040). GPT-2 was not scored on that split, so these values are not a comparative result.



## Architecture



[![Decoder diagram](architecture.png)](architecture.svg)



The model uses pre-RMSNorm, causal grouped-query attention, QK normalization, rotary positional embeddings, a sigmoid gate per attention head, and a SwiGLU feed-forward sublayer with residual connections. Input and output embeddings are tied. This is a custom SLM-Tavenel architecture; loading the supplied Transformers implementation requires `trust_remote_code=True`. It does not implement a KV cache, so generation is slower than optimized decoder implementations.



## Use



```python

from transformers import AutoModelForCausalLM, AutoTokenizer



model_id = "athenista/slm-tavenel-124m"

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)

inputs = tokenizer("La recherche scientifique.", return_tensors="pt")

outputs = model.generate(**inputs, max_new_tokens=32, do_sample=False)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))

```



Install versions recorded for the package with `pip install -r requirements.txt`.



## Training data and provenance



Training used a locally filtered candidate corpus built from pinned subsets of [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/tree/87f09149ef4734204d70ed1d046ddc9ca3f2b8f9) and [FineWeb2-HQ](https://huggingface.co/datasets/epfml/FineWeb2-HQ/tree/c0c06e94fd3a44ae9e802b2b0fc533817601eb5e). The source dataset cards declare ODC-By 1.0; both datasets also describe Common Crawl terms. FineWeb-Edu selects educationally scored English web pages; FineWeb2-HQ selects quality-filtered French web pages. The corpus manifest records pinned source files and hashes, filters, counts, benchmark decontamination and split hashes. The tokenizer is from [SmolLM2-360M](https://huggingface.co/HuggingFaceTB/SmolLM2-360M/tree/f8027fd0eaeea54caa13c31d31b9fdc459c38b49), whose model card declares Apache-2.0; its exact file hash is recorded in the tokenizer manifest.



This is traceability of dataset declarations and processing. It is **not independent verification of rights in each source page**. The corpus project did not independently clear the original web pages. Regex redaction for email, phone, IPv4 and IBAN is not a complete privacy review. Exact 13-gram benchmark decontamination does not rule out paraphrases, translations or prior exposure. Raw training text is not included in this model repository. Review the included `training_record/corpus-manifest.json` before use.



## Reproducibility and limitations



- The run manifest, configuration, metrics, evaluation log, observability log, tokenizer manifest, corpus manifest, selected training source snapshot and paired benchmark summaries are included under `training_record/`. A one-probe local CPU parity check matched the tokenizer IDs and native/HF logits exactly (maximum absolute error 0.0), and greedy generation appended two tokens; see `inference-parity-report.json`. This smoke check does not test broad compatibility or generation quality.

- The run manifest records Git commit `21aaf6a03d1bab6770dfd37d358e144af62d67c9` and a **dirty worktree at training time**. The exact training-time diff was not archived; the source snapshot is from that commit and this limitation remains.

- The run was interrupted and resumed several times after computer shutdowns. Training uses a single seed. Paired bootstrap intervals do not estimate variation across training seeds.

- GPT-2's training data and token budget are not disclosed precisely in the source used here; the corpora, tokenizers and compute budgets differ. The six-task comparison is a same-harness descriptive comparison, not a controlled training-data or architecture ablation.

- This is a base model with no instruction tuning, preference optimization or safety tuning. Outputs may be inaccurate or unsuitable for consequential use.

- The benchmark figures and held-out BPB do not establish broad language competence, causal effects of architecture, absence of memorization, legal clearance or privacy certification.



## License and citation



The project offers the weights and included SLM-Tavenel inference code under MIT; see `LICENSE`. The tokenizer is separately attributed in `TOKENIZER-NOTICE.md`. The MIT file does not relicense training sources or settle third-party claims over source pages. This repository makes no claim of OSI certification or legal clearance.



For the experiment protocol, paired results and methods, see [EXP-0036 in the SLM-Tavenel repository](https://github.com/thekester/slm-tavenel/blob/92c215f/experiments/EXP-0036-beat-gpt2-at-equal-parameters-2026-10-30.md). Cite this artifact as **SLM-Tavenel 124M, EXP-0036, seed 20261031**; no paper or DOI is assigned.
