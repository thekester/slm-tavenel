# EXP-0001: first 10M model on Wikipedia EN+FR

*Status:* concluded
*Dates:* 2026-10-18
*Kind:* exploratory learning run (not pre-registered, no decision rule). The corpus comes from
`candidate` sources processed with `--allow-unapproved`; nothing from this run may be released as
project weights.

## Question

Does the full chain (fetch, pipeline, tokenizer, shards, training, evaluation, export) work on real
text at a scale that fits one 8 GB laptop GPU, and what do the observability logs show?

## Design

| | Value |
|---|---|
| Model | `configs/model/slm-10m.yaml`: 9,968,128 parameters, 10 layers, width 256, 4/2 heads, vocab 8,192 |
| Training | `configs/train/slm-10m-wiki.yaml`: 15,000 steps x 32,768 tokens = 492M tokens (~49 tokens/param, 0.58 epoch), WSD, peak LR 3e-3 |
| Data | `configs/data/pipeline-wiki-v0.yaml`: 3 EN + 2 FR shards of the 2023-11-01 Wikipedia dump, revision `b04c8d1c` |
| Tokenizer | byte-level BPE, 8,192 tokens, trained on 120k sampled documents (seed 0) |
| Hardware | RTX 4060 Laptop GPU, 8 GB |

## Data processing

770,589 articles in, 605,832 kept (79%; 2.74B characters). Removals: repetition 71,488, Gopher quality
60,844, too short 30,362, near duplicates 2,039, exact duplicates 24. Kept per source: EN 367,745, FR
238,087. Tokens: 850M train, 8.8M validation (1% by document-id hash).

A first pass revealed that the generic phone, IP and IBAN patterns matched mostly dates, ISBNs,
section numbers and identifiers (13,433 false "phone numbers"). The detectors were tightened (explicit
phone layouts, contextual cue for IPv4, IBAN checksum), the observed false positives became regression
tests, and the corpus was rebuilt. Final redactions: 145 emails, 123 phones, 42 IPs, 1 IBAN.

## Results

| Metric | Value |
|---|---|
| Validation loss (step 15,000) | 2.750 (2.945 at the start of decay, step 12,000) |
| Bits per byte, EN validation (500 docs) | 1.316 |
| Bits per byte, FR validation (500 docs) | 1.156 |
| Training throughput | 62k-74k tokens/s, 3.5 GB peak memory, 121.5 min |
| Generation | 58 new tokens/s (batch 1), 0.11 s to first token, 172.9 J for 384 tokens |
| Export | Qwen3 format, 20 MB, generates in `transformers` |

The FR-EN gap in bits per byte is not a language comparison: the validation articles differ in
content and length, and French has more training tokens.

Samples are fluent in Wikipedia style and factually unreliable, as expected at 10M parameters
(e.g. "Paris is the capital of the city of Saint-Germain-en-Laye").

## Observations from the logs

- **Attention sinks emerged during training.** Mean attention mass on the first token was 3-4% in
  every layer at initialisation; at the end it reaches 29% in layer 4, 26% in layer 6 and 32% in
  layer 7, while layers 0-2 stay near 2.5%.
- **Massive activations accompany them.** The residual stream's maximum absolute value reaches
  about 700 while its RMS in the last layer is about 21-30.
- These are correlations in logs, not explanations. The next step is an intervention: ablate or
  patch the first-token position and the high-magnitude residual dimensions, and measure the effect
  on loss. Ablation A1 (head-wise output gate) tests whether a gate removes the sinks.

## Follow-ups

1. Interventions on the sink heads and the massive-activation dimensions (causal test of the above).
2. Ablation A1 (gated attention) at 10M with the same data and seed, as a cheap first comparison
   before 135M.
3. Generation is launch-bound at this size (58 tokens/s); batch prompts or use CUDA graphs for cost
   measurements.
4. Review the Wikipedia sources (license terms, privacy) so that a reviewed corpus can replace this
   exploratory one.
