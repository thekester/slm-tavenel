# Additional evaluation ideas from recent papers

**Reviewed:** 2026-10-25
**Scope:** follow-up evaluations for SLM-Tavenel. These papers do not change EXP-0028's training data, checkpoints, or already-fixed benchmark protocol. Every benchmark remains test-only.

## 1. Local-language access to cultural knowledge

Yang Zhang et al., [The Masked Advantage: Uncovering Local-Language Access to Cultural Knowledge in LLMs](https://arxiv.org/abs/2606.07422), proposes matched culture-agnostic and culture-specific questions in local languages and English. It estimates a 1PL/Rasch ability model to distinguish general language proficiency from culturally grounded knowledge access.

The paper draws culture-agnostic items from Global-MMLU and culture-specific items from CMMLU, ArabicMMLU, GreekMMLU, MILU, KMMLU, and INCLUDE. For French it reports 2,058 culture-agnostic items and 139 culture-specific items from INCLUDE, with English counterparts translated and human-checked. The study filters out models below 0.35 accuracy on each subset; its reported population includes models down to 0.6B parameters. SLM-Tavenel 50M/100M is below that scale, so the paper's model population is a methodological reference, not a directly matched baseline.

**Candidate follow-up:** add a separate, explicitly exploratory French audit after the locked EXP-0028 tests. A first, narrower option is the French culture-agnostic split from [CohereLabs/Global-MMLU](https://huggingface.co/datasets/CohereLabs/Global-MMLU), whose card declares Apache-2.0. Before scoring, pin a revision, inspect the actual French split and its source terms, and run exact plus approximate contamination checks against every training corpus. If the culture-specific French items and validated English counterparts used by the paper cannot be located with suitable terms, do not claim to reproduce its KnowledgeGap. With only 139 French culture-specific items, show item-level uncertainty and avoid strong rankings. The paper's 1PL-IRT result should only be reproduced with the matched items, languages, scoring, and fitting assumptions documented.

## 2. Iterative generation chains

Mingmeng Geng et al., [Markovian Generation Chains in Large Language Models](https://arxiv.org/abs/2603.11228), studies repeated inference-time rewriting and round-trip translation. It samples 150 seed sentences from each of BookSum, ScriptBase-alpha, and BBC News2024, then follows 50 iterations. Measures include distinct-sentence count, first recurrence, METEOR/ROUGE-1/BLEU and TF-IDF drift, under greedy decoding and sampling (temperature 0.7, top-p 0.9). The authors explicitly distinguish these inference-time dynamics from training-time model collapse.

**Candidate follow-up:** treat this as a behavioral/observability diagnostic, not a capability benchmark or a training recipe. Current SLM-Tavenel checkpoints are base language models, while the paper's models are instruction-tuned 7B/8B systems. A prompt-following reproduction would therefore not be directly comparable. If run later, label it as an adapted diagnostic, freeze prompts and decoding settings, report recurrence/diversity and drift separately, and use 50 independent seeds or repeated runs as feasible.

The paper's seed corpora are not automatically cleared for redistribution or model training. BookSum's Hugging Face card labels its packaged dataset BSD-3-Clause but includes a separate legal note assigning responsibility for source rights; ScriptBase-alpha and News2024 also need source-level review. Do not download or ingest them into training on the basis of this paper alone. Prefer locally sourced, rights-reviewed held-out seed texts if the goal is only to measure generation dynamics.

## Queue and separation

1. Finish EXP-0028's 100M continuation and run the already-registered corpus and six benchmark evaluations unchanged.
2. Review whether the exact Global-MMLU French test subset and the paper's French INCLUDE pairs are available, pinned, and compatible with our scoring setup; estimate coverage and contamination before adding an optional study.
3. Decide later whether to implement the iterative-generation diagnostic using seed texts with documented provenance.

Neither paper authorizes adding evaluation items to pretraining. The existing EXP-0028 results remain the primary planned comparison; these are proposed follow-up studies only.
