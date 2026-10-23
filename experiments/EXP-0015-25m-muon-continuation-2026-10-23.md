# EXP-0015: Muon continuation at 25M

**Status (2026-10-23 15:27 Paris):** the run stopped with `torch.AcceleratorError: CUDA error: an illegal memory access was encountered` after logging step 300/1,500. No checkpoint was written (the first checkpoint was scheduled for step 500), so this is a failed/incomplete pilot with no outcome comparison. The GPU was heavily contended when checked after the failure; whether contention caused the error is unknown. Do not interpret the partial training losses as a result. Any retry needs a fresh run directory, a stability check and an isolated GPU window; no weights were published or substituted.

## Question

Does the repository's Muon variant improve held-out FSC text prediction or the existing general multiple-choice scores when used for a short continuation of the grown 25M checkpoint?

Muon is motivated by the project's positive 10M fixed-document BPB findings (EXP-0005) and research on orthogonalized momentum updates. This is a transfer test, not an assumption that the 10M result will transfer to 25M. The 25M checkpoint already received AdamW training, so optimizer state is reset at this continuation boundary for both the archived AdamW controls and this Muon treatment.

## Protocol

- Initialize from the exact EXP-0009 grown-25M step-15,000 checkpoint; record and compare its SHA-256 with EXP-0014's initialization hash.
- Train 1,500 steps on the same FSC scaling shards, batch size, sequence length, token budget (49.152M positions), seed, schedule, LR range, weight decay and precision as the EXP-0014 FSC-only continuation. Change only the optimizer to the repository's Muon plus AdamW split.
- Primary outcome: held-out FSC English and French BPB, scored on the same fixed validation documents as the 25M baseline and EXP-0014.
- Secondary exploratory outcomes: six pre-used same-harness multiple-choice tasks. They are repeatedly inspected and are not an untouched confirmatory test. Report paired item intervals, conditional on the evaluated items and checkpoints; they do not capture seed uncertainty.
- Retain checkpoint, full training/observability logs, evaluator inputs and hashes. No checkpoint is published. Existing source-rights and privacy issues in the FSC corpus remain unresolved.

## Decision rule

Treat a lower held-out BPB point estimate in both languages as a promising signal, not confirmation. Treat multiple-choice movement as exploratory and do not select a release model based on these repeatedly used benchmark files. Any performance claim needs paired training seeds, tuned optimizer schedules decided on validation data, and a separate test set.

## References

- Jordan (2024), [Muon optimizer notes](https://kellerjordan.github.io/posts/muon/).
- Liu et al. (2025), [Muon is Scalable for LLM Training](https://arxiv.org/abs/2502.16982).
- Project's 10M transfer motivation and limitations: [EXP-0005](EXP-0005-muon-10m.md).
- Prior 25M continuation results and corpus limitations: [EXP-0014](EXP-0014-25m-continuation-mixture-2026-10-23.md).
