# EXP-0009 grown 25M: resume diagnostic

**Date:** 22 October 2026\
**Run:** `runs/fsc-scaling/25m-gated-grown-s1234/`\
**Durable resume point:** step 3,000\
**Archived failed-attempt record:** `runs/fsc-scaling/25m-gated-grown-s1234/diagnostics/runtime-mismatch-2026-10-22-torch240/attempt-report.json`

## Incident and correction

The run was first resumed from checkpoint 3,000 in a different local environment: Python 3.12.3 and PyTorch 2.4.0+cu118. Validation loss at step 3,500 reached 4.82375, sharply above the original trace. Training was stopped before a newer checkpoint was written. The attempt's logs and manifest were archived; its post-step-3,000 rows were removed from the active run logs. The retained checkpoint was unchanged.

The original checkpoint provenance records Python 3.11.9 and PyTorch 2.10.0+cu130. Under that runtime, validation on checkpoint 3,000 recomputed to 2.469376 (recorded as 2.46938). The resumed step-3,050 training loss was 2.38977, matching the historical trace. Validation at step 3,500 was 2.45157, also matching the historical record. These checks localize the divergence to the runtime mismatch rather than a damaged durable checkpoint.

The active run was restarted from step 3,000 with Python 3.11.9 / PyTorch 2.10.0+cu130. Its current metrics and evaluations are appended to the retained historical logs. The resume is not bitwise deterministic: the training config has `deterministic: false`, and the run's original code worktree was dirty. The matching losses are a useful consistency check, not proof that every future operation will reproduce bit for bit.

## Traceability

The 3,000-step checkpoint SHA-256 is `40fe52e11b7d7dfe466db56b57d07f98af0b45bc3e499bae0d57e07d10c797a5`. The archived mismatch diagnostic SHA-256 is `a71c62d80a81dd7eda01bf43ea39269d25c5bb0c8226c7a6e39692a5ab777a29`. The run manifest for the corrected resume records the runtime and the source checkpoint. Keep the archived attempt visible in future model cards and distinguish it from the completed training trajectory.
