# Research axis (2026-10-31): world models and predictive representations

**Status: exploratory proposal, not preregistered.** Inspiration: Y. LeCun, "World Models: Enabling the
Next AI Revolution" (MAIN 2026 keynote, as reported on LinkedIn) and the hierarchical JEPA (H-JEPA)
figure of LeCun (2022), *A Path Towards Autonomous Machine Intelligence* (OpenReview). We do not
assume this axis improves performance; it is explored, benchmarked and documented either way.

## Question

Can a ~1B-parameter model learn representations of an environment that let it predict, plan and act
more efficiently than through language-based reasoning alone? Sub-question: is the world model part
of the parameter budget (an auxiliary objective/head inside the same network) or an external module?

## Relevant prior work (to read and cite before any claim)

- JEPA family: I-JEPA (Assran et al., 2023), V-JEPA (Bardes et al., 2024), V-JEPA 2 with the
  action-conditioned V-JEPA 2-AC used for planning (Assran et al., 2025). Predict target-encoder
  embeddings (EMA target, stop-gradient) instead of raw observations.
- LLM-JEPA (Huang, LeCun & Balestriero, 2025): a JEPA loss in the embedding space of a language model
  between two views of the same content (e.g. text and code), added to next-token prediction.
- Emergent world models in sequence models: Othello-GPT (Li et al., 2023; linear probes by Nanda et
  al., 2023) and chess-playing LMs (Karvonen, 2024): a model trained only on move sequences builds a
  probe-decodable board state. Cheap, verifiable, and suited to this project's hook-based interpretability.
- Model-based RL and planning in latent space: MuZero (Schrittwieser et al., 2020), Dreamer (Hafner et
  al., 2020–2023). Latent reasoning for LMs: Coconut (Hao et al., 2024).
- Local, verifiable environments: MiniGrid/BabyAI (Chevalier-Boisvert et al.), TextWorld (Côté et
  al., 2018), ALFWorld (Shridhar et al., 2021), Sokoban, Othello.

## Ideas taken from the discussion around the keynote

- **Bytes are not information** (comment on the "4-year-old vs LLM bytes" slide): raw bandwidth is not
  semantic content, and blind children reach typical cognition. Consequence for us: compare
  configurations by information-normalized measures (bits per byte, sample efficiency in environment
  transitions, task success per compute) rather than by data volume.
- **Causality** (P. Bormann): evaluate interventions, not only observational prediction: does the
  model predict the effect of an action it never saw paired with that state? Counterfactual and
  intervention splits are built into the environment generators.
- **World model first, then a self model** (Tony M.): add a calibration/metacognition probe: does the
  model know when its state prediction is uncertain (expected calibration error on transitions)?
- **Revisit symbolic AI** (P. Davidson): the "agentic" configuration can call an exact symbolic
  simulator as a tool; it is a natural upper bound and a neuro-symbolic baseline.
- **Learning needs a verifier in the loop** (Jordan L.): a child's actions are graded by the world for
  free, whereas a text corpus reports experiments without their outcomes. Consequence: besides
  offline trajectories, add a closed-loop variant in which the model acts in the simulator, receives
  the true next state and success signal, and trains on its own graded experience (Dyna-style), with
  the number of environment interactions counted as the data budget.
- **Far transfer as evidence of understanding** (E. B. Coleman): held-out environment variants
  (larger grids, unseen object combinations, a rule change, a different game) measure transfer, not
  only in-distribution accuracy.
- **System 1 vs System 2** (A. García Fente): a JEPA predictor alone is a fast "System 1" model;
  planning by search over predicted states is the "System 2" part. Configuration C is evaluated with
  and without search, so the two contributions are separated.
- **Frugality** (M. Danelsky, "20 W brain"): report energy or GPU-time per solved task, not accuracy alone.
- A comment states that a recent LeCun-group world model "cannot get causality"; this is not
  verified and is not used until a primary source is read.
- Claims without a testable definition (e.g. "unconscious archetypes", "a formula for AGI") are not adopted.

## Proposed experiment (three configurations, same backbone and budget)

| Configuration | Training signal | At test time |
|---|---|---|
| A. Classic | Next-token prediction on serialized trajectories (state, action, next state, text) | Generates states/plans as text |
| B. Agentic | As A, plus tool-call traces to an exact simulator | Plans by querying the simulator |
| C. World model | As A, plus a JEPA loss: predict the target-encoder embedding of s(t+1) from (s(t), a(t)) inside the same network; optional external latent predictor as a separate arm | Predicts directly ("System 1") or plans by search over predicted latent states ("System 2") |
| D. Closed loop (variant of A or C) | Acts in the simulator, trains on its own verifier-graded transitions | Same as its parent configuration; budget counted in environment interactions |

Environments, in order: Othello (state tracking, probes), a deterministic text gridworld / MiniGrid as
text (navigation, keys/doors), Sokoban (irreversible actions, planning depth). All generated locally
with seeds, with an exact verifier and held-out intervention/counterfactual splits.

Metrics: next-state accuracy (seen and intervened actions), far-transfer accuracy on held-out
environment variants, plan success rate and optimality gap,
episodes or tokens needed to reach a success threshold, GPU time per solved task, calibration of state predictions,
and linear-probe decodability of the true state from hidden activations (hooks).

Scale: pilots at 25–124M (hours on the 8 GB GPU), then 362M; the 1B arm only if the pilots show an
effect, because a 1B run costs about three GPU days. Each comparison uses the same data order, seed
and token/compute budget, with configuration C's extra encoder/predictor parameters reported (inside
vs outside the parameter budget as separate arms).
