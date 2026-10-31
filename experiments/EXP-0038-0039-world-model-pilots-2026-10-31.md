# EXP-0038 / EXP-0039: world-model pilots (preregistered 2026-10-31, before any result)

Axis and literature: `experiments/AXIS-world-models-2026-10-31.md`. Both pilots use the project's
decoder (RoPE, RMSNorm, SwiGLU, GQA, QK-norm, head-wise gate) at about 25–31M parameters, the Muon
optimizer adopted in EXP-0036a, a WSD schedule and one training seed per arm. All data are synthetic,
generated locally with fixed seeds (no rights or privacy questions). Queues:
`scripts/run_world_model_pilot_othello.ps1`, then `scripts/run_world_model_pilot_gridworld.ps1`.

## EXP-0038: Othello (state tracking)

Data (`scripts/build_othello_corpus.py`, `data/worlds/othello-v1`): 1,000,000 training and 10,000
validation games with uniformly random legal moves (the synthetic setting of Li et al., 2023);
10,000 test games from a greedy most-flips policy (a different play style, far transfer). One game per
64-token row. Model `configs/model/othello-31m.yaml` (8 layers x 512, as Othello-GPT).

| Arm | Config | Objective |
|---|---|---|
| A, classic | `exp0038-othello-a-ntp.yaml` | next-move prediction |
| C, world model | `exp0038-othello-c-jepa.yaml` | A + action-conditioned JEPA (token level): predict the EMA model's layer-normalised h(t+1) from h(t) and move t+1, weight 1.0 |

Same seed, data order, schedule and 15,000 updates of 128 games (123M token positions).
Outcomes (`scripts/evaluate_othello_world_model.py`): legal-move rate of the top prediction; linear-probe
board accuracy (empty / mover's / opponent's per square) after every layer, fit on 2,000 validation
games, tested on 2,000 others (in distribution) and 2,000 greedy-policy games (transfer); majority-class
baseline. **Criterion:** C is "better as a world model" if its best-layer probe accuracy exceeds A's on
both test sets by at least one percentage point without a lower legal-move rate. Smaller differences
are reported as no measurable difference.

## EXP-0039: key-door gridworld (prediction, interventions, planning)

World (`src/slm/worlds/gridworld.py`): 6 x 6 room, internal wall with a locked door, key, goal; six
actions, invalid ones leave the state unchanged; exact BFS planner. Data
(`scripts/build_gridworld_corpus.py`, `data/worlds/gridworld-v1`): 1,000,000 training transitions
from a behaviour policy that is optimal with probability 0.8; 20 of 24 wall/door layouts for training,
4 held out. Model `configs/model/gridworld-25m.yaml`; each arm runs 10,000 updates of 64 rows.

| Arm | Config | Training data and objective | Planning at test time |
|---|---|---|---|
| A, classic | `exp0039-grid-a-transitions.yaml` | next-token prediction on (state, action, next state) rows | BFS with the model as simulator ("System 2") |
| C, world model | `exp0039-grid-c-jepa.yaml` | A + state-level JEPA: from the action position predict the pooled EMA representation of the next state, weight 1.0 | as A |
| B, agentic | `exp0039-grid-b-plans.yaml` | writes the optimal plan after the start state | greedy plan; then up to 5 sampled plans checked by the exact simulator (verifier tool) |

Oracle: exact BFS (100% success, optimal length). Outcomes: next-state exact match (all 65 tokens) on
validation, on the intervention split (all six actions from validation states, split into actions
that change the state and actions that do nothing) and on held-out layouts (`evaluate_gridworld_transitions.py`);
planning success and plan length over optimal on 200 in-distribution and 200 held-out-layout start
worlds (`evaluate_gridworld_planning.py`). **Criteria:** C beats A as a world model if its intervention
exact-match and held-out-layout planning success are both higher by at least two percentage points;
B is compared with A/C on planning success only, reporting first attempt and with the verifier tool
separately (the tool gives B information A and C do not get).

Configuration D (closed loop: training on the model's own verifier-graded transitions) is a second
phase, designed after A/B/C, because its data depend on a trained model; it will be preregistered
separately before it runs.

## Limitations fixed in advance

One seed per arm; bootstrap intervals over games/worlds only. The JEPA weight (1.0) and EMA (0.996) are
not tuned; a null result for C does not rule out other settings. B's training rows carry fewer
informative tokens than A's and C's for the same number of updates. Pilots at ~25–31M parameters do
not predict behaviour at 362M–1B.
