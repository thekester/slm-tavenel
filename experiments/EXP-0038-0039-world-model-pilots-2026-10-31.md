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

### Amendment 2026-11-02 (before any EXP-0038 run): symmetry arms

Motivated by the geometric-deep-learning argument that built-in symmetries improve data efficiency
(Bronstein et al., 2021). Only 4 of the board's 8 symmetries leave the Othello opening unchanged
(identity, 180-degree rotation, both diagonal reflections; `slm.worlds.othello.SYMMETRIES`); only they
map legal games to legal games. Because the uniform random policy already makes the game distribution
symmetric, augmentation is tested in a low-data regime:

| Arm | Data | Updates |
|---|---|---|
| S0 | first 50,000 training games (`data/worlds/othello-v1-50k`) | 5,000 x 128 games |
| S1 | the same games under the 4 symmetries, 200,000 rows (`data/worlds/othello-v1-50k-sym4`) | 5,000 x 128 games |

Same model, optimizer, schedule shape, evaluation games and metrics as A and C, plus an equivariance
measure for every arm (top-move agreement and mean total-variation distance between p(next | s(g)) and
p(next | g) mapped by s, on 500 validation games). **Criterion:** S1 shows a data-efficiency gain if its
legal-move rate and best-layer probe accuracy both exceed S0's by at least one percentage point;
equivariance is reported for all arms (A and C included) as a descriptive measure.

### Results A vs C (2026-11-04)

Both arms ran the full 15,000 updates without instability (C: JEPA loss ~0.058 at the end).
Evaluations: `runs/EXP-0038-evaluations/othello-a-ntp.json` and `othello-c-jepa.json` (final
checkpoints, sha256 `7a6afc39…` and `8b1e4de5…`).

| Measure | A, next move | C, + JEPA | C − A |
|---|---:|---:|---:|
| Legal-move rate, in distribution | 98.59% | 98.45% | −0.14 pt |
| Legal-move rate, greedy transfer | 97.92% | 97.88% | −0.04 pt |
| Best-layer probe accuracy, in distribution | 92.28% (layer 7) | 93.00% (layer 8) | +0.72 pt |
| Best-layer probe accuracy, greedy transfer | 91.95% (layer 7) | 92.66% (layer 8) | +0.71 pt |
| Majority-class baseline (in dist. / transfer) | 53.3% / 53.9% | same | – |
| Equivariance: top-move agreement / mean TV | 0.239 / 0.109 | 0.234 / 0.114 | – |

**Verdict: no measurable difference** under the preregistered rule (probe gain below one point on both
test sets, and a slightly lower legal-move rate). Descriptively, JEPA shifts where the board is most
linearly decodable: C is lower than A at layers 5–6 (81.8/89.0% vs 83.0/90.9%) but higher at the
output layers 7–8, where A's accuracy drops after layer 7 and C's keeps rising, consistent with the
JEPA loss being applied to the last hidden state. Both models track the board well above the majority
baseline and transfer to the greedy play style with under one point of loss, so the linear world
representation emerges from next-move prediction alone, as in Li et al. (2023) and Nanda et al.
(2023, where a linear "mine/theirs" encoding is found).
Equivariance is low for both (top-move agreement ~0.24): under a uniformly random training policy many
legal moves are near-equally likely, so the top move is unstable; the total-variation distance (~0.11)
is the more informative number and is similar for A and C. Single seed per arm.

### Results S0 vs S1, symmetry arms (2026-11-05)

Evaluations: `runs/EXP-0038-evaluations/othello-s0-50k.json` and `othello-s1-50k-sym4.json` (final
checkpoints, sha256 `d4b98f6f…` and `eb39b2bc…`). The S1 evaluation crashed natively on the GPU
(exit 0xC0000005, no Python traceback) and was rerun unchanged on the CPU.

| Measure | S0, 50k games | S1, same games x 4 symmetries | S1 − S0 |
|---|---:|---:|---:|
| Legal-move rate, in distribution | 80.62% | 94.49% | +13.9 pt |
| Legal-move rate, greedy transfer | 78.93% | 92.79% | +13.9 pt |
| Best-layer probe accuracy (layer 7), in distribution | 78.33% | 85.89% | +7.6 pt |
| Best-layer probe accuracy (layer 7), greedy transfer | 78.74% | 85.60% | +6.9 pt |
| Validation loss, step 1,000 / 3,000 / 5,000 | 2.58 / 3.08 / 4.29 | 2.33 / 2.24 / 2.23 | – |
| Equivariance: top-move agreement / mean TV | 0.141 / 0.710 | 0.243 / 0.246 | – |

**Verdict: the preregistered data-efficiency criterion is met** (both legal-move rate and best-layer
probe accuracy higher by far more than one point). The mechanism is visible in the validation loss: S0
sees each of its 50,000 games about 13 times and overfits from step 1,000 on (validation loss rises
from 2.58 to 4.29), while S1 sees each of its 200,000 symmetric rows about 3 times and keeps improving.
Augmentation also makes the model far more equivariant (TV 0.71 to 0.25), approaching the 1M-game arms
(0.11). For reference, A (1M distinct games, 15,000 updates) reaches 98.6% legal moves and 92.3% probe
accuracy, so four symmetric copies recover a large part, but not all, of the gap to 20x more data.

Limits: this shows that the 4 opening-preserving symmetries are a valid and effective data multiplier
in a low-data regime, as the geometric-deep-learning argument predicts; it does not separate symmetry
from "any 4x more distinct sequences" (an arm with 200,000 real games would, and is a cheap follow-up).
Single seed; S0's numbers are those of its final, overfitted checkpoint, as preregistered (no early
stopping).

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

### Results A / B / C (2026-11-05)

All arms ran 10,000 updates without instability (final grad_norm 0.02–0.04; C's JEPA loss fell to ~1e-5).
Evaluations: `runs/EXP-0039-evaluations/` (final checkpoints, sha256 A `b8f775b5…`, B `40dde040…`, C
`af1aa208…`).

| Measure | A, transitions | C, + JEPA | B, plans |
|---|---:|---:|---:|
| Next-state exact match, validation (5,000) | 99.98% | 100% | – |
| Intervention split, exact match (4,998) | 99.94% (3 errors, all on state-changing actions) | 100% | – |
| Held-out layouts, exact match (5,000) | 100% | 100% | – |
| Planning success, in distribution (200) | 100%, optimal length | 100%, optimal length | 100% first attempt |
| Planning success, held-out layouts (200) | 100%, optimal length | 100%, optimal length | 100% first attempt |

**Verdict: no measurable difference; the pilot is at ceiling and therefore uninformative.** C vs A:
+0.06 pt on interventions and 0 pt on held-out planning, far below the preregistered two points. B solves
every world on its first plan, so its verifier tool was never needed (mean attempts 1.0). A and C also
expand exactly the same number of nodes (277.1 / 270.9 on average), because the model-based BFS only
depends on predicted transitions, which are all correct. A 6 x 6 room with one key and one door and
plans of 6–19 steps is too easy for 25M-parameter models trained on 1M transitions; held-out layouts are
solved perfectly too, so this world does not separate "next-token learner" from "world model".

Consequence for phase 2 (to be preregistered before any run): a harder world where errors exist, e.g.
larger rooms (10 x 10+), several keys/doors with ordering constraints, partial observability or
stochastic transitions, longer optimal plans (30+ steps), and fewer training transitions (data-efficiency
curve), before testing D (closed loop). Single seed per arm.

Configuration D (closed loop: training on the model's own verifier-graded transitions) is a second
phase, designed after A/B/C, because its data depend on a trained model; it will be preregistered
separately before it runs.

## Limitations fixed in advance

One seed per arm; bootstrap intervals over games/worlds only. The JEPA weight (1.0) and EMA (0.996) are
not tuned; a null result for C does not rule out other settings. B's training rows carry fewer
informative tokens than A's and C's for the same number of updates. Pilots at ~25–31M parameters do
not predict behaviour at 362M–1B.
