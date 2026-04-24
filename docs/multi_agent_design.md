# Multi-Agent Design Specification (Round 2)

This document is a **locked-down specification** for the three new multi-agent
scenarios layered on top of the Round 1 environment. It defines the exact
action menus, observation fields, reward weights, seed ranges, reputation
mechanics, competitor personalities, and seed-to-slot rotation. Any deviation
from this document in Prompts 2/3 requires a revision here first.

All scenarios are described in **universal game-theoretic terms**. Telecom
flavor appears only inside per-scenario description strings shown to the agent.

---

## 1. Scenario taxonomy (game-theoretic)

| Short name | Game type                                                    | Players | Horizon    |
| ---------- | ------------------------------------------------------------ | ------- | ---------- |
| Task A     | Sealed-bid auction with partial observability                | 3       | 6 rounds   |
| Task B     | Dispute-resolution game w/ opponent-type inference           | 2       | 1 round    |
| Task C     | Iterated prisoner's dilemma with a reputation-tracking referee | 3     | N episodes |

---

## 2. Action menus (exact enum values)

### 2.1 Task A — sealed-bid auction
`MultiAgentAction` is populated as follows:

| Field              | Type   | Values                          |
| ------------------ | ------ | ------------------------------- |
| `bid_amount`       | float  | `[0.0, remaining_budget]`       |
| `dispute_choice`   | None   | unused                          |
| `cooperation_flag` | None   | unused                          |
| `justification`    | str    | free-form, capped at 500 chars  |

### 2.2 Task B — dispute resolution
`dispute_choice` is a `DisputeChoice` enum:

| Enum member | Semantics                                               |
| ----------- | ------------------------------------------------------- |
| `CONCEDE`   | Unilaterally yield; payoff = low, no escalation         |
| `NEGOTIATE` | Offer a compromise; payoff depends on opponent type     |
| `ESCALATE`  | Refuse to yield; payoff = high vs soft, low vs hard     |
| `AUDIT`     | Request third-party audit; costly but reveals true type |

### 2.3 Task C — iterated prisoner's dilemma with referee
`cooperation_flag` is a `CooperationChoice` enum:

| Enum member | Semantics                                          |
| ----------- | -------------------------------------------------- |
| `COOPERATE` | Share resource; grow reputation                    |
| `DEFECT`    | Hoard resource; shrink reputation                  |
| `ABSTAIN`   | No action this round; reputation unchanged         |

### 2.4 Universal fields

| Field           | Type  | Notes                                             |
| --------------- | ----- | ------------------------------------------------- |
| `justification` | str   | Required by process-bonus scoring (all tasks)     |

Single-agent Round 1 code continues to construct `SpectrumAction` directly;
`MultiAgentAction` fields are all `Optional` so the new type can also be
deserialized from a minimal payload without breaking Round 1 call sites.

---

## 3. Observation fields (exact names, types, units)

`MultiAgentObservation` extends `SpectrumObservation`. The agent sees:

| Field name                  | Type                           | Units / semantics                                                                                               |
| --------------------------- | ------------------------------ | --------------------------------------------------------------------------------------------------------------- |
| *(inherited)*               | —                              | All Round 1 fields (`spectrum_grid`, `current_request`, `regulatory_rules`, `step_number`, etc.)                |
| `competitor_bid_history`    | `List[List[float]]`            | Per-competitor list of past bid amounts, in abstract currency units. Empty inner list if a competitor hasn't bid yet. Slot index is stable within an episode (see §7). |
| `reputation_score`          | `float`                        | This agent's own reputation, in `[0.0, 1.0]`. See §5 for mechanics.                                             |
| `oversight_events`          | `List[Dict[str, Any]]`         | Serialized `OversightEvent` records emitted by the referee in prior steps. Ordered oldest-first.                |
| `remaining_budget`          | `float`                        | Currency units the agent has left to spend in the current episode. Applies to Task A only; `0.0` otherwise.     |
| `opponent_slot_indices`     | `List[int]`                    | Stable integer IDs (0..N-1) for other players in the game. No personality label attached — the agent must infer type from `competitor_bid_history`. |
| `round_index`               | `int`                          | Zero-indexed round within the current episode.                                                                  |
| `total_rounds`              | `int`                          | Episode horizon. Equals 6 for Task A, 1 for Task B, `episode_length` for Task C.                                |

**Critical:** `competitor_bid_history` never contains a `"personality"` field.
The agent sees only observable behavior and must infer type.

---

## 4. Reward component weights

Per-step reward is a convex combination of four components. Weights sum to
exactly **1.0**.

| Component          | Weight | Score source                                                                    |
| ------------------ | ------ | ------------------------------------------------------------------------------- |
| `revenue`          | 0.35   | Normalized payoff from auction proceeds / game payoff matrix                    |
| `interference`     | 0.20   | Penalty for breaking band-adjacency or interfering with other players' awards   |
| `compliance`       | 0.25   | Regulatory rule adherence, measured from regulator oversight events             |
| `justification`    | 0.20   | Keyword rubric over the reasoning string plus process bonuses (see below)       |

> **Weight reconciliation (Prompt 2 revision):** Prior revision of this
> document listed weights `0.35 / 0.20 / 0.30 / 0.15`. Prompt 2 of the
> implementation plan respecifies them to `0.35 / 0.20 / 0.25 / 0.20`;
> the canonical values in `rewards.py::REWARD_WEIGHTS` reflect the new
> split. All weights still sum to `1.0`.

**Process-bonus sub-weights inside `justification` (applied to the raw
`[0, 1]` justification score *before* the 0.20 bucket weight is
applied):**

| Sub-component                              | Weight inside bucket |
| ------------------------------------------ | -------------------- |
| Competitor-reference bonus                 | 0.05                 |
| Budget-reference bonus                     | 0.05                 |

The remaining 0.90 of the `[0, 1]` justification score is a base
keyword rubric over six topic families (reasoning, auction-awareness,
decision-theoretic terms, compliance, opponent-awareness, cost/benefit).
This base score is identical in structure to Round 1's justification
scorer so Round 1 compatibility is preserved.

---

## 5. Seed ranges

| Partition            | Seeds      | Count | Use                                               |
| -------------------- | ---------- | ----- | ------------------------------------------------- |
| Training             | `0..199`   | 200   | Exposed to the learner during training            |
| Held-out evaluation  | `200..299` | 100   | Used for final scoring only, never seen in training |

The two ranges **must not overlap**. Any scenario generator that returns the
same concrete scenario for a training seed and an eval seed is considered a
data leak and is a test failure (see `tests/test_new_scenarios.py::test_train_eval_disjoint`).

---

## 6. Reputation score mechanics

Reputation is a scalar in `[0.0, 1.0]` maintained per operator by the
referee. At episode start the value is `0.5`. Updates are piecewise-linear:

| Event                             | Delta      |
| --------------------------------- | ---------- |
| Defection recorded                | `-0.10`    |
| Cooperation recorded              | `+0.05`    |
| Abstention                        | `0.00`     |

After applying the delta, the score is clamped:

```
reputation = min(1.0, max(0.0, reputation + delta))
```

Floor: `0.0`. Cap: `1.0`. Reputation is observed via the `reputation_score`
field and is **the only reputation signal the learner sees** — there is no
leaked personality label.

---

## 7. Competitor personality types

Three personality archetypes govern the scripted opponents. Personalities
are described purely by **behavior pattern**; the agent never sees a label.

| Archetype      | Behavior                                                                                                                                               |
| -------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Aggressive** | In auctions: bids 0.80–0.95 of remaining budget early rounds, tapers to 0.40 late. In disputes: `ESCALATE` with probability 0.80. In coalitions: `DEFECT` with probability 0.70. |
| **Conservative** | In auctions: bids 0.15–0.30 of remaining budget with low variance. In disputes: `NEGOTIATE` (0.60) / `CONCEDE` (0.40). In coalitions: `COOPERATE` with probability 0.80.        |
| **Mimicking**  | Plays the last observed action of the highest-reputation peer, breaking ties toward `COOPERATE` / `NEGOTIATE`. On round 0, acts Conservative.                                   |

The learner **must infer** which archetype it faces from
`competitor_bid_history` (auction) or `oversight_events` (dispute / coalition).

---

## 8. Seed → competitor-slot rotation

Each episode has `N-1` competitor slots (2 for Task A, 1 for Task B, 2 for
Task C). Personalities are assigned to slots by a deterministic rotation
derived from the seed:

```python
ARCHETYPES = ["Aggressive", "Conservative", "Mimicking"]

def rotate(seed: int, num_slots: int) -> list[str]:
    base = list(ARCHETYPES)
    # rotate so that slot 0 = ARCHETYPES[seed % 3]
    rot = seed % len(base)
    ordered = base[rot:] + base[:rot]
    # reverse pairing for even seeds to further diversify
    if seed % 2 == 0:
        ordered = list(reversed(ordered))
    return ordered[:num_slots]
```

This yields a stable mapping: `seed → ordered list of archetype strings`
truncated to `num_slots`. Across seeds `0..199` and `200..299`, every
archetype appears in every slot position in roughly equal proportion, so
training cannot shortcut-learn "slot 0 is always Aggressive".

The rotation function lives in `scenarios.py` as `_rotate_archetypes(seed, num_slots)`
and is imported by all three generators.

---

## 9. Ground-truth computation (summary; full details in generators)

* **Task A** — Ground-truth bid sequence is computed by **symmetric Bayesian
  Nash equilibrium approximation for first-price sealed-bid auctions**:
  `bid_i = v_i * (n-1) / n` where `v_i` is the operator's valuation and `n`
  is the number of bidders. A light backward-induction pass over the 6 rounds
  adjusts bids downward when remaining supply exceeds remaining demand.
  This is a **documented approximation**; computing the true equilibrium over
  3-player correlated-valuation games is not tractable inside this generator.

* **Task B** — Ground truth is the **best response to the posterior over
  opponent type** given prior round behavior. Priors over {Aggressive,
  Conservative, Mimicking} are uniform at episode start. A fixed 4×3 payoff
  matrix (action × opponent type) is used; the optimal action is
  `argmax_a E[payoff(a, type)]`.

* **Task C** — Ground truth is the simplification: cooperate when
  `reputation < 0.7`, otherwise either action is acceptable. This is
  explicitly **not** the full cooperative equilibrium of the iterated game;
  computing the latter is out of scope for the hackathon.

---

## 10. Out of scope for this document

The following are deliberately **not** specified here and are deferred to
Prompt 2 / Prompt 3:

* Concrete referee logic that emits `OversightEvent`s
* Concrete scripted operator policies for each archetype
* Reward-function code that consumes the ground truth
* Multi-agent `step()` orchestration in `server/spectrum_environment.py`
* `openenv.yaml` task entries for the new scenarios
* README updates
