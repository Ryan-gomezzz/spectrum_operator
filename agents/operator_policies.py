"""
Scripted operator policies for Round 2 multi-agent games.

Each policy is a *deterministic* function from (observation, state) to an
action choice. "Deterministic" means that for fixed observation + state,
the returned value is byte-identical across calls in the same process
and across processes. No call to ``random.random()`` without an explicit
seed is permitted; all pseudo-randomness is derived from
``hash_to_unit(observation, state, salt)`` which folds the inputs into a
stable integer modulus.

Policies are labelled by *behavior pattern*, not by the label itself —
the learned agent never sees which archetype it faces. Archetypes are
implementation detail of the scripted opponent, not observable state.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Protocol, Tuple, Union

from models import (
    CooperationChoice,
    DisputeChoice,
    MultiAgentObservation,
    OperatorState,
)
from scenarios import _rotate_archetypes


# ─── Deterministic hashing helpers ─────────────────────────────────────


def _canonical_obs_state_tuple(
    observation: MultiAgentObservation, state: OperatorState
) -> Tuple[Any, ...]:
    """Project observation + state to a hashable, order-stable tuple.

    Only fields that are relevant to policy decisions are included, so
    irrelevant state churn (e.g. incremented metadata counters) does not
    flip the policy output.
    """
    bid_hist = tuple(tuple(inner) for inner in observation.competitor_bid_history)
    oversight = tuple(
        (
            ev.get("event_type"),
            ev.get("operator_id"),
            float(ev.get("severity", 0.0)),
            int(ev.get("step_number", 0)),
        )
        for ev in observation.oversight_events
    )
    return (
        bid_hist,
        round(float(observation.reputation_score), 6),
        oversight,
        round(float(observation.remaining_budget), 6),
        tuple(observation.opponent_slot_indices),
        int(observation.round_index),
        int(observation.total_rounds),
        state.operator_id,
        round(float(state.budget), 6),
        round(float(state.reputation), 6),
        tuple(state.licenses_held),
    )


def _hash_to_unit(
    observation: MultiAgentObservation,
    state: OperatorState,
    salt: str,
) -> float:
    """Deterministic map (observation, state, salt) → float in [0.0, 1.0).

    Uses SHA-256 over a canonical JSON projection rather than Python's
    built-in ``hash``, because ``hash`` is randomized per process for
    strings and would break cross-process determinism.
    """
    payload = json.dumps(
        [salt, _canonical_obs_state_tuple(observation, state)],
        sort_keys=True,
        default=str,
    ).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    # Take the first 8 bytes as an unsigned 64-bit int.
    n = int.from_bytes(digest[:8], "big", signed=False)
    return n / 2 ** 64


def _hash_pick(
    options: List[Any],
    observation: MultiAgentObservation,
    state: OperatorState,
    salt: str,
) -> Any:
    """Deterministically pick one element of ``options``."""
    u = _hash_to_unit(observation, state, salt)
    idx = int(u * len(options))
    if idx >= len(options):
        idx = len(options) - 1
    return options[idx]


def _scale_to_range(u: float, lo: float, hi: float) -> float:
    """Map a unit-interval value u∈[0,1) to [lo, hi]."""
    return lo + u * (hi - lo)


# ─── Policy protocol ───────────────────────────────────────────────────


class OperatorPolicy(Protocol):
    """Minimal interface every scripted operator must satisfy."""

    def decide_bid(
        self, observation: MultiAgentObservation, state: OperatorState
    ) -> float: ...

    def decide_dispute_response(
        self, observation: MultiAgentObservation, state: OperatorState
    ) -> DisputeChoice: ...

    def decide_cooperation(
        self, observation: MultiAgentObservation, state: OperatorState
    ) -> bool: ...


# ─── Archetype: high-risk-tolerance bidder ─────────────────────────────


class AggressiveOperator:
    """Scripted competitor exhibiting high-risk-tolerance bidding.

    Behavioral signature:
      * Bids in the 0.70–0.90 fraction of remaining budget band, tapering
        toward the lower end in late rounds.
      * Responds to disputes with ESCALATE as the dominant strategy,
        occasionally AUDIT when the observed competitor has a long bid
        history that might reveal more info.
      * Defects in coalition games whenever its own reputation is above
        0.3 — i.e. it will only cooperate when its reputation is so low
        that further defection would tank it below the floor.
    """

    label = "Aggressive"

    def decide_bid(
        self, observation: MultiAgentObservation, state: OperatorState
    ) -> float:
        budget = max(0.0, float(state.budget))
        if budget <= 0.0:
            return 0.0

        total = max(1, int(observation.total_rounds))
        r = max(0, min(total - 1, int(observation.round_index)))
        # Taper: early rounds near 0.90, late rounds near 0.70.
        progress = r / max(1, total - 1) if total > 1 else 0.0
        ceiling = 0.90 - 0.20 * progress  # 0.90 → 0.70
        floor = max(0.0, ceiling - 0.10)

        u = _hash_to_unit(observation, state, salt="aggressive:bid")
        frac = _scale_to_range(u, floor, ceiling)
        bid = frac * budget
        # Hard-enforce budget constraint (never bid above what we have).
        return round(min(bid, budget), 6)

    def decide_dispute_response(
        self, observation: MultiAgentObservation, state: OperatorState
    ) -> DisputeChoice:
        # 80% ESCALATE / 20% AUDIT distribution.
        u = _hash_to_unit(observation, state, salt="aggressive:dispute")
        return DisputeChoice.ESCALATE if u < 0.80 else DisputeChoice.AUDIT

    def decide_cooperation(
        self, observation: MultiAgentObservation, state: OperatorState
    ) -> bool:
        # Defect whenever reputation > 0.3. At/below 0.3, cooperate to
        # avoid the floor.
        return not (state.reputation > 0.3)


# ─── Archetype: budget-preserving conservative bidder ──────────────────


class ConservativeOperator:
    """Scripted competitor exhibiting budget preservation.

    Behavioral signature:
      * Bids in the 0.20–0.40 fraction of remaining budget band with low
        variance.
      * Responds to disputes with NEGOTIATE (60%) or CONCEDE (40%) —
        never escalates, never audits.
      * Cooperates in coalition games whenever its own reputation is
        above 0.4 (i.e. the default cooperative case).
    """

    label = "Conservative"

    def decide_bid(
        self, observation: MultiAgentObservation, state: OperatorState
    ) -> float:
        budget = max(0.0, float(state.budget))
        if budget <= 0.0:
            return 0.0

        u = _hash_to_unit(observation, state, salt="conservative:bid")
        frac = _scale_to_range(u, 0.20, 0.40)
        bid = frac * budget
        return round(min(bid, budget), 6)

    def decide_dispute_response(
        self, observation: MultiAgentObservation, state: OperatorState
    ) -> DisputeChoice:
        u = _hash_to_unit(observation, state, salt="conservative:dispute")
        return DisputeChoice.NEGOTIATE if u < 0.60 else DisputeChoice.CONCEDE

    def decide_cooperation(
        self, observation: MultiAgentObservation, state: OperatorState
    ) -> bool:
        return state.reputation > 0.4


# ─── Archetype: reflective / adaptive bidder ───────────────────────────


class MimickingOperator:
    """Scripted competitor that adapts to observed learner behavior.

    Behavioral signature:
      * Bids at the running average of all observed competitor bids
        across the current episode, clipped to [0, budget]. On round 0
        (no observations yet) falls back to the Conservative schedule.
      * Mirrors the learner's last observed dispute response, read off
        the oversight-event log. Falls back to NEGOTIATE when no prior
        response is visible.
      * Mirrors the learner's last observed cooperation action similarly,
        with COOPERATE as the tie-break fallback.

    This forces the learned agent to account for its own reflection in
    the opponent — deceptive play or sudden defection is echoed back.
    """

    label = "Mimicking"

    _CONSERVATIVE_FALLBACK = ConservativeOperator()

    def decide_bid(
        self, observation: MultiAgentObservation, state: OperatorState
    ) -> float:
        budget = max(0.0, float(state.budget))
        if budget <= 0.0:
            return 0.0

        all_bids: List[float] = []
        for per_slot in observation.competitor_bid_history:
            all_bids.extend(float(b) for b in per_slot)
        if not all_bids:
            return self._CONSERVATIVE_FALLBACK.decide_bid(observation, state)
        running_avg = sum(all_bids) / len(all_bids)
        return round(max(0.0, min(running_avg, budget)), 6)

    def decide_dispute_response(
        self, observation: MultiAgentObservation, state: OperatorState
    ) -> DisputeChoice:
        # Walk oversight events newest-first; pull any event carrying a
        # ``dispute_choice`` payload that was attributed to a *different*
        # operator than self.
        for ev in reversed(observation.oversight_events):
            choice = ev.get("dispute_choice")
            source = ev.get("operator_id")
            if choice and source and source != state.operator_id:
                try:
                    return DisputeChoice(choice)
                except ValueError:
                    continue
        return DisputeChoice.NEGOTIATE  # tie-break fallback

    def decide_cooperation(
        self, observation: MultiAgentObservation, state: OperatorState
    ) -> bool:
        for ev in reversed(observation.oversight_events):
            flag = ev.get("cooperation_flag")
            source = ev.get("operator_id")
            if flag and source and source != state.operator_id:
                if flag == CooperationChoice.COOPERATE.value:
                    return True
                if flag == CooperationChoice.DEFECT.value:
                    return False
                # ABSTAIN / other → fall through.
        return True  # tie-break toward cooperation


# ─── Rotation helper ───────────────────────────────────────────────────


_LABEL_TO_POLICY = {
    "Aggressive": AggressiveOperator,
    "Conservative": ConservativeOperator,
    "Mimicking": MimickingOperator,
}


def rotate_policies(scenario_seed: int, num_slots: int = 2) -> List[OperatorPolicy]:
    """Return ``num_slots`` scripted competitor policies in deterministic
    seed-dependent order.

    This wraps ``scenarios._rotate_archetypes`` — the single source of
    truth for the seed → archetype ordering — and instantiates each
    archetype into a concrete policy object. Across training and
    evaluation seed ranges, every archetype appears in every slot with
    roughly equal frequency, so the learner cannot shortcut-learn
    "slot 0 is always Aggressive".
    """
    labels = _rotate_archetypes(scenario_seed, num_slots)
    return [_LABEL_TO_POLICY[label]() for label in labels]
