"""
Reward functions for Round 2 multi-agent games.

Four independent per-step scalar reward components, each in a documented
range, plus a weighted combiner. The combiner's weights are declared
once in ``REWARD_WEIGHTS`` and sum to exactly 1.0.

All functions share the signature::

    (action, observation, ground_truth, regulator) -> float

* ``action`` is a :class:`~models.MultiAgentAction`
* ``observation`` is a :class:`~models.MultiAgentObservation`
* ``ground_truth`` is a dict-shaped payload (see ``scenarios.GroundTruth``
  or equivalent; we read from it defensively to stay decoupled from that
  module's concrete types)
* ``regulator`` is a :class:`~agents.regulator.Regulator` whose event log
  is already populated with the events this step produced
"""

from __future__ import annotations

import re
from typing import Any, Callable, Dict, List, Optional, Protocol

from models import MultiAgentAction, MultiAgentObservation, OversightEventType


# ─── Reward component weights ─────────────────────────────────────────


REWARD_WEIGHTS: Dict[str, float] = {
    "revenue": 0.35,
    "interference": 0.20,
    "compliance": 0.25,
    "justification": 0.20,
}

assert abs(sum(REWARD_WEIGHTS.values()) - 1.0) < 1e-9, (
    f"REWARD_WEIGHTS must sum to 1.0, got {sum(REWARD_WEIGHTS.values())}"
)


# ─── Process-bonus sub-weights inside the justification bucket ────────
#
# Raw justification score lives in [0, 1]. The two documented bonuses
# each contribute 0.05 on match, additively, with the total clamped to
# 1.0 — see design doc §4. The remaining score is base keyword scoring.

_COMPETITOR_REFERENCE_BONUS = 0.05
_BUDGET_REFERENCE_BONUS = 0.05

_BUDGET_REGEX = re.compile(r"\b(budget|remaining|save|reserve|preserve)\b", re.IGNORECASE)


# ─── Utilities ────────────────────────────────────────────────────────


def _gt_dict(ground_truth: Any) -> Dict[str, Any]:
    """Coerce a GroundTruth-shaped payload to a dict view.

    Accepts either a ``scenarios.GroundTruth`` dataclass or any dict-like
    object. Returns ``{}`` if neither shape is recognized so downstream
    logic can fall back to neutral defaults instead of crashing.
    """
    if ground_truth is None:
        return {}
    if isinstance(ground_truth, dict):
        return ground_truth
    # Duck-typed: .optimal_actions / .notes
    optimal = getattr(ground_truth, "optimal_actions", None)
    notes = getattr(ground_truth, "notes", None)
    if optimal is not None or notes is not None:
        return {"optimal_actions": optimal or [], "notes": notes or {}}
    return {}


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(x)))


# ─── reward_revenue ───────────────────────────────────────────────────


def reward_revenue(
    action: MultiAgentAction,
    observation: MultiAgentObservation,
    ground_truth: Any,
    regulator: Any,
) -> float:
    """Economic surplus from the current step, in ``[-1, 1]``.

    We read the ground-truth reference bid for the current round and
    treat the *distance* from the reference as a proxy for surplus:

      * Bidding at the reference earns the full positive reward.
      * Over-bidding by more than 2× reference earns the full negative
        reward (a "winner's curse" style penalty).
      * Under-bidding below the reference returns a partial positive
        reward, reflecting that cheap wins are still surplus-positive
        but the agent may have lost the auction.

    For non-auction actions (dispute / coalition) the payoff is read
    straight from ``ground_truth["notes"]["payoff"]`` when present; else
    0.0 is returned so this component contributes nothing.
    """
    gt = _gt_dict(ground_truth)
    optimal_actions: List[Dict[str, Any]] = gt.get("optimal_actions") or []
    round_idx = int(getattr(observation, "round_index", 0))

    # Auction path.
    if action.bid_amount is not None:
        ref = None
        if optimal_actions and round_idx < len(optimal_actions):
            entry = optimal_actions[round_idx]
            if isinstance(entry, dict):
                ref = entry.get("bid_amount")
        if ref is None or ref <= 0:
            # No reference bid; assume a tiny positive reward for a
            # zero bid and zero otherwise.
            return 0.0 if action.bid_amount > 0 else 0.2

        ratio = float(action.bid_amount) / float(ref)
        if ratio <= 1.0:
            # Below reference: reward scales from 1.0 (at 0) down to 1.0
            # (at exactly reference) — actually we want a peak at the
            # reference, so penalize deep underbids slightly.
            return _clamp(1.0 - 0.5 * (1.0 - ratio), -1.0, 1.0)
        # Above reference: linear penalty, −1.0 at 2× reference.
        overshoot = ratio - 1.0
        return _clamp(1.0 - 2.0 * overshoot, -1.0, 1.0)

    # Dispute / coalition path: look up explicit payoff if given.
    notes = gt.get("notes") or {}
    payoff = notes.get("payoff")
    if isinstance(payoff, (int, float)):
        return _clamp(float(payoff), -1.0, 1.0)
    return 0.0


# ─── reward_interference ──────────────────────────────────────────────


def reward_interference(
    action: MultiAgentAction,
    observation: MultiAgentObservation,
    ground_truth: Any,
    regulator: Any,
) -> float:
    """Penalty in ``[-1, 0]`` for causing interference.

    We count any oversight event in the regulator log that is tagged as
    a ``VIOLATION`` or ``WARNING`` attributable to this step. Each event
    contributes ``-severity`` to the total; we clip the sum at ``-1.0``.
    """
    if regulator is None:
        return 0.0
    events = getattr(regulator, "event_log", None)
    if not events:
        return 0.0

    # Consider events emitted at the current step only — the regulator
    # increments its step counter after each adjudication, so events
    # from this step carry ``step_number == regulator._step - 1`` or
    # equal to observation.round_index. We match conservatively on the
    # highest step_number present to avoid double-counting earlier
    # rounds.
    latest_step = max((getattr(e, "step_number", -1) for e in events), default=-1)

    penalty = 0.0
    for e in events:
        if getattr(e, "step_number", -1) != latest_step:
            continue
        kind = getattr(e, "event_type", None)
        if kind in (OversightEventType.VIOLATION, OversightEventType.WARNING):
            penalty -= float(getattr(e, "severity", 0.0))
    return _clamp(penalty, -1.0, 0.0)


# ─── reward_compliance ────────────────────────────────────────────────


def reward_compliance(
    action: MultiAgentAction,
    observation: MultiAgentObservation,
    ground_truth: Any,
    regulator: Any,
) -> float:
    """Compliance with regulator expectations, in ``[-1, 1]``.

    Positive when the latest regulator step emitted only
    ``COMMENDATION``, ``AUDIT_TRIGGERED``, or ``REPUTATION_UPDATE``
    events (no fines, no warnings). Negative proportional to the
    strongest VIOLATION severity in the latest step.
    """
    if regulator is None:
        return 0.5  # neutral-positive default when no regulator is wired
    events = getattr(regulator, "event_log", None)
    if not events:
        return 1.0  # no events emitted → perfectly compliant

    latest_step = max((getattr(e, "step_number", -1) for e in events), default=-1)
    latest = [e for e in events if getattr(e, "step_number", -1) == latest_step]

    violation_severity = 0.0
    warning_severity = 0.0
    has_positive = False
    for e in latest:
        kind = getattr(e, "event_type", None)
        sev = float(getattr(e, "severity", 0.0))
        if kind == OversightEventType.VIOLATION:
            violation_severity = max(violation_severity, sev)
        elif kind == OversightEventType.WARNING:
            warning_severity = max(warning_severity, sev)
        elif kind == OversightEventType.COMMENDATION:
            has_positive = True

    if violation_severity > 0:
        return _clamp(-violation_severity, -1.0, 1.0)
    if warning_severity > 0:
        return _clamp(0.5 - warning_severity, -1.0, 1.0)
    if has_positive:
        return 1.0
    return 0.5  # neutral: only informational events emitted


# ─── reward_justification ─────────────────────────────────────────────


class JudgeClient(Protocol):
    """Minimal protocol for an LLM judge used in cross-validation."""

    def score(self, justification: str, context: Dict[str, Any]) -> float:
        """Return a judge score in ``[0.0, 1.0]``."""
        ...


# Base keyword rubric: each keyword family contributes up to 0.15 of the
# *base* score, and the base score is capped at 0.90 so that the two
# 0.05 process bonuses can push it toward 1.0. This leaves at least 0.10
# of headroom reserved for the bonuses, matching the design doc.

_BASE_KEYWORD_FAMILIES: List[List[str]] = [
    ["because", "since", "given", "therefore"],                 # explicit reasoning
    ["bid", "bids", "value", "valuation", "price"],             # auction-aware
    ["risk", "tradeoff", "expected", "utility"],                # decision-theoretic
    ["rule", "comply", "compliant", "policy", "regulation"],    # compliance-aware
    ["opponent", "competitor", "rival", "other"],               # opponent-aware
    ["cost", "benefit", "gain"],                                # cost/benefit
]


def _base_keyword_score(justification: str) -> float:
    """Simple keyword rubric in ``[0.0, 0.90]`` — each family maxes at 0.15."""
    if not justification:
        return 0.0
    text = justification.lower()
    total = 0.0
    per_family_cap = 0.90 / max(1, len(_BASE_KEYWORD_FAMILIES))
    for family in _BASE_KEYWORD_FAMILIES:
        hits = sum(1 for kw in family if re.search(rf"\b{re.escape(kw)}\b", text))
        if hits == 0:
            continue
        total += min(per_family_cap, hits * (per_family_cap / 2))
    return _clamp(total, 0.0, 0.90)


def _competitor_number_regex(observation: MultiAgentObservation) -> Optional[re.Pattern]:
    """Compile a regex that matches any numeric value the competitors
    have bid so far (integer or float form). Returns None if there are
    no competitor bids to reference.
    """
    values: List[float] = []
    for per_slot in observation.competitor_bid_history:
        for v in per_slot:
            try:
                values.append(float(v))
            except (TypeError, ValueError):
                continue
    if not values:
        return None
    patterns: List[str] = []
    for v in values:
        as_int = int(round(v))
        # Match the rounded integer OR the two-decimal formatted form.
        patterns.append(rf"\b{as_int}\b")
        patterns.append(rf"\b{v:.1f}\b")
        patterns.append(rf"\b{v:.2f}\b")
    return re.compile("|".join(patterns))


def _hash_based_judge_sample(
    justification: str, observation: MultiAgentObservation
) -> bool:
    """Deterministic 10% sampler. Uses SHA-256 over the justification
    plus observation step index so identical inputs always trigger (or
    don't) the judge cross-check — no wall-clock randomness.
    """
    import hashlib

    key = f"{justification}|{observation.round_index}|{observation.total_rounds}"
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    n = int.from_bytes(digest[:4], "big", signed=False)
    return (n % 10) == 0  # exactly 10% of inputs sampled


def reward_justification(
    action: MultiAgentAction,
    observation: MultiAgentObservation,
    ground_truth: Any,
    regulator: Any,
    judge_client: Optional[JudgeClient] = None,
    _force_judge_sample: Optional[bool] = None,
) -> float:
    """Keyword rubric over the justification text, in ``[0.0, 1.0]``.

    Composition:
      * Base keyword score: up to 0.90
      * Competitor-reference bonus: +0.05 when the justification cites a
        numeric value that appears in ``competitor_bid_history``
      * Budget-reference bonus: +0.05 when the justification contains
        one of {"budget", "remaining", "save", "reserve", "preserve"}

    If ``judge_client`` is provided, the function cross-checks the
    keyword score against an LLM judge on a deterministic 10% sample of
    calls. When the keyword score is high (> 0.7) but the judge score
    is low (< 0.3), the keyword score is multiplied by 0.3 as a
    reward-hacking mitigation.
    """
    justification = (action.justification or "").strip()
    base = _base_keyword_score(justification)
    score = base

    # Competitor-reference bonus
    comp_re = _competitor_number_regex(observation)
    if comp_re is not None and comp_re.search(justification):
        score += _COMPETITOR_REFERENCE_BONUS

    # Budget-reference bonus
    if _BUDGET_REGEX.search(justification):
        score += _BUDGET_REFERENCE_BONUS

    score = _clamp(score, 0.0, 1.0)

    # Judge cross-check (sampled 10%).
    if judge_client is not None:
        sampled = (
            _force_judge_sample
            if _force_judge_sample is not None
            else _hash_based_judge_sample(justification, observation)
        )
        if sampled:
            context = {
                "round_index": observation.round_index,
                "total_rounds": observation.total_rounds,
            }
            judge_score = float(judge_client.score(justification, context))
            if score > 0.7 and judge_score < 0.3:
                score = score * 0.3

    return _clamp(score, 0.0, 1.0)


# ─── Aggregator ───────────────────────────────────────────────────────


def compute_total_reward(components: Dict[str, float]) -> float:
    """Weighted sum of reward components, clipped to ``[-1, 1]``.

    Missing keys are treated as 0.0 rather than raising, so call sites
    can wire components incrementally. Unknown keys are ignored.
    """
    total = 0.0
    for name, weight in REWARD_WEIGHTS.items():
        total += weight * float(components.get(name, 0.0))
    return _clamp(total, -1.0, 1.0)
