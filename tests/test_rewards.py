"""
Tests for reward components and the weighted aggregator.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models import (  # noqa: E402
    CooperationChoice,
    DisputeChoice,
    MultiAgentAction,
    MultiAgentObservation,
    OversightEvent,
    OversightEventType,
)
from agents.regulator import Regulator  # noqa: E402
from rewards import (  # noqa: E402
    REWARD_WEIGHTS,
    compute_total_reward,
    reward_compliance,
    reward_interference,
    reward_justification,
    reward_revenue,
)


# ─── Fixtures ────────────────────────────────────────────────────────


def make_obs(**overrides) -> MultiAgentObservation:
    defaults = dict(
        competitor_bid_history=[[]],
        reputation_score=0.5,
        oversight_events=[],
        remaining_budget=50.0,
        opponent_slot_indices=[0],
        round_index=0,
        total_rounds=6,
    )
    defaults.update(overrides)
    return MultiAgentObservation(**defaults)


def auction_gt(ref_bid: float = 5.0, round_idx: int = 0) -> Dict[str, Any]:
    return {
        "optimal_actions": [{"bid_amount": ref_bid, "round_index": round_idx}],
        "notes": {},
    }


# ─── Weight invariants ──────────────────────────────────────────────


def test_reward_weights_sum_to_one():
    assert abs(sum(REWARD_WEIGHTS.values()) - 1.0) < 1e-9


def test_reward_weights_exact_values():
    assert REWARD_WEIGHTS == {
        "revenue": 0.45,
        "interference": 0.05,
        "compliance": 0.10,
        "justification": 0.40,
    }


# ─── reward_revenue ─────────────────────────────────────────────────


def test_revenue_returns_float_in_range():
    action = MultiAgentAction(bid_amount=4.0)
    obs = make_obs()
    r = reward_revenue(action, obs, auction_gt(5.0), Regulator())
    assert isinstance(r, float)
    assert -1.0 <= r <= 1.0


def test_revenue_peaks_at_reference_bid():
    obs = make_obs()
    gt = auction_gt(10.0)
    reg = Regulator()
    r_exact = reward_revenue(MultiAgentAction(bid_amount=10.0), obs, gt, reg)
    r_over = reward_revenue(MultiAgentAction(bid_amount=25.0), obs, gt, reg)
    r_under = reward_revenue(MultiAgentAction(bid_amount=0.0), obs, gt, reg)
    assert r_exact >= r_over
    assert r_exact >= r_under


def test_revenue_heavy_overbid_returns_negative():
    obs = make_obs()
    r = reward_revenue(MultiAgentAction(bid_amount=30.0), obs, auction_gt(5.0), Regulator())
    assert r < 0


def test_revenue_non_auction_action_returns_zero():
    action = MultiAgentAction(dispute_choice=DisputeChoice.NEGOTIATE)
    r = reward_revenue(action, make_obs(), {"notes": {}}, Regulator())
    assert r == 0.0


# ─── reward_interference ────────────────────────────────────────────


def test_interference_returns_zero_with_empty_log():
    action = MultiAgentAction(bid_amount=5.0)
    r = reward_interference(action, make_obs(), auction_gt(), Regulator())
    assert r == 0.0


def test_interference_penalizes_violations():
    reg = Regulator()
    # Force a violation event via collusion detection.
    reg.resolve_auction_round({"op-A": 5.0, "op-B": 5.0})
    r = reward_interference(
        MultiAgentAction(bid_amount=5.0), make_obs(), auction_gt(), reg
    )
    assert -1.0 <= r < 0.0


def test_interference_range_bounded():
    reg = Regulator()
    for _ in range(5):
        reg.resolve_auction_round({"op-A": 5.0, "op-B": 5.0})
    r = reward_interference(
        MultiAgentAction(bid_amount=5.0), make_obs(), auction_gt(), reg
    )
    assert -1.0 <= r <= 0.0


# ─── reward_compliance ──────────────────────────────────────────────


def test_compliance_full_when_no_events():
    r = reward_compliance(
        MultiAgentAction(bid_amount=5.0), make_obs(), auction_gt(), Regulator()
    )
    assert r == 1.0


def test_compliance_negative_after_violation():
    reg = Regulator()
    reg.resolve_auction_round({"op-A": 5.0, "op-B": 5.0})
    r = reward_compliance(
        MultiAgentAction(bid_amount=5.0), make_obs(), auction_gt(), reg
    )
    assert -1.0 <= r < 0.0


def test_compliance_full_after_commendation():
    reg = Regulator()
    reg.adjudicate_dispute(DisputeChoice.NEGOTIATE, DisputeChoice.NEGOTIATE)
    r = reward_compliance(
        MultiAgentAction(dispute_choice=DisputeChoice.NEGOTIATE),
        make_obs(),
        {"notes": {}},
        reg,
    )
    assert r == 1.0


# ─── reward_justification ───────────────────────────────────────────


def test_justification_zero_on_empty_text():
    r = reward_justification(
        MultiAgentAction(bid_amount=1.0, justification=""),
        make_obs(),
        auction_gt(),
        Regulator(),
    )
    assert r == 0.0


def test_justification_range():
    action = MultiAgentAction(
        bid_amount=5.0,
        justification="bid because value of risk vs cost, remaining budget",
    )
    r = reward_justification(action, make_obs(), auction_gt(), Regulator())
    assert 0.0 <= r <= 1.0


def test_justification_bonus_fires_on_competitor_number_reference():
    obs_no_history = make_obs(competitor_bid_history=[[]])
    obs_with_history = make_obs(competitor_bid_history=[[7.0]])
    action = MultiAgentAction(
        bid_amount=5.0,
        justification="I bid because the opponent bid 7",
    )
    base = reward_justification(action, obs_no_history, auction_gt(), Regulator())
    boosted = reward_justification(action, obs_with_history, auction_gt(), Regulator())
    assert boosted > base
    assert abs((boosted - base) - 0.05) < 1e-6


def test_justification_bonus_fires_on_budget_reference():
    obs = make_obs()
    action_with_budget = MultiAgentAction(
        bid_amount=5.0,
        justification="I will preserve some remaining budget for later rounds",
    )
    action_without_budget = MultiAgentAction(
        bid_amount=5.0,
        justification="I think we should go for it because cost exceeds gain",
    )
    with_bonus = reward_justification(action_with_budget, obs, auction_gt(), Regulator())
    without_bonus = reward_justification(action_without_budget, obs, auction_gt(), Regulator())
    # The two justifications hit overlapping keyword families, so the
    # difference between them is exactly the budget bonus plus/minus any
    # differing family hits. We specifically assert the budget bonus is
    # present in the first message (its raw score > the second's).
    assert with_bonus > 0.0
    assert without_bonus > 0.0
    # The budget-reference message must include the +0.05 bonus.
    # We verify by diffing against the same message with the budget
    # keyword stripped out.
    action_stripped = MultiAgentAction(
        bid_amount=5.0,
        justification="I will store some extra funds for later rounds",
    )
    stripped = reward_justification(action_stripped, obs, auction_gt(), Regulator())
    assert with_bonus > stripped


def test_justification_keyword_stuffing_mitigation_triggers():
    """When keyword score > 0.7 but judge score < 0.3, return keyword * 0.3."""

    class StubJudge:
        def score(self, justification: str, context: Dict[str, Any]) -> float:
            return 0.05  # very low

    action = MultiAgentAction(
        bid_amount=5.0,
        justification=(
            "because bid value risk tradeoff rule comply opponent cost "
            "budget remaining since competitor rival expected utility policy"
        ),
    )
    obs = make_obs(competitor_bid_history=[[7.0]])
    raw = reward_justification(action, obs, auction_gt(), Regulator())
    assert raw > 0.7, f"precondition failed, raw score = {raw}"

    # Force the judge to be sampled.
    with_judge = reward_justification(
        action,
        obs,
        auction_gt(),
        Regulator(),
        judge_client=StubJudge(),
        _force_judge_sample=True,
    )
    assert with_judge == pytest.approx(raw * 0.3, abs=1e-9), (
        f"expected keyword-stuffing penalty; raw={raw}, with_judge={with_judge}"
    )


def test_justification_no_mitigation_when_judge_agrees():
    class AgreeJudge:
        def score(self, justification: str, context: Dict[str, Any]) -> float:
            return 0.9

    action = MultiAgentAction(
        bid_amount=5.0,
        justification=(
            "because bid value risk tradeoff rule comply opponent cost "
            "budget remaining since competitor rival"
        ),
    )
    obs = make_obs(competitor_bid_history=[[7.0]])
    raw = reward_justification(action, obs, auction_gt(), Regulator())
    with_judge = reward_justification(
        action,
        obs,
        auction_gt(),
        Regulator(),
        judge_client=AgreeJudge(),
        _force_judge_sample=True,
    )
    assert with_judge == raw


# ─── compute_total_reward ───────────────────────────────────────────


def test_total_reward_is_weighted_sum():
    comps = {"revenue": 1.0, "interference": 0.0, "compliance": 0.5, "justification": 0.4}
    expected = (
        0.45 * 1.0 + 0.05 * 0.0 + 0.10 * 0.5 + 0.40 * 0.4
    )
    assert compute_total_reward(comps) == pytest.approx(expected)


def test_total_reward_clips_to_minus_one_one():
    extreme = {"revenue": -10.0, "interference": -10.0, "compliance": -10.0, "justification": -10.0}
    assert compute_total_reward(extreme) == -1.0
    extreme_pos = {"revenue": 10.0, "interference": 10.0, "compliance": 10.0, "justification": 10.0}
    assert compute_total_reward(extreme_pos) == 1.0


def test_total_reward_handles_missing_components():
    assert compute_total_reward({"revenue": 1.0}) == pytest.approx(0.45)
    assert compute_total_reward({}) == 0.0
