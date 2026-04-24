"""
Tests for scripted operator policies.

Verifies:
  * Determinism — same (obs, state) → same output across repeated calls
  * Budget constraint — every bid satisfies ``bid <= state.budget``
  * rotate_policies yields different orderings for different seeds
  * rotate_policies is itself deterministic per seed
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models import (  # noqa: E402
    CooperationChoice,
    DisputeChoice,
    MultiAgentObservation,
    OperatorState,
)
from agents.operator_policies import (  # noqa: E402
    AggressiveOperator,
    ConservativeOperator,
    MimickingOperator,
    rotate_policies,
)


# ─── Fixtures ────────────────────────────────────────────────────────


def make_obs(**overrides):
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


def make_state(**overrides):
    defaults = dict(operator_id="op-x", budget=50.0, reputation=0.5)
    defaults.update(overrides)
    return OperatorState(**defaults)


# ─── Determinism ─────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "PolicyCls",
    [AggressiveOperator, ConservativeOperator, MimickingOperator],
)
def test_policy_is_deterministic(PolicyCls):
    policy = PolicyCls()
    obs = make_obs()
    state = make_state()

    bids = {policy.decide_bid(obs, state) for _ in range(3)}
    disputes = {policy.decide_dispute_response(obs, state) for _ in range(3)}
    coops = {policy.decide_cooperation(obs, state) for _ in range(3)}

    assert len(bids) == 1, f"Non-deterministic bid: {bids}"
    assert len(disputes) == 1, f"Non-deterministic dispute: {disputes}"
    assert len(coops) == 1, f"Non-deterministic cooperation: {coops}"


@pytest.mark.parametrize(
    "PolicyCls",
    [AggressiveOperator, ConservativeOperator, MimickingOperator],
)
def test_policy_changes_output_with_input(PolicyCls):
    """Different states should be able to produce different bids.

    (Sanity check against a degenerate policy that always returns the
    same value regardless of input.)
    """
    policy = PolicyCls()
    bids = set()
    for rep in (0.1, 0.5, 0.9):
        for budget in (20.0, 60.0, 100.0):
            obs = make_obs(reputation_score=rep, remaining_budget=budget)
            state = make_state(budget=budget, reputation=rep)
            bids.add(policy.decide_bid(obs, state))
    # Aggressive / Conservative vary with budget. Mimicking falls back
    # to Conservative when no history. In every case we expect at least
    # 2 distinct bid values across the 9 inputs above.
    assert len(bids) >= 2, f"Policy output did not vary with input: {bids}"


# ─── Budget constraint ──────────────────────────────────────────────


@pytest.mark.parametrize(
    "PolicyCls",
    [AggressiveOperator, ConservativeOperator, MimickingOperator],
)
def test_bid_never_exceeds_budget(PolicyCls):
    policy = PolicyCls()
    for budget in (0.0, 1.0, 10.0, 50.0, 1000.0):
        for round_idx in range(6):
            obs = make_obs(
                remaining_budget=budget,
                round_index=round_idx,
                competitor_bid_history=[[3.0, 4.0]],
            )
            state = make_state(budget=budget)
            bid = policy.decide_bid(obs, state)
            assert bid <= budget + 1e-9, (
                f"{PolicyCls.__name__} bid {bid} > budget {budget}"
            )
            assert bid >= 0.0


def test_zero_budget_yields_zero_bid():
    for PolicyCls in (AggressiveOperator, ConservativeOperator, MimickingOperator):
        policy = PolicyCls()
        obs = make_obs(remaining_budget=0.0)
        state = make_state(budget=0.0)
        assert policy.decide_bid(obs, state) == 0.0


# ─── Behavior spot-checks (archetype signatures) ────────────────────


def test_aggressive_dispute_mostly_escalate():
    policy = AggressiveOperator()
    escalate_count = 0
    trials = 50
    for i in range(trials):
        obs = make_obs(round_index=i % 6, reputation_score=(i % 10) / 10.0)
        state = make_state(budget=50.0, reputation=(i % 10) / 10.0)
        if policy.decide_dispute_response(obs, state) == DisputeChoice.ESCALATE:
            escalate_count += 1
    # Expected 80% → at least 60% observed.
    assert escalate_count / trials > 0.6


def test_conservative_never_escalates():
    policy = ConservativeOperator()
    for i in range(20):
        obs = make_obs(round_index=i % 6, reputation_score=(i % 10) / 10.0)
        state = make_state(reputation=(i % 10) / 10.0)
        choice = policy.decide_dispute_response(obs, state)
        assert choice in (DisputeChoice.NEGOTIATE, DisputeChoice.CONCEDE)


def test_aggressive_defects_when_reputation_above_0_3():
    policy = AggressiveOperator()
    obs = make_obs()
    assert policy.decide_cooperation(obs, make_state(reputation=0.5)) is False
    assert policy.decide_cooperation(obs, make_state(reputation=0.2)) is True


def test_conservative_cooperates_when_reputation_above_0_4():
    policy = ConservativeOperator()
    obs = make_obs()
    assert policy.decide_cooperation(obs, make_state(reputation=0.5)) is True
    assert policy.decide_cooperation(obs, make_state(reputation=0.3)) is False


def test_mimicking_mirrors_prior_cooperation():
    policy = MimickingOperator()
    # The oversight log shows a DEFECT from someone else → mirror it.
    obs = make_obs(
        oversight_events=[
            {
                "event_type": "reputation_update",
                "operator_id": "other",
                "cooperation_flag": CooperationChoice.DEFECT.value,
                "step_number": 0,
            }
        ]
    )
    assert policy.decide_cooperation(obs, make_state(operator_id="op-mimic")) is False

    obs2 = make_obs(
        oversight_events=[
            {
                "event_type": "reputation_update",
                "operator_id": "other",
                "cooperation_flag": CooperationChoice.COOPERATE.value,
                "step_number": 0,
            }
        ]
    )
    assert policy.decide_cooperation(obs2, make_state(operator_id="op-mimic")) is True


def test_mimicking_bid_is_running_average_when_history_exists():
    policy = MimickingOperator()
    obs = make_obs(competitor_bid_history=[[10.0, 20.0], [30.0]])
    state = make_state(budget=100.0)
    bid = policy.decide_bid(obs, state)
    assert abs(bid - 20.0) < 1e-6, f"expected avg=20.0, got {bid}"


# ─── Rotation helper ────────────────────────────────────────────────


def test_rotate_policies_deterministic_per_seed():
    for seed in (0, 1, 42, 199, 250):
        a = [type(p).__name__ for p in rotate_policies(seed, 2)]
        b = [type(p).__name__ for p in rotate_policies(seed, 2)]
        assert a == b


def test_rotate_policies_varies_across_seeds():
    orderings = set()
    for seed in range(20):
        orderings.add(tuple(type(p).__name__ for p in rotate_policies(seed, 2)))
    assert len(orderings) > 1, f"rotate_policies produced only {orderings}"


def test_rotate_policies_returns_requested_size():
    assert len(rotate_policies(0, 1)) == 1
    assert len(rotate_policies(0, 2)) == 2
    assert len(rotate_policies(0, 3)) == 3
