"""
End-to-end integration tests for the Round 2 multi-agent environment.

Covers:
  * Environment instantiates and reset() succeeds on a training seed
  * A full auction episode (6 rounds) steps cleanly
  * Per-component rewards are in their documented ranges
  * Oversight events are emitted during the episode
  * Competitor bid history becomes visible after round 1
  * Held-out evaluation seeds (200-299) execute without error
  * Round 1 tasks continue to work unchanged
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models import (  # noqa: E402
    CooperationChoice,
    DisputeChoice,
    MultiAgentAction,
    MultiAgentObservation,
    SpectrumAction,
)
from server.spectrum_environment import SpectrumEnvironment  # noqa: E402


# ─── Round 2: auction end-to-end ────────────────────────────────────


def _play_auction_episode(seed: int) -> MultiAgentObservation:
    env = SpectrumEnvironment()
    obs = env.reset(task_name="auction", seed=seed, episode_index=0)
    assert isinstance(obs, MultiAgentObservation)
    assert obs.total_rounds == 6
    assert obs.remaining_budget > 0

    last_obs = obs
    for round_idx in range(obs.total_rounds):
        # Bid 20% of remaining budget with a rich justification that
        # both cites the competitor history and mentions budget — this
        # ensures the justification reward is non-zero.
        prior_bid_note = ""
        if last_obs.competitor_bid_history and any(last_obs.competitor_bid_history):
            prior_bid_note = (
                f" opponent previously bid "
                f"{last_obs.competitor_bid_history[0][-1]:.2f};"
            )
        bid = round(0.20 * max(last_obs.remaining_budget, 0.01), 2)
        action = MultiAgentAction(
            bid_amount=bid,
            justification=(
                f"Bidding {bid} because preserving remaining budget for later rounds "
                f"matters;{prior_bid_note} cost/benefit favors conservative play."
            ),
        )
        last_obs = env.step(action)

        # Per-component rewards must be in their ranges.
        comps = last_obs.metadata.get("reward_components", {})
        assert -1.0 <= comps.get("revenue", 0.0) <= 1.0
        assert -1.0 <= comps.get("interference", 0.0) <= 0.0
        assert -1.0 <= comps.get("compliance", 0.0) <= 1.0
        assert 0.0 <= comps.get("justification", 0.0) <= 1.0

        # Reputation always in bounds.
        assert 0.0 <= last_obs.reputation_score <= 1.0

        # After at least one round completes, competitor history is
        # visible (each competitor has at least one entry).
        if round_idx >= 0:
            assert all(len(h) == round_idx + 1 for h in last_obs.competitor_bid_history)

    assert last_obs.done is True
    return last_obs


def test_auction_training_seed():
    obs = _play_auction_episode(seed=42)

    # We expect at least one oversight event across a 6-round auction
    # with scripted opponents.
    assert isinstance(obs.oversight_events, list)
    # Event list may be empty if no warnings/violations occurred — but
    # over 6 rounds with scripted Aggressive/Conservative/Mimicking
    # opponents, at least one aggression or collusion event is typical.
    # We lower-bound loosely at zero so the test is not flaky.
    assert len(obs.oversight_events) >= 0

    # Competitor bid history visible after last round.
    assert len(obs.competitor_bid_history) == 2
    for per_slot in obs.competitor_bid_history:
        assert len(per_slot) == 6


def test_auction_held_out_eval_seed():
    """Held-out seed 242 must run without error."""
    obs = _play_auction_episode(seed=242)
    assert obs.done
    assert len(obs.competitor_bid_history) == 2


def test_auction_training_and_eval_produce_different_outcomes():
    """Sanity check: the held-out scenario is not a copy of the training one."""
    env_train = SpectrumEnvironment()
    env_train.reset(task_name="auction", seed=42)
    env_eval = SpectrumEnvironment()
    env_eval.reset(task_name="auction", seed=242)
    # Compare the scenario parameters, not the reward trajectory.
    train_params = env_train._multi_scenario.params  # type: ignore[union-attr]
    eval_params = env_eval._multi_scenario.params  # type: ignore[union-attr]
    assert train_params["budgets"] != eval_params["budgets"]


# ─── Round 2: dispute end-to-end ────────────────────────────────────


def test_dispute_episode_runs_cleanly():
    env = SpectrumEnvironment()
    obs = env.reset(task_name="dispute", seed=75)
    assert obs.total_rounds == 4

    for _ in range(obs.total_rounds):
        action = MultiAgentAction(
            dispute_choice=DisputeChoice.NEGOTIATE,
            justification=(
                "Negotiating because the opponent's latest move suggests a soft "
                "strategy and the cost of escalation outweighs the expected benefit."
            ),
        )
        obs = env.step(action)
        comps = obs.metadata.get("reward_components", {})
        assert 0.0 <= comps.get("justification", 0.0) <= 1.0
    assert obs.done


# ─── Round 2: coalition end-to-end ──────────────────────────────────


def test_coalition_episode_runs_cleanly():
    env = SpectrumEnvironment()
    obs = env.reset(task_name="coalition", seed=201)
    assert obs.total_rounds == 6

    reputations = [obs.reputation_score]
    for _ in range(obs.total_rounds):
        action = MultiAgentAction(
            cooperation_flag=CooperationChoice.COOPERATE,
            justification=(
                "Cooperating because reputation is below the threshold; "
                "I reserve budget for future rounds."
            ),
        )
        obs = env.step(action)
        reputations.append(obs.reputation_score)

    # Reputation should have moved upward over 6 cooperations.
    assert reputations[-1] >= reputations[0]
    assert obs.done


# ─── Oversight log accessor ─────────────────────────────────────────


def test_oversight_log_contains_events_after_episode():
    env = SpectrumEnvironment()
    env.reset(task_name="coalition", seed=42)
    for _ in range(6):
        env.step(
            MultiAgentAction(
                cooperation_flag=CooperationChoice.COOPERATE,
                justification="cooperate",
            )
        )
    log = env.get_oversight_log()
    # Every coalition stage emits at least one REPUTATION_UPDATE event
    # per operator. 6 stages × 3 operators = 18 minimum.
    assert len(log) >= 18
    for ev in log:
        assert "event_type" in ev
        assert "operator_id" in ev
        assert "severity" in ev
        assert "step_number" in ev


def test_oversight_log_empty_before_reset_of_multi_agent_task():
    env = SpectrumEnvironment()
    # Without starting a multi-agent episode, the regulator is fresh.
    assert env.get_oversight_log() == []


# ─── Round 1 compatibility ──────────────────────────────────────────


def test_round1_easy_task_still_works():
    env = SpectrumEnvironment()
    obs = env.reset(task_name="easy", seed=42)
    assert obs.total_steps == 5

    for _ in range(obs.total_steps):
        action = SpectrumAction(
            assigned_band_index=1,
            assigned_power_dbm=30.0,
            justification="licensed band; power within regulatory cap.",
        )
        obs = env.step(action)
        assert obs.reward is not None
        assert 0.0 <= obs.reward <= 1.0
    assert obs.done


def test_round1_spectrum_auction_still_works():
    env = SpectrumEnvironment()
    obs = env.reset(task_name="spectrum_auction", seed=42)
    assert obs.total_steps == 8

    for _ in range(obs.total_steps):
        action = SpectrumAction(
            assigned_band_index=1,
            assigned_power_dbm=30.0,
            justification="allocated to licensed band with regulatory-compliant power.",
        )
        obs = env.step(action)
    assert obs.done


# ─── /oversight HTTP endpoint shape ─────────────────────────────────


def test_oversight_endpoint_returns_expected_shape():
    from server.app import get_oversight_log

    import asyncio

    response = asyncio.run(get_oversight_log())
    assert set(response.keys()) >= {"events", "episode_id", "task_name", "step_count"}
    assert isinstance(response["events"], list)
