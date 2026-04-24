"""
Tests for the Regulator (referee / adjudicator).

Verifies:
  * Auction: highest bidder wins, ties are broken deterministically
  * Dispute payoff matrix is symmetric for symmetric actions
  * Coalition: one cooperation shifts reputation by the documented delta
  * Every adjudication call emits at least one OversightEvent
  * Emitted events carry all required fields
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models import (  # noqa: E402
    CooperationChoice,
    DisputeChoice,
    OversightEvent,
    OversightEventType,
)
from agents.regulator import Regulator  # noqa: E402


# ─── Auction resolution ───────────────────────────────────────────────


def test_auction_highest_bidder_wins():
    r = Regulator()
    out = r.resolve_auction_round({"op-A": 5.0, "op-B": 12.0, "op-C": 9.0})
    assert out["winner"] == "op-B"
    assert out["price"] == 12.0
    assert out["tied"] is False


def test_auction_tie_breaking_is_deterministic_and_lexicographic():
    r = Regulator()
    # All bids equal → lex tie-break, "op-A" wins.
    bids = {"op-C": 7.0, "op-A": 7.0, "op-B": 7.0}
    out1 = r.resolve_auction_round(dict(bids))
    r2 = Regulator()
    out2 = r2.resolve_auction_round(dict(bids))
    assert out1["winner"] == "op-A"
    assert out1["winner"] == out2["winner"]
    assert out1["tied"] is True


def test_auction_aggression_warning_fires_above_80_percent():
    r = Regulator()
    out = r.resolve_auction_round(
        {"op-A": 90.0, "op-B": 40.0},
        budgets={"op-A": 100.0, "op-B": 100.0},
    )
    aggressive_events = [e for e in out["events"] if e.event_type == OversightEventType.WARNING]
    assert aggressive_events, "expected aggression warning for 90% bid"
    assert any(ev.operator_id == "op-A" for ev in aggressive_events)


def test_auction_collusion_detection_on_equal_bids():
    r = Regulator()
    out = r.resolve_auction_round({"op-A": 5.0, "op-B": 5.0})
    collusion = [e for e in out["events"] if e.event_type == OversightEventType.VIOLATION]
    assert len(collusion) == 2, "expected one collusion event per tied operator"


def test_auction_no_events_when_single_unique_bid_under_threshold():
    r = Regulator()
    out = r.resolve_auction_round(
        {"op-A": 10.0, "op-B": 20.0, "op-C": 30.0},
        budgets={"op-A": 100.0, "op-B": 100.0, "op-C": 100.0},
    )
    assert out["events"] == []


def test_auction_rejects_empty_bids():
    r = Regulator()
    with pytest.raises(ValueError):
        r.resolve_auction_round({})


# ─── Dispute adjudication ─────────────────────────────────────────────


def test_dispute_matrix_symmetric_for_same_action():
    """When both parties play the same action, their payoffs must equal."""
    r = Regulator()
    for action in DisputeChoice:
        out = r.adjudicate_dispute(action, action)
        assert out["agent_payoff"] == out["competitor_payoff"], (
            f"asymmetric payoff for ({action}, {action}): {out}"
        )


def test_dispute_swap_symmetry():
    """payoff(a, b) for the agent must equal payoff(b, a) for the competitor."""
    r = Regulator()
    for a in DisputeChoice:
        for b in DisputeChoice:
            out1 = r.adjudicate_dispute(a, b)
            out2 = r.adjudicate_dispute(b, a)
            assert out1["agent_payoff"] == out2["competitor_payoff"], (
                f"swap-asymmetry: agent({a},{b}) != competitor({b},{a})"
            )
            assert out1["competitor_payoff"] == out2["agent_payoff"]


def test_dispute_emits_at_least_one_event():
    r = Regulator()
    for a in DisputeChoice:
        for b in DisputeChoice:
            pre = len(r.event_log)
            r.adjudicate_dispute(a, b)
            post = len(r.event_log)
            assert post - pre >= 1, f"no event emitted for ({a}, {b})"


def test_dispute_dual_escalation_issues_fine():
    r = Regulator()
    out = r.adjudicate_dispute(DisputeChoice.ESCALATE, DisputeChoice.ESCALATE)
    assert out["fine_amount"] > 0
    assert any(e.event_type == OversightEventType.VIOLATION for e in out["events"])


def test_dispute_commendation_on_negotiated_resolution():
    r = Regulator()
    out = r.adjudicate_dispute(DisputeChoice.NEGOTIATE, DisputeChoice.NEGOTIATE)
    assert any(e.event_type == OversightEventType.COMMENDATION for e in out["events"])


# ─── Coalition evaluation ─────────────────────────────────────────────


def test_coalition_cooperation_raises_reputation_by_0_05():
    r = Regulator()
    out = r.evaluate_coalition(
        {"op-0": CooperationChoice.COOPERATE},
        {"op-0": 0.50},
    )
    assert abs(out["posterior_reputations"]["op-0"] - 0.55) < 1e-9


def test_coalition_defection_lowers_reputation_by_0_10():
    r = Regulator()
    out = r.evaluate_coalition(
        {"op-0": CooperationChoice.DEFECT},
        {"op-0": 0.50},
    )
    assert abs(out["posterior_reputations"]["op-0"] - 0.40) < 1e-9


def test_coalition_abstention_leaves_reputation_unchanged():
    r = Regulator()
    out = r.evaluate_coalition(
        {"op-0": CooperationChoice.ABSTAIN},
        {"op-0": 0.50},
    )
    assert out["posterior_reputations"]["op-0"] == 0.50


def test_coalition_reputation_clamps_at_bounds():
    r = Regulator()
    out_low = r.evaluate_coalition(
        {"op-0": CooperationChoice.DEFECT},
        {"op-0": 0.02},
    )
    assert out_low["posterior_reputations"]["op-0"] == 0.0

    r2 = Regulator()
    out_high = r2.evaluate_coalition(
        {"op-0": CooperationChoice.COOPERATE},
        {"op-0": 0.98},
    )
    assert out_high["posterior_reputations"]["op-0"] == 1.0


def test_coalition_emits_warning_for_high_reputation_defection():
    r = Regulator()
    out = r.evaluate_coalition(
        {"op-0": CooperationChoice.DEFECT},
        {"op-0": 0.80},
    )
    warns = [e for e in out["events"] if e.event_type == OversightEventType.WARNING]
    assert warns, "expected warning for high-reputation defection"
    assert any("High-reputation defection" in e.explanation for e in warns)


def test_coalition_emits_at_least_one_event_per_operator():
    r = Regulator()
    out = r.evaluate_coalition(
        {"op-0": CooperationChoice.COOPERATE, "op-1": CooperationChoice.ABSTAIN},
        {"op-0": 0.5, "op-1": 0.5},
    )
    ids = {e.operator_id for e in out["events"]}
    assert ids == {"op-0", "op-1"}


# ─── Event log structure ──────────────────────────────────────────────


def test_events_carry_all_required_fields():
    r = Regulator()
    r.adjudicate_dispute(DisputeChoice.ESCALATE, DisputeChoice.ESCALATE)
    r.evaluate_coalition({"x": CooperationChoice.COOPERATE}, {"x": 0.5})
    r.resolve_auction_round({"x": 10.0, "y": 10.0})
    assert r.get_event_log(), "expected a non-empty event log"
    for e in r.get_event_log():
        assert isinstance(e, OversightEvent)
        assert isinstance(e.event_type, OversightEventType)
        assert e.operator_id
        assert 0.0 <= e.severity <= 1.0
        assert e.step_number >= 0


def test_event_log_reset_clears_state():
    r = Regulator()
    r.adjudicate_dispute(DisputeChoice.CONCEDE, DisputeChoice.CONCEDE)
    assert r.get_event_log()
    r.reset()
    assert r.get_event_log() == []


def test_event_step_numbers_are_monotonic():
    r = Regulator()
    r.adjudicate_dispute(DisputeChoice.CONCEDE, DisputeChoice.CONCEDE)
    step0 = r.get_event_log()[-1].step_number
    # Use an auction call that guarantees at least one event (collusion).
    r.resolve_auction_round({"a": 5.0, "b": 5.0})
    step1 = r.get_event_log()[-1].step_number
    assert step1 > step0
