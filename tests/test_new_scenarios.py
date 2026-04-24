"""
Unit tests for Round 2 multi-agent scenario generators and Pydantic models.

Scope (Prompt 1):
  * Determinism:    same seed → same scenario
  * Diversity:      different seeds → different scenarios
  * Ground-truth well-formedness
  * Train / eval seed range disjointness
  * Round-trip serialization of every new Pydantic class
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict

import pytest

# Ensure repo root is importable when tests are run from the tests/ dir.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models import (  # noqa: E402
    CooperationChoice,
    DisputeChoice,
    MultiAgentAction,
    MultiAgentObservation,
    OperatorState,
    OversightEvent,
    OversightEventType,
)
from scenarios import (  # noqa: E402
    _coalition_reset,
    _rotate_archetypes,
    generate_auction_scenario,
    generate_coalition_scenario,
    generate_dispute_scenario,
)


TRAIN_SEEDS = list(range(0, 200))
EVAL_SEEDS = list(range(200, 300))


# ─── Helpers ──────────────────────────────────────────────────────────


def _scenario_signature(scenario) -> tuple:
    """Canonical hashable fingerprint for a scenario (excluding seed)."""
    # We compare params + archetypes + description. Seed is excluded so
    # that "same scenario different seed" would be flagged; but we also
    # exclude it naturally since any scenario computed from a seed
    # inherits uniqueness from its params.
    params_items = tuple(sorted(
        (k, repr(v)) for k, v in scenario.params.items()
    ))
    return (
        scenario.task_name,
        scenario.num_rounds,
        params_items,
        tuple(scenario.opponent_archetypes),
        scenario.description,
    )


# ─── Determinism ──────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "generator",
    [generate_auction_scenario, generate_dispute_scenario],
)
def test_determinism_same_seed_same_scenario(generator):
    for seed in (0, 1, 42, 199, 250):
        s1, g1 = generator(seed)
        s2, g2 = generator(seed)
        assert _scenario_signature(s1) == _scenario_signature(s2)
        assert g1.optimal_actions == g2.optimal_actions
        assert g1.notes == g2.notes


def test_coalition_determinism_same_seed():
    # Coalition is stateful, so we explicitly reset the cache.
    for seed in (0, 1, 42, 199, 250):
        _coalition_reset(seed)
        s1, g1 = generate_coalition_scenario(seed)
        _coalition_reset(seed)
        s2, g2 = generate_coalition_scenario(seed)
        assert _scenario_signature(s1) == _scenario_signature(s2)
        assert g1.optimal_actions == g2.optimal_actions


# ─── Diversity ────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "generator",
    [generate_auction_scenario, generate_dispute_scenario],
)
def test_diversity_different_seeds_differ(generator):
    # Two arbitrary seed pairs should differ in at least one of:
    # scenario params, archetypes, or ground truth.
    sigs = set()
    for seed in range(20):
        s, _ = generator(seed)
        sigs.add(_scenario_signature(s))
    # We should see strictly more than one distinct scenario across 20 seeds.
    assert len(sigs) > 1, f"Generator produced only {len(sigs)} distinct scenarios across 20 seeds"


def test_coalition_diversity():
    sigs = set()
    for seed in range(20):
        _coalition_reset(seed)
        s, _ = generate_coalition_scenario(seed)
        sigs.add(_scenario_signature(s))
    assert len(sigs) > 1


# ─── Ground-truth well-formedness ─────────────────────────────────────


def test_auction_ground_truth_well_formed():
    for seed in (0, 5, 123, 250):
        s, g = generate_auction_scenario(seed)
        assert len(g.optimal_actions) == s.num_rounds == 4
        for entry in g.optimal_actions:
            assert "bid_amount" in entry
            assert "round_index" in entry
            assert isinstance(entry["bid_amount"], float)
            assert entry["bid_amount"] >= 0.0
            assert entry["bid_amount"] <= s.params["learner_budget"] + 1e-6
        assert g.notes.get("method", "").startswith("symmetric_bne")
        assert g.notes.get("approximation") is True


def test_dispute_ground_truth_well_formed():
    allowed = {"concede", "negotiate", "escalate", "audit"}
    for seed in (0, 5, 42, 250):
        s, g = generate_dispute_scenario(seed)
        assert len(g.optimal_actions) == 1
        entry = g.optimal_actions[0]
        assert entry["dispute_choice"] in allowed
        assert entry["round_index"] == 0
        # payoff for the best action must be >= all others under the stated prior
        payoffs = s.params["expected_payoffs_per_action"]
        best_score = payoffs[entry["dispute_choice"]]
        for a, score in payoffs.items():
            assert best_score >= score - 1e-9


def test_coalition_ground_truth_well_formed():
    for seed in (0, 5, 42, 250):
        _coalition_reset(seed)
        s, g = generate_coalition_scenario(seed)
        assert len(g.optimal_actions) == 1
        entry = g.optimal_actions[0]
        assert entry["cooperation_flag"] in {"cooperate", "defect", "abstain"}
        # Under the simplification, when reputation < 0.7 the GT is
        # strictly "cooperate"; otherwise "cooperate" or "defect" are acceptable.
        rep = s.params["learner_reputation"]
        if rep < 0.7:
            assert entry["cooperation_flag"] == "cooperate"
        assert "acceptable_actions" in g.notes


# ─── Train / eval seed ranges ─────────────────────────────────────────


def test_train_and_eval_seed_ranges_are_disjoint_sets():
    assert set(TRAIN_SEEDS).isdisjoint(set(EVAL_SEEDS))
    assert len(TRAIN_SEEDS) == 200
    assert len(EVAL_SEEDS) == 100


@pytest.mark.parametrize(
    "generator",
    [generate_auction_scenario, generate_dispute_scenario],
)
def test_train_and_eval_scenarios_disjoint(generator):
    train_sigs = {_scenario_signature(generator(s)[0]) for s in TRAIN_SEEDS}
    eval_sigs = {_scenario_signature(generator(s)[0]) for s in EVAL_SEEDS}
    overlap = train_sigs & eval_sigs
    assert not overlap, (
        f"Data leak: {len(overlap)} scenarios present in both training "
        f"and evaluation seed ranges."
    )


def test_train_and_eval_coalition_disjoint():
    train_sigs = set()
    for s in TRAIN_SEEDS:
        _coalition_reset(s)
        train_sigs.add(_scenario_signature(generate_coalition_scenario(s)[0]))
    eval_sigs = set()
    for s in EVAL_SEEDS:
        _coalition_reset(s)
        eval_sigs.add(_scenario_signature(generate_coalition_scenario(s)[0]))
    overlap = train_sigs & eval_sigs
    assert not overlap


# ─── Rotation properties ──────────────────────────────────────────────


def test_rotation_is_deterministic_and_sized_correctly():
    for seed in (0, 1, 42, 199, 250):
        r1 = _rotate_archetypes(seed, 2)
        r2 = _rotate_archetypes(seed, 2)
        assert r1 == r2
        assert len(r1) == 2
        for label in r1:
            assert label in {"Aggressive", "Conservative", "Mimicking"}


def test_rotation_spreads_archetypes_across_slots():
    # Across training seeds, each archetype should appear in slot 0 with
    # non-trivial frequency (> 10% of the time). This guards against a
    # degenerate rotation function that always puts the same archetype
    # in slot 0.
    slot0_counts: Dict[str, int] = {"Aggressive": 0, "Conservative": 0, "Mimicking": 0}
    for seed in TRAIN_SEEDS:
        r = _rotate_archetypes(seed, 2)
        slot0_counts[r[0]] += 1
    for label, count in slot0_counts.items():
        assert count / len(TRAIN_SEEDS) > 0.1, (
            f"Archetype {label} appears in slot 0 only {count}/200 times"
        )


# ─── Round-trip serialization of new Pydantic classes ─────────────────


def _roundtrip(cls, instance):
    """model_dump → model_validate → equality."""
    dumped = instance.model_dump()
    restored = cls.model_validate(dumped)
    assert restored.model_dump() == dumped
    return restored


def test_multi_agent_action_roundtrip_minimal():
    a = MultiAgentAction()
    _roundtrip(MultiAgentAction, a)


def test_multi_agent_action_roundtrip_auction_variant():
    a = MultiAgentAction(bid_amount=12.5, justification="high valuation this round")
    r = _roundtrip(MultiAgentAction, a)
    assert r.bid_amount == 12.5


def test_multi_agent_action_roundtrip_dispute_variant():
    a = MultiAgentAction(dispute_choice=DisputeChoice.NEGOTIATE)
    r = _roundtrip(MultiAgentAction, a)
    assert r.dispute_choice == DisputeChoice.NEGOTIATE


def test_multi_agent_action_roundtrip_coalition_variant():
    a = MultiAgentAction(cooperation_flag=CooperationChoice.COOPERATE)
    r = _roundtrip(MultiAgentAction, a)
    assert r.cooperation_flag == CooperationChoice.COOPERATE


def test_multi_agent_observation_roundtrip():
    o = MultiAgentObservation(
        competitor_bid_history=[[1.0, 2.0], [3.0]],
        reputation_score=0.75,
        oversight_events=[{"event_type": "warning", "operator_id": "op-1"}],
        remaining_budget=42.0,
        opponent_slot_indices=[0, 1],
        round_index=3,
        total_rounds=6,
    )
    r = _roundtrip(MultiAgentObservation, o)
    assert r.reputation_score == 0.75
    assert r.remaining_budget == 42.0
    assert r.round_index == 3


def test_operator_state_roundtrip():
    s = OperatorState(
        operator_id="op-0",
        budget=100.0,
        licenses_held=[1, 3, 5],
        reputation=0.4,
        action_history=[{"bid_amount": 5.0}],
    )
    r = _roundtrip(OperatorState, s)
    assert r.budget == 100.0
    assert r.licenses_held == [1, 3, 5]


def test_oversight_event_roundtrip():
    e = OversightEvent(
        event_type=OversightEventType.VIOLATION,
        operator_id="op-2",
        severity=0.4,
        explanation="bid exceeded declared budget",
        step_number=7,
    )
    r = _roundtrip(OversightEvent, e)
    assert r.event_type == OversightEventType.VIOLATION
    assert r.severity == 0.4
    assert r.step_number == 7


# ─── Field bound sanity on new classes ────────────────────────────────


def test_reputation_score_is_clamped_by_validator():
    with pytest.raises(Exception):
        MultiAgentObservation(reputation_score=1.5)
    with pytest.raises(Exception):
        MultiAgentObservation(reputation_score=-0.1)


def test_bid_amount_rejects_negative():
    with pytest.raises(Exception):
        MultiAgentAction(bid_amount=-1.0)
