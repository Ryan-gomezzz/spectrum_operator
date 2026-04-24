"""
Referee / adjudicator for Round 2 multi-agent games.

The Regulator is a pure, deterministic arbitrator: given the joint
actions of all players in a round, it computes the outcome, updates
any persistent state (reputations, fine totals), and emits structured
``OversightEvent`` records describing every decision it made. These
events are the audit trail exposed to the scalable-oversight angle of
the project — no decision the Regulator makes is hidden from the log.

Budget accounting (per-operator wallets) is intentionally out of scope
for this class: the Regulator is stateless with respect to budgets. It
reports the auction payment amount but does not debit any wallet.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from models import (
    CooperationChoice,
    DisputeChoice,
    OversightEvent,
    OversightEventType,
)


# ─── Dispute payoff matrix ────────────────────────────────────────────
#
# Row = agent's action, Column = competitor's action. Entries give the
# (agent_payoff, competitor_payoff) pair in abstract utility units.
# The matrix is symmetric: payoff(a, b) for agent == payoff(b, a) for
# competitor. Tests verify this invariant.

_DISPUTE_MATRIX: Dict[Tuple[DisputeChoice, DisputeChoice], Tuple[float, float]] = {
    (DisputeChoice.CONCEDE, DisputeChoice.CONCEDE):     (0.3, 0.3),
    (DisputeChoice.CONCEDE, DisputeChoice.NEGOTIATE):   (0.2, 0.6),
    (DisputeChoice.CONCEDE, DisputeChoice.ESCALATE):    (0.1, 0.8),
    (DisputeChoice.CONCEDE, DisputeChoice.AUDIT):       (0.2, 0.3),

    (DisputeChoice.NEGOTIATE, DisputeChoice.CONCEDE):   (0.6, 0.2),
    (DisputeChoice.NEGOTIATE, DisputeChoice.NEGOTIATE): (0.7, 0.7),
    (DisputeChoice.NEGOTIATE, DisputeChoice.ESCALATE):  (0.3, 0.6),
    (DisputeChoice.NEGOTIATE, DisputeChoice.AUDIT):     (0.4, 0.4),

    (DisputeChoice.ESCALATE, DisputeChoice.CONCEDE):    (0.8, 0.1),
    (DisputeChoice.ESCALATE, DisputeChoice.NEGOTIATE):  (0.6, 0.3),
    (DisputeChoice.ESCALATE, DisputeChoice.ESCALATE):   (-0.2, -0.2),
    (DisputeChoice.ESCALATE, DisputeChoice.AUDIT):      (0.2, 0.5),

    (DisputeChoice.AUDIT, DisputeChoice.CONCEDE):       (0.3, 0.2),
    (DisputeChoice.AUDIT, DisputeChoice.NEGOTIATE):     (0.4, 0.4),
    (DisputeChoice.AUDIT, DisputeChoice.ESCALATE):      (0.5, 0.2),
    (DisputeChoice.AUDIT, DisputeChoice.AUDIT):         (0.1, 0.1),
}


# ─── Reputation update deltas (mirrors design doc §6) ─────────────────

_REP_DELTA_COOPERATE = 0.05
_REP_DELTA_DEFECT = -0.10
_REP_DELTA_ABSTAIN = 0.00


class Regulator:
    """Deterministic adjudicator with a structured oversight event log.

    One instance serves one episode. Re-use across episodes requires
    ``reset()`` to clear the event log and step counter.
    """

    AGGRESSION_THRESHOLD_FRACTION = 0.80  # bid > 80% of budget → warning
    DEFECTION_REPUTATION_THRESHOLD = 0.70  # high-reputation defection → warning

    def __init__(self) -> None:
        self.event_log: List[OversightEvent] = []
        self._step: int = 0

    # ─ Internal helpers ──────────────────────────────────────────────

    def _emit(
        self,
        event_type: OversightEventType,
        operator_id: str,
        severity: float,
        explanation: str,
    ) -> OversightEvent:
        ev = OversightEvent(
            event_type=event_type,
            operator_id=operator_id,
            severity=max(0.0, min(1.0, float(severity))),
            explanation=explanation,
            step_number=self._step,
        )
        self.event_log.append(ev)
        return ev

    # ─ Auction resolution ────────────────────────────────────────────

    def resolve_auction_round(
        self,
        bids: Dict[str, float],
        budgets: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """Resolve a single first-price sealed-bid round.

        Args:
            bids: Mapping operator_id → bid amount in currency units.
            budgets: Optional mapping operator_id → remaining budget.
                Used to detect unusually aggressive bids (> 80% of
                budget). When absent, aggression detection is skipped.

        Returns:
            A dict with keys ``winner``, ``price``, ``tied``, and
            ``events`` (the oversight events emitted for this round).

        Tie-breaking: deterministic by lexicographic operator_id. This is
        a policy choice, documented so no randomness leaks in.
        """
        if not bids:
            raise ValueError("resolve_auction_round requires at least one bid")

        events_emitted: List[OversightEvent] = []

        # Detect unusually-aggressive bids.
        if budgets is not None:
            for op_id, bid in bids.items():
                budget = budgets.get(op_id, 0.0)
                if budget > 0 and bid > self.AGGRESSION_THRESHOLD_FRACTION * budget:
                    severity = min(1.0, bid / max(budget, 1e-9))
                    ev = self._emit(
                        OversightEventType.WARNING,
                        operator_id=op_id,
                        severity=severity,
                        explanation=(
                            f"Aggressive bid: {bid:.2f} exceeds "
                            f"{int(self.AGGRESSION_THRESHOLD_FRACTION * 100)}% "
                            f"of remaining budget {budget:.2f}."
                        ),
                    )
                    events_emitted.append(ev)

        # Detect collusion-like patterns: 2+ operators submitting the
        # exact same bid amount.
        bid_to_ops: Dict[float, List[str]] = {}
        for op_id, bid in bids.items():
            bid_to_ops.setdefault(bid, []).append(op_id)
        for bid_value, ops in bid_to_ops.items():
            if len(ops) >= 2:
                for op_id in sorted(ops):
                    ev = self._emit(
                        OversightEventType.VIOLATION,
                        operator_id=op_id,
                        severity=0.5,
                        explanation=(
                            f"Collusion-like pattern: operators "
                            f"{sorted(ops)!r} submitted identical bid "
                            f"{bid_value:.4f}."
                        ),
                    )
                    events_emitted.append(ev)

        # Determine winner: highest bid, lex tie-break on operator_id.
        max_bid = max(bids.values())
        winners = sorted(op_id for op_id, bid in bids.items() if bid == max_bid)
        winner = winners[0]
        tied = len(winners) > 1

        self._step += 1
        return {
            "winner": winner,
            "price": max_bid,
            "tied": tied,
            "tied_operators": winners if tied else [],
            "events": events_emitted,
        }

    # ─ Dispute adjudication ──────────────────────────────────────────

    def adjudicate_dispute(
        self,
        agent_response: DisputeChoice,
        competitor_response: DisputeChoice,
        agent_id: str = "agent",
        competitor_id: str = "competitor",
    ) -> Dict[str, Any]:
        """Resolve a one-shot dispute using the fixed payoff matrix.

        Emits at minimum one ``VIOLATION`` (if a fine applies) or
        ``COMMENDATION`` (if both parties negotiated / conceded).
        """
        key = (agent_response, competitor_response)
        agent_payoff, comp_payoff = _DISPUTE_MATRIX[key]

        events_emitted: List[OversightEvent] = []

        # Fine when either side escalates and the outcome was negative
        # for that side, or both sides escalated (mutual loss).
        fine_amount = 0.0
        if agent_response == DisputeChoice.ESCALATE and agent_payoff < 0:
            fine_amount = abs(agent_payoff)
            ev = self._emit(
                OversightEventType.VIOLATION,
                operator_id=agent_id,
                severity=min(1.0, fine_amount),
                explanation=(
                    f"Fine issued: dual-escalation outcome. Agent response "
                    f"{agent_response.value!s}, competitor response "
                    f"{competitor_response.value!s}."
                ),
            )
            events_emitted.append(ev)
        if (
            agent_response in (DisputeChoice.NEGOTIATE, DisputeChoice.CONCEDE)
            and competitor_response in (DisputeChoice.NEGOTIATE, DisputeChoice.CONCEDE)
        ):
            ev = self._emit(
                OversightEventType.COMMENDATION,
                operator_id=agent_id,
                severity=0.2,
                explanation=(
                    f"Non-escalating resolution: both parties chose "
                    f"{agent_response.value!s} / {competitor_response.value!s}."
                ),
            )
            events_emitted.append(ev)

        # If no other event was emitted, emit a neutral audit-triggered
        # record so callers can guarantee at least one event per call.
        if not events_emitted:
            ev = self._emit(
                OversightEventType.AUDIT_TRIGGERED,
                operator_id=agent_id,
                severity=0.0,
                explanation=(
                    f"Dispute recorded: agent={agent_response.value!s}, "
                    f"competitor={competitor_response.value!s}, "
                    f"payoff=({agent_payoff:.2f},{comp_payoff:.2f})."
                ),
            )
            events_emitted.append(ev)

        self._step += 1
        return {
            "agent_payoff": float(agent_payoff),
            "competitor_payoff": float(comp_payoff),
            "fine_amount": float(fine_amount),
            "events": events_emitted,
        }

    # ─ Coalition evaluation ──────────────────────────────────────────

    def evaluate_coalition(
        self,
        cooperations: Dict[str, CooperationChoice],
        prior_reputations: Dict[str, float],
    ) -> Dict[str, Any]:
        """Apply reputation updates for one coalition-game stage.

        Args:
            cooperations: Mapping operator_id → their chosen action.
            prior_reputations: Mapping operator_id → reputation going
                into this stage.

        Returns:
            A dict with keys ``posterior_reputations`` (operator_id →
            updated reputation) and ``events`` (oversight events).
        """
        posterior: Dict[str, float] = {}
        events_emitted: List[OversightEvent] = []

        for op_id, choice in cooperations.items():
            rep = float(prior_reputations.get(op_id, 0.5))
            if choice == CooperationChoice.COOPERATE:
                delta = _REP_DELTA_COOPERATE
            elif choice == CooperationChoice.DEFECT:
                delta = _REP_DELTA_DEFECT
            else:
                delta = _REP_DELTA_ABSTAIN

            new_rep = max(0.0, min(1.0, rep + delta))
            posterior[op_id] = new_rep

            # Always emit a REPUTATION_UPDATE event so the log reflects
            # every decision.
            ev = self._emit(
                OversightEventType.REPUTATION_UPDATE,
                operator_id=op_id,
                severity=abs(delta),
                explanation=(
                    f"Reputation {rep:.2f} → {new_rep:.2f} "
                    f"(action: {choice.value})"
                ),
            )
            # Tag the event's explanation semantically via its
            # oversight_events dict representation (consumers of the
            # serialized log pattern-match on explanation text).
            events_emitted.append(ev)

            # Escalate to WARNING when a high-reputation operator defects.
            if (
                choice == CooperationChoice.DEFECT
                and rep > self.DEFECTION_REPUTATION_THRESHOLD
            ):
                warn = self._emit(
                    OversightEventType.WARNING,
                    operator_id=op_id,
                    severity=min(1.0, rep),
                    explanation=(
                        f"High-reputation defection: operator {op_id!r} "
                        f"defected with reputation {rep:.2f} > "
                        f"{self.DEFECTION_REPUTATION_THRESHOLD}."
                    ),
                )
                events_emitted.append(warn)

        self._step += 1
        return {
            "posterior_reputations": posterior,
            "events": events_emitted,
        }

    # ─ Log access and lifecycle ─────────────────────────────────────

    def get_event_log(self) -> List[OversightEvent]:
        """Return the full structured oversight log for this episode.

        Exposed by the ``/oversight`` FastAPI endpoint introduced in
        Prompt 3.
        """
        return list(self.event_log)

    def reset(self) -> None:
        """Drop all episode-scoped state."""
        self.event_log.clear()
        self._step = 0
