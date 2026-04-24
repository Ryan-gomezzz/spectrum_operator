"""
RF Spectrum Allocation Environment - Models
=============================================
Typed Pydantic models for actions, observations, and state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field
from openenv.core.env_server.types import Action, Observation, State  # noqa: F401


# ── Spectrum domain types ────────────────────────────────────────────


@dataclass
class FrequencyBand:
    """A contiguous block of radio spectrum."""

    start_mhz: float
    end_mhz: float
    label: str  # e.g. "Band-A (700 MHz LTE)"

    @property
    def bandwidth_mhz(self) -> float:
        return self.end_mhz - self.start_mhz


@dataclass
class Allocation:
    """An existing spectrum allocation occupying a band."""

    band_index: int  # index into the spectrum grid
    user_type: str  # "primary" | "secondary" | "emergency"
    user_id: str
    power_dbm: float
    remaining_steps: int  # how many steps until this allocation expires


@dataclass
class AllocationRequest:
    """An incoming request for spectrum."""

    request_id: str
    requester_type: str  # "emergency", "commercial", "iot", "amateur", "military"
    requester_id: str
    bandwidth_needed_mhz: float
    preferred_band_index: Optional[int]
    priority: int  # 1 (highest) .. 5 (lowest)
    duration_steps: int
    power_dbm: float
    description: str  # natural language description of the request


# ── OpenEnv models ───────────────────────────────────────────────────


class SpectrumAction(Action):
    """Agent's allocation decision."""

    assigned_band_index: int = Field(
        ..., description="Index of the frequency band to assign (0-based), or -1 to reject"
    )
    assigned_power_dbm: float = Field(
        ..., description="Transmit power in dBm to allocate"
    )
    justification: str = Field(
        default="", description="Brief justification for the decision"
    )


class SpectrumObservation(Observation):
    """What the agent sees each step."""

    # Current spectrum state
    spectrum_grid: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of bands with occupancy info",
    )
    active_allocations: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Currently active allocations",
    )

    # Incoming request
    current_request: Dict[str, Any] = Field(
        default_factory=dict,
        description="The allocation request to process",
    )

    # Regulatory constraints
    regulatory_rules: List[str] = Field(
        default_factory=list,
        description="Active regulatory constraints",
    )

    # Episode info
    task_difficulty: str = Field(
        default="easy", description="easy | medium | hard"
    )
    step_number: int = Field(default=0)
    total_steps: int = Field(default=0)
    spectral_efficiency: float = Field(
        default=0.0,
        description="Current spectrum utilization ratio 0.0-1.0",
    )
    episode_reward_so_far: float = Field(default=0.0)
    last_action_error: Optional[str] = Field(default=None)

    # Look-ahead field: populated only for the spectrum_auction task
    upcoming_requests: List[Dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "Preview of the next 1-2 upcoming requests (spectrum_auction task only). "
            "Each entry exposes requester_type, priority, bandwidth_needed_mhz, "
            "preferred_band_index, and description — no ground-truth fields revealed."
        ),
    )

    # Metadata inherited from Observation base
    # done: bool, reward: float, metadata: dict


class SpectrumState(State):
    """Internal episode state."""

    task_name: str = ""
    accumulated_reward: float = 0.0
    requests_processed: int = 0
    requests_total: int = 0
    successful_allocations: int = 0
    rejected_requests: int = 0
    interference_events: int = 0
    preemptions: int = 0


# ── Multi-agent (Round 2) extensions ─────────────────────────────────
#
# The classes below layer onto the Round 1 schema without modifying any
# existing class. All new optional fields are designed so that a Round 1
# payload still deserializes successfully into the Round 1 classes. Every
# class is documented in universal game-theoretic terms; domain flavor is
# restricted to per-scenario description strings constructed elsewhere.


class DisputeChoice(str, Enum):
    """Action set for a one-shot dispute-resolution game where the
    learner's best response depends on inferring the opponent's type
    from prior observable play."""

    CONCEDE = "concede"
    NEGOTIATE = "negotiate"
    ESCALATE = "escalate"
    AUDIT = "audit"


class CooperationChoice(str, Enum):
    """Action set for an iterated prisoner's-dilemma-style coalition
    game with a reputation-tracking referee."""

    COOPERATE = "cooperate"
    DEFECT = "defect"
    ABSTAIN = "abstain"


class OversightEventType(str, Enum):
    """Tagged event emitted by the reputation-tracking referee to
    signal normative judgments over player actions."""

    WARNING = "warning"
    VIOLATION = "violation"
    COMMENDATION = "commendation"
    AUDIT_TRIGGERED = "audit_triggered"
    REPUTATION_UPDATE = "reputation_update"


class MultiAgentAction(Action):
    """Action schema for games with two or more strategic players.

    Fields are mutually exclusive by scenario: an auction task uses
    `bid_amount`; a dispute task uses `dispute_choice`; a coalition task
    uses `cooperation_flag`. All fields are optional so a caller wiring
    a single-agent Round 1 episode can omit them entirely and still
    produce a valid instance.
    """

    bid_amount: Optional[float] = Field(
        default=None,
        ge=0.0,
        description=(
            "Sealed bid expressed in abstract currency units, bounded above "
            "by the agent's remaining budget. Used only in sealed-bid auction "
            "games; None for all other games."
        ),
    )
    dispute_choice: Optional[DisputeChoice] = Field(
        default=None,
        description=(
            "One element of the dispute-game action set. Used only in "
            "one-shot dispute-resolution games; None otherwise."
        ),
    )
    cooperation_flag: Optional[CooperationChoice] = Field(
        default=None,
        description=(
            "One element of the cooperation-game action set. Used only in "
            "iterated prisoner's-dilemma coalition games; None otherwise."
        ),
    )
    justification: str = Field(
        default="",
        max_length=500,
        description=(
            "Free-form reasoning string. Scored against process-bonus "
            "criteria in the justification reward bucket."
        ),
    )


class MultiAgentObservation(SpectrumObservation):
    """Observation schema for games with two or more strategic players.

    Extends `SpectrumObservation` with fields describing observable
    opponent play, the agent's own normative standing (reputation),
    referee-emitted oversight records, and a per-episode budget for
    resource-constrained games. No personality labels are leaked — the
    agent must infer opponent type from behavior alone.
    """

    competitor_bid_history: List[List[float]] = Field(
        default_factory=list,
        description=(
            "Per-opponent list of past bid amounts in currency units. "
            "Outer index aligns with `opponent_slot_indices`. Inner list "
            "is ordered oldest-first and is empty when the corresponding "
            "opponent has not yet played."
        ),
    )
    reputation_score: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description=(
            "The agent's own reputation in [0.0, 1.0] as maintained by "
            "the referee. See design doc §6 for update mechanics."
        ),
    )
    oversight_events: List[Dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "Serialized OversightEvent records emitted by the referee in "
            "prior steps of the current episode, ordered oldest-first."
        ),
    )
    remaining_budget: float = Field(
        default=0.0,
        ge=0.0,
        description=(
            "Currency units remaining in the current episode's budget. "
            "Populated for auction games; 0.0 for all other games."
        ),
    )
    opponent_slot_indices: List[int] = Field(
        default_factory=list,
        description=(
            "Stable integer IDs (0..N-1) for opponents in the current "
            "episode. No personality information attached."
        ),
    )
    round_index: int = Field(
        default=0,
        ge=0,
        description="Zero-indexed round within the current episode.",
    )
    total_rounds: int = Field(
        default=1,
        ge=1,
        description="Episode horizon (number of rounds).",
    )


class OperatorState(BaseModel):
    """Per-player bookkeeping state tracked across rounds of a game.

    This is a pure data container with no environment semantics. It
    describes one player's cumulative situation: available budget,
    resources currently held, normative standing, and the full log of
    actions taken so far.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    operator_id: str = Field(
        ..., description="Stable identifier for the player across episodes."
    )
    budget: float = Field(
        default=0.0,
        ge=0.0,
        description="Currency units currently available to spend.",
    )
    licenses_held: List[int] = Field(
        default_factory=list,
        description=(
            "Abstract resource identifiers currently owned by the player. "
            "Integers are opaque — interpreted only by the scenario logic."
        ),
    )
    reputation: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Normative standing in [0.0, 1.0].",
    )
    action_history: List[Dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "Ordered log of past action payloads (dumped `MultiAgentAction` "
            "dicts), oldest-first."
        ),
    )


class OversightEvent(BaseModel):
    """Structured record emitted by the reputation-tracking referee.

    Each event is a normative judgment attached to a specific player at
    a specific step. Events are surfaced to all players via the
    `oversight_events` observation field and are the primary mechanism
    by which the referee influences future play.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    event_type: OversightEventType = Field(
        ..., description="The kind of normative judgment being emitted."
    )
    operator_id: str = Field(
        ..., description="Identifier of the player this event pertains to."
    )
    severity: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description=(
            "Magnitude of the judgment in [0.0, 1.0]. Interpretation is "
            "event-type specific (e.g. size of reputation delta)."
        ),
    )
    explanation: str = Field(
        default="",
        max_length=500,
        description="Human-readable justification for the judgment.",
    )
    step_number: int = Field(
        ..., ge=0, description="Episode step at which the event was emitted."
    )
