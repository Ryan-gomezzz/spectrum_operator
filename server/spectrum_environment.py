"""
RF Spectrum Allocation Environment - Core Logic
=================================================
Implements the OpenEnv Environment interface with step(), reset(), state().
"""

from __future__ import annotations

import os
import random
import signal
import sys
import time
import uuid
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple, Union

# Ensure project root is on path when running from server/ directory or Docker
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

from models import (
    CooperationChoice,
    DisputeChoice,
    MultiAgentAction,
    MultiAgentObservation,
    OperatorState,
    OversightEvent,
    OversightEventType,
    SpectrumAction,
    SpectrumObservation,
    SpectrumState,
)
from scenarios import (
    SPECTRUM_GRID,
    TASK_REGISTRY,
    GroundTruth,
    Scenario,
    ScenarioRequest,
    _coalition_reset,
    generate_auction_scenario,
    generate_coalition_scenario,
    generate_dispute_scenario,
    get_rules,
    get_scenarios,
    get_spectrum_grid,
)

from agents.operator_policies import (
    MimickingOperator,
    OperatorPolicy,
    rotate_policies,
)
from agents.regulator import Regulator
from rewards import (
    compute_total_reward,
    reward_compliance,
    reward_interference,
    reward_justification,
    reward_revenue,
)


# ── Multi-agent task registry (Round 2) ────────────────────────────

_MULTI_AGENT_TASKS = {"auction", "dispute", "coalition"}

_MULTI_AGENT_GENERATORS = {
    "auction": generate_auction_scenario,
    "dispute": generate_dispute_scenario,
    "coalition": generate_coalition_scenario,
}

_MULTI_AGENT_ROUNDS = {
    "auction": 6,
    "dispute": 4,   # 4 repeated one-shot disputes per episode
    "coalition": 6,
}


# Hard wall-clock bound per step() call. Exceeding this raises and the
# current round is marked done with an error observation. 30 s is well
# above any legitimate work we do (all policies are O(1) Python), so
# hitting this bound means something upstream has hung.
_STEP_TIMEOUT_SECONDS = 30.0


class SpectrumEnvironment(Environment):
    """
    RF Spectrum Allocation OpenEnv Environment.

    The agent manages a simulated radio spectrum, processing allocation requests
    and making assignment decisions while respecting regulatory constraints.
    """

    def __init__(self, task_name: str = "easy", episode_index: int = 0, seed: int | None = None):
        super().__init__()
        self._task_name = task_name
        self._episode_index = episode_index
        self._seed = seed
        self._state = SpectrumState(episode_id=str(uuid.uuid4()), step_count=0)

        # Will be populated on reset()
        self._scenarios: List[List[ScenarioRequest]] = []
        self._current_episode: List[ScenarioRequest] = []
        self._active_allocations: List[Dict[str, Any]] = []
        self._step_rewards: List[float] = []
        self._rules: List[str] = []

        # ── Round 2 multi-agent bookkeeping ─────────────────────────
        self._regulator: Regulator = Regulator()
        self._competitor_policies: List[OperatorPolicy] = []
        self._operator_states: Dict[str, OperatorState] = {}
        self._multi_scenario: Optional[Scenario] = None
        self._multi_ground_truth: Optional[GroundTruth] = None
        self._multi_round_rewards: List[Dict[str, float]] = []
        # Per-slot competitor bid history, revealed to the learner only
        # *after* a round completes (sealed-bid semantics).
        self._competitor_bid_history: List[List[float]] = []
        # Cooperation / dispute history for mimicking opponents — these
        # live in the oversight log, not in observation.competitor_bid_history.
        self._last_cooperation: Dict[str, CooperationChoice] = {}
        self._last_dispute: Dict[str, DisputeChoice] = {}

    # ── OpenEnv interface ────────────────────────────────────────────

    def reset(self, seed: int | None = None, task_name: str | None = None,
              episode_index: int | None = None, **kwargs) -> SpectrumObservation:
        """Initialize a new episode."""
        if task_name is not None:
            self._task_name = task_name
        if episode_index is not None:
            self._episode_index = episode_index
        effective_seed = seed if seed is not None else self._seed

        # ── Multi-agent (Round 2) branch ────────────────────────────
        if self._task_name in _MULTI_AGENT_TASKS:
            return self._reset_multi_agent(effective_seed)

        # ── Round 1 branch (unchanged) ──────────────────────────────
        self._scenarios = get_scenarios(self._task_name, seed=effective_seed)
        self._rules = get_rules(self._task_name)

        # Clamp episode index
        ep_idx = self._episode_index % len(self._scenarios)
        self._current_episode = self._scenarios[ep_idx]

        # Reset state
        self._active_allocations = []
        self._step_rewards = []
        self._state = SpectrumState(
            episode_id=str(uuid.uuid4()),
            step_count=0,
            task_name=self._task_name,
            requests_total=len(self._current_episode),
        )

        # Inject background occupancy for non-easy tasks (seeded, deterministic)
        if self._task_name != "easy":
            bg_rng = random.Random((effective_seed or 42) + (self._episode_index * 7))
            num_background = bg_rng.randint(2, 4)
            background_users = ["bg-carrier", "bg-enterprise", "bg-municipal", "bg-broadcast"]
            for i in range(num_background):
                band_idx = bg_rng.choice([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
                band = SPECTRUM_GRID[band_idx]
                self._active_allocations.append({
                    "band_index": band_idx,
                    "user_type": "commercial",
                    "user_id": bg_rng.choice(background_users) + f"-{i}",
                    "power_dbm": round(band.max_power_dbm * bg_rng.uniform(0.5, 0.8), 1),
                    "remaining_steps": bg_rng.randint(3, 8),
                    "request_id": f"bg-{self._episode_index}-{i}",
                })

        return self._build_observation(done=False, reward=None)

    def step(self, action) -> SpectrumObservation:
        """Process one decision.

        Dispatches to the Round 1 per-request evaluator for legacy tasks
        and to the multi-agent orchestrator for ``auction`` / ``dispute``
        / ``coalition``. Accepts either a :class:`SpectrumAction` (Round 1)
        or a :class:`MultiAgentAction` (Round 2) — duck-typed, the
        dispatch decides which fields it cares about.

        A 30-second wall-clock timeout bounds each call; exceeding it
        marks the current step done with an error observation.
        """
        start_time = time.monotonic()

        if self._task_name in _MULTI_AGENT_TASKS:
            try:
                return self._step_multi_agent(action, start_time)
            except TimeoutError as te:
                return self._build_multi_agent_observation(
                    done=True,
                    reward=0.0,
                    error=f"step() timeout: {te}",
                )

        step_idx = self._state.step_count

        if step_idx >= len(self._current_episode):
            return self._build_observation(done=True, reward=0.0, error="Episode already complete.")

        request = self._current_episode[step_idx]
        reward, error = self._evaluate_action(action, request)

        num_bands = len(SPECTRUM_GRID)

        # Update allocations for any valid band assignment (0..N-1)
        if 0 <= action.assigned_band_index < num_bands:
            # Handle preemption: count then remove lower-priority allocations
            if request.priority <= 2:
                preempted = [
                    a for a in self._active_allocations
                    if a["band_index"] == action.assigned_band_index
                ]
                self._state.preemptions += len(preempted)
                self._active_allocations = [
                    a for a in self._active_allocations
                    if a["band_index"] != action.assigned_band_index
                ]

            self._active_allocations.append({
                "band_index": action.assigned_band_index,
                "user_type": request.requester_type,
                "user_id": request.requester_id,
                "power_dbm": action.assigned_power_dbm,
                "remaining_steps": request.duration_steps,
                "request_id": request.request_id,
            })
            self._state.successful_allocations += 1

            # Check guard band interference with adjacent active allocations
            assigned = SPECTRUM_GRID[action.assigned_band_index]
            for alloc in self._active_allocations:
                if alloc["request_id"] == request.request_id:
                    continue
                other = SPECTRUM_GRID[alloc["band_index"]]
                gap = max(assigned.start_mhz, other.start_mhz) - min(assigned.end_mhz, other.end_mhz)
                required_guard = max(assigned.guard_band_mhz, other.guard_band_mhz)
                if 0 < gap < required_guard:
                    self._state.interference_events += 1

        elif action.assigned_band_index == -1:
            self._state.rejected_requests += 1

        # Decay remaining steps on all allocations
        for alloc in self._active_allocations:
            alloc["remaining_steps"] = max(0, alloc["remaining_steps"] - 1)
        self._active_allocations = [a for a in self._active_allocations if a["remaining_steps"] > 0]

        self._step_rewards.append(reward)
        self._state.step_count += 1
        self._state.requests_processed += 1
        self._state.accumulated_reward += reward

        done = self._state.step_count >= len(self._current_episode)

        return self._build_observation(done=done, reward=reward, error=error)

    @property
    def state(self) -> SpectrumState:
        return self._state

    # ── Reward computation ───────────────────────────────────────────

    def _evaluate_action(self, action: SpectrumAction, request: ScenarioRequest) -> tuple[float, str | None]:
        """
        Compute reward for a single action against ground truth.

        Returns (reward: 0.0-1.0, error_message or None).

        Reward breakdown:
        - Band selection:     0.35  (correct band assignment)
        - Power compliance:   0.25  (within regulatory limits)
        - Priority handling:  0.25  (correct preemption / rejection logic)
        - Justification:      0.15  (mentions relevant concepts)
        """
        total = 0.0
        error = None
        num_bands = len(SPECTRUM_GRID)

        # ── Band selection (0.35) ────────────────────────────────────
        if action.assigned_band_index == -1:
            # Agent chose to reject
            if request.gt_best_band_index == -1:
                total += 0.35  # correct rejection
            elif request.gt_reject_reason is not None:
                total += 0.15  # partially correct, rejection was acceptable
            else:
                total += 0.0  # incorrect rejection
                error = "Rejected a valid request that should have been allocated."
        elif 0 <= action.assigned_band_index < num_bands:
            if action.assigned_band_index == request.gt_best_band_index:
                total += 0.35  # optimal band
            elif action.assigned_band_index in request.gt_acceptable_bands:
                total += 0.25  # acceptable but not optimal
            else:
                chosen_type = SPECTRUM_GRID[action.assigned_band_index].band_type
                gt_type = SPECTRUM_GRID[request.gt_best_band_index].band_type if request.gt_best_band_index >= 0 else ""
                if chosen_type == gt_type:
                    total += 0.10  # right type of band, wrong specific band
                else:
                    total += 0.0
                    error = f"Assigned to incompatible band type '{chosen_type}'."

                if chosen_type == "protected" and request.requester_type not in ("emergency", "military"):
                    total -= 0.1
                    error = "Regulatory violation: non-emergency user in protected band."
        else:
            # Out-of-range band index: score zero, skip remaining components
            return 0.0, f"Invalid band index: {action.assigned_band_index}."

        # ── Power compliance (0.25) ──────────────────────────────────
        if 0 <= action.assigned_band_index < num_bands:
            band = SPECTRUM_GRID[action.assigned_band_index]
            gt_max = request.gt_max_power_dbm

            if action.assigned_power_dbm <= gt_max:
                if action.assigned_power_dbm <= band.max_power_dbm:
                    total += 0.25
                else:
                    total += 0.10
                    error = error or "Power exceeds band maximum."
            else:
                overshoot = action.assigned_power_dbm - gt_max
                if overshoot <= 3.0:
                    total += 0.15
                elif overshoot <= 10.0:
                    total += 0.05
                else:
                    total += 0.0
                error = error or f"Power {action.assigned_power_dbm} dBm exceeds limit {gt_max} dBm."

        # ── Guard band interference penalty ──────────────────────────
        if 0 <= action.assigned_band_index < num_bands:
            assigned = SPECTRUM_GRID[action.assigned_band_index]
            for alloc in self._active_allocations:
                other = SPECTRUM_GRID[alloc["band_index"]]
                if alloc["band_index"] == action.assigned_band_index:
                    continue
                gap = max(assigned.start_mhz, other.start_mhz) - min(assigned.end_mhz, other.end_mhz)
                required_guard = max(assigned.guard_band_mhz, other.guard_band_mhz)
                if 0 < gap < required_guard:
                    total -= 0.05
                    error = error or "Guard band violation: insufficient separation from adjacent allocation."
                    break

        # ── Priority handling (0.25) ──────────────────────────────────
        # Priority bonus is partially gated on band selection correctness
        band_was_correct = (
            action.assigned_band_index == request.gt_best_band_index
            or action.assigned_band_index in request.gt_acceptable_bands
        )

        if request.priority == 1:
            # Emergency/military: must be allocated immediately
            if action.assigned_band_index >= 0:
                total += 0.15  # Base credit for not rejecting emergency
                if band_was_correct:
                    total += 0.05  # Bonus for correct band
                if request.gt_should_preempt:
                    total += 0.05  # Bonus for preemption awareness
            else:
                total += 0.0  # Rejected emergency = zero
                error = error or "Rejected emergency/military request."
        elif request.gt_should_preempt:
            if action.assigned_band_index >= 0:
                if band_was_correct:
                    total += 0.25  # Full marks: preempted into correct band
                else:
                    total += 0.10  # Allocated but wrong band
            else:
                total += 0.05  # Rejected when should have preempted
        else:
            # Standard requests: priority score gated on band correctness
            if action.assigned_band_index >= 0 or request.gt_best_band_index == -1:
                if band_was_correct or request.gt_best_band_index == -1:
                    total += 0.25  # Correct allocation or correct rejection
                else:
                    total += 0.10  # Allocated to wrong band
            else:
                total += 0.05  # Rejected when should have allocated

        # ── Justification quality (0.15) ─────────────────────────────
        justification = action.justification.lower()
        keywords_present = 0
        relevant_keywords = []

        if request.requester_type == "emergency":
            relevant_keywords = ["emergency", "priority", "public safety", "first responder"]
        elif request.requester_type == "military":
            relevant_keywords = ["military", "commandeer", "priority", "exclusive"]
        elif request.requester_type == "iot":
            relevant_keywords = ["iot", "sensor", "power", "unlicensed", "ism"]
        elif "cognitive" in request.description.lower() or "cbrs" in request.description.lower():
            relevant_keywords = ["cognitive", "secondary", "primary", "cbrs", "sensing", "gaa", "pal"]
        else:
            relevant_keywords = ["band", "power", "frequency", "allocat", "assign"]

        for kw in relevant_keywords:
            if kw in justification:
                keywords_present += 1

        if keywords_present >= 3:
            total += 0.15
        elif keywords_present >= 2:
            total += 0.10
        elif keywords_present >= 1:
            total += 0.05

        # Clamp to [0.0, 1.0]
        total = max(0.0, min(1.0, total))
        return round(total, 4), error

    # ── Observation builder ──────────────────────────────────────────

    def _build_observation(self, done: bool, reward: float | None,
                           error: str | None = None) -> SpectrumObservation:
        """
        Construct the SpectrumObservation for the current step.

        Args:
            done:   Whether the episode has ended.
            reward: Per-step reward (None on reset, float after each step).
            error:  Optional regulatory violation message from _evaluate_action.

        Returns:
            A fully populated SpectrumObservation including spectrum grid
            occupancy, current request, regulatory rules, and (for the
            spectrum_auction task) a look-ahead preview of upcoming requests.
        """
        step_idx = self._state.step_count

        # Current request (or empty dict when the episode is finished)
        if step_idx < len(self._current_episode) and not done:
            req = self._current_episode[step_idx]
            current_request = {
                "request_id": req.request_id,
                "requester_type": req.requester_type,
                "requester_id": req.requester_id,
                "bandwidth_needed_mhz": req.bandwidth_needed_mhz,
                "preferred_band_index": req.preferred_band_index,
                "priority": req.priority,
                "duration_steps": req.duration_steps,
                "power_dbm": req.power_dbm,
                "description": req.description,
            }
        else:
            current_request = {}

        # Spectrum grid annotated with live occupancy information
        grid = get_spectrum_grid()
        for band_info in grid:
            occupants = [
                a for a in self._active_allocations
                if a["band_index"] == band_info["index"]
            ]
            band_info["occupied"] = len(occupants) > 0
            band_info["occupants"] = [
                {
                    "user_type": a["user_type"],
                    "user_id": a["user_id"],
                    "power_dbm": a["power_dbm"],
                    "remaining_steps": a["remaining_steps"],
                }
                for a in occupants
            ]

        # Spectral efficiency: fraction of total bandwidth currently allocated
        occupied_bw = sum(
            SPECTRUM_GRID[a["band_index"]].end_mhz - SPECTRUM_GRID[a["band_index"]].start_mhz
            for a in self._active_allocations
        )
        total_bw = sum(b.end_mhz - b.start_mhz for b in SPECTRUM_GRID)
        efficiency = occupied_bw / total_bw if total_bw > 0 else 0.0

        # Look-ahead: expose next 2 requests for the spectrum_auction task only.
        # Ground-truth fields are intentionally omitted — only observable attributes
        # (type, priority, bandwidth, preferred band, description) are revealed.
        upcoming: List[Dict[str, Any]] = []
        if self._task_name == "spectrum_auction":
            for future_idx in range(step_idx + 1, min(step_idx + 3, len(self._current_episode))):
                future_req = self._current_episode[future_idx]
                upcoming.append({
                    "requester_type": future_req.requester_type,
                    "priority": future_req.priority,
                    "bandwidth_needed_mhz": future_req.bandwidth_needed_mhz,
                    "preferred_band_index": future_req.preferred_band_index,
                    "description": future_req.description,
                })

        return SpectrumObservation(
            spectrum_grid=grid,
            active_allocations=self._active_allocations.copy(),
            current_request=current_request,
            regulatory_rules=self._rules,
            task_difficulty=self._task_name,
            step_number=step_idx,
            total_steps=len(self._current_episode),
            spectral_efficiency=round(efficiency, 4),
            episode_reward_so_far=round(self._state.accumulated_reward, 4),
            last_action_error=error,
            upcoming_requests=upcoming,
            done=done,
            reward=reward,
        )


    # ── Round 2 multi-agent orchestration ────────────────────────────

    def _reset_multi_agent(self, seed: Optional[int]) -> MultiAgentObservation:
        """Initialize a multi-agent episode for Round 2 tasks.

        Strictly separates training (seeds 0-199) from held-out evaluation
        (seeds 200-299) — no cross-pollination happens because each
        scenario generator takes the seed directly.
        """
        effective_seed = int(seed if seed is not None else 42)

        # Coalition generators hold stage state across calls with the
        # same seed; clear it so each reset() starts at stage 0.
        if self._task_name == "coalition":
            _coalition_reset(effective_seed)

        gen = _MULTI_AGENT_GENERATORS[self._task_name]
        scenario, ground_truth = gen(effective_seed)

        self._multi_scenario = scenario
        self._multi_ground_truth = ground_truth
        self._rules = [scenario.description]
        self._regulator = Regulator()
        self._competitor_policies = rotate_policies(effective_seed, num_slots=2)
        self._competitor_bid_history = [[] for _ in self._competitor_policies]
        self._last_cooperation = {}
        self._last_dispute = {}
        self._multi_round_rewards = []
        self._active_allocations = []
        self._step_rewards = []

        # Initialize per-operator state. Learner is always op-0;
        # competitors are op-1, op-2.
        budgets = scenario.params.get("budgets") or [
            scenario.params.get("learner_budget", 100.0)
        ] * (1 + len(self._competitor_policies))
        while len(budgets) < 1 + len(self._competitor_policies):
            budgets.append(100.0)

        self._operator_states = {}
        for i in range(1 + len(self._competitor_policies)):
            op_id = f"op-{i}"
            self._operator_states[op_id] = OperatorState(
                operator_id=op_id,
                budget=float(budgets[i]),
                licenses_held=[],
                reputation=0.5,
                action_history=[],
            )

        total_rounds = _MULTI_AGENT_ROUNDS[self._task_name]

        self._state = SpectrumState(
            episode_id=str(uuid.uuid4()),
            step_count=0,
            task_name=self._task_name,
            requests_total=total_rounds,
        )

        return self._build_multi_agent_observation(done=False, reward=None)

    def _step_multi_agent(self, action, start_time: float) -> MultiAgentObservation:
        """Orchestrate one round of a multi-agent game.

        Ordering (sealed-bid semantics):
          1. Validate the learner's action.
          2. Query each scripted competitor against the *current*
             observation (which does not yet include this round's bids).
          3. Pass all joint actions to the regulator for adjudication.
          4. Collect oversight events into the per-episode log.
          5. Update per-operator state (budgets, reputations, licenses).
          6. Reveal competitor bids by appending to the history.
          7. Compute the four reward components and the weighted total.
          8. Return the updated observation.
        """
        if self._multi_scenario is None or self._multi_ground_truth is None:
            raise RuntimeError("multi-agent task requires reset() before step()")

        total_rounds = _MULTI_AGENT_ROUNDS[self._task_name]
        if self._state.step_count >= total_rounds:
            return self._build_multi_agent_observation(
                done=True, reward=0.0, error="Episode already complete."
            )

        self._check_deadline(start_time)

        learner_action = self._coerce_multi_agent_action(action)
        learner_state = self._operator_states["op-0"]
        pre_step_observation = self._build_multi_agent_observation(
            done=False, reward=None, include_this_step_bids=False
        )

        # ── Query competitors against the pre-step observation.
        competitor_actions: List[MultiAgentAction] = []
        for i, policy in enumerate(self._competitor_policies, start=1):
            self._check_deadline(start_time)
            op_id = f"op-{i}"
            op_state = self._operator_states[op_id]
            competitor_actions.append(
                self._competitor_decision(
                    policy, pre_step_observation, op_state
                )
            )

        # ── Regulator adjudicates.
        self._check_deadline(start_time)
        regulator_result: Dict[str, Any] = {}
        if self._task_name == "auction":
            regulator_result = self._adjudicate_auction(
                learner_action, competitor_actions
            )
        elif self._task_name == "dispute":
            regulator_result = self._adjudicate_dispute(
                learner_action, competitor_actions
            )
        elif self._task_name == "coalition":
            regulator_result = self._adjudicate_coalition(
                learner_action, competitor_actions
            )

        # ── Reveal competitor bids / choices on the observation that
        #    this call will return (post-round visibility). We also
        #    synthesize an AUDIT_TRIGGERED record per competitor action
        #    so the choice is retrievable from oversight_events by
        #    operator_id — both for the Mimicking policy's mirror logic
        #    and for the inference trace display.
        for i, comp_action in enumerate(competitor_actions):
            op_id = f"op-{i + 1}"
            if comp_action.bid_amount is not None:
                self._competitor_bid_history[i].append(float(comp_action.bid_amount))
            if comp_action.dispute_choice is not None:
                self._last_dispute[op_id] = comp_action.dispute_choice
                self._regulator._emit(
                    OversightEventType.AUDIT_TRIGGERED,
                    operator_id=op_id,
                    severity=0.0,
                    explanation=f"Competitor chose dispute={comp_action.dispute_choice.value}.",
                )
            if comp_action.cooperation_flag is not None:
                self._last_cooperation[op_id] = comp_action.cooperation_flag

        # ── Update action history on each operator state.
        learner_state.action_history.append(learner_action.model_dump())
        for i, comp_action in enumerate(competitor_actions, start=1):
            self._operator_states[f"op-{i}"].action_history.append(
                comp_action.model_dump()
            )

        # ── Compute reward components.
        self._check_deadline(start_time)
        post_step_observation = self._build_multi_agent_observation(
            done=False, reward=None, include_this_step_bids=True
        )
        components = self._compute_reward_components(
            learner_action, post_step_observation
        )
        total = compute_total_reward(components)

        self._multi_round_rewards.append(components)
        self._step_rewards.append(total)
        self._state.accumulated_reward += total
        self._state.step_count += 1
        self._state.requests_processed += 1

        done = self._state.step_count >= total_rounds
        obs = self._build_multi_agent_observation(
            done=done,
            reward=total,
            include_this_step_bids=True,
            reward_components=components,
            regulator_result=regulator_result,
        )
        return obs

    def _check_deadline(self, start_time: float) -> None:
        if time.monotonic() - start_time > _STEP_TIMEOUT_SECONDS:
            raise TimeoutError(
                f"step() exceeded wall-clock budget of {_STEP_TIMEOUT_SECONDS}s"
            )

    def _coerce_multi_agent_action(self, action) -> MultiAgentAction:
        """Accept either a MultiAgentAction or an object with the same
        attribute surface; wrap anything else in an empty
        ``MultiAgentAction`` so downstream code never sees a raw dict."""
        if isinstance(action, MultiAgentAction):
            return action
        if isinstance(action, dict):
            return MultiAgentAction(**action)
        payload: Dict[str, Any] = {}
        for field in ("bid_amount", "dispute_choice", "cooperation_flag", "justification"):
            if hasattr(action, field):
                val = getattr(action, field)
                if val is not None:
                    payload[field] = val
        return MultiAgentAction(**payload)

    def _competitor_decision(
        self,
        policy: OperatorPolicy,
        observation: MultiAgentObservation,
        state: OperatorState,
    ) -> MultiAgentAction:
        """Invoke a scripted policy and wrap its output as a
        :class:`MultiAgentAction`. The task name selects which of the
        three decision methods is queried.
        """
        if self._task_name == "auction":
            bid = policy.decide_bid(observation, state)
            # Hard-enforce the budget constraint once more at the env
            # level — we never trust a policy to honor it.
            bid = max(0.0, min(float(bid), float(state.budget)))
            return MultiAgentAction(
                bid_amount=bid,
                justification=f"[scripted {type(policy).__name__}]",
            )
        if self._task_name == "dispute":
            choice = policy.decide_dispute_response(observation, state)
            return MultiAgentAction(
                dispute_choice=choice,
                justification=f"[scripted {type(policy).__name__}]",
            )
        # coalition
        coop = policy.decide_cooperation(observation, state)
        flag = CooperationChoice.COOPERATE if coop else CooperationChoice.DEFECT
        return MultiAgentAction(
            cooperation_flag=flag,
            justification=f"[scripted {type(policy).__name__}]",
        )

    def _adjudicate_auction(
        self,
        learner_action: MultiAgentAction,
        competitor_actions: List[MultiAgentAction],
    ) -> Dict[str, Any]:
        """Resolve one sealed-bid auction round and debit the winner."""
        bids: Dict[str, float] = {}
        budgets: Dict[str, float] = {}

        learner_bid = float(learner_action.bid_amount or 0.0)
        # Enforce the learner's budget constraint. Bids exceeding the
        # remaining budget are clipped at the budget and the overage is
        # reported in the regulator log via the normal aggression rule.
        learner_bid = max(0.0, min(learner_bid, self._operator_states["op-0"].budget))
        bids["op-0"] = learner_bid
        budgets["op-0"] = self._operator_states["op-0"].budget
        for i, comp in enumerate(competitor_actions, start=1):
            op_id = f"op-{i}"
            bid = float(comp.bid_amount or 0.0)
            bids[op_id] = bid
            budgets[op_id] = self._operator_states[op_id].budget

        result = self._regulator.resolve_auction_round(bids, budgets=budgets)

        winner = result["winner"]
        price = float(result["price"])
        winner_state = self._operator_states[winner]
        winner_state.budget = max(0.0, round(winner_state.budget - price, 6))
        # Licenses are abstract integer IDs — we assign the current
        # round index as the license identifier.
        winner_state.licenses_held.append(int(self._state.step_count))
        return result

    def _adjudicate_dispute(
        self,
        learner_action: MultiAgentAction,
        competitor_actions: List[MultiAgentAction],
    ) -> Dict[str, Any]:
        """Resolve one dispute round against the first competitor slot."""
        agent_choice = learner_action.dispute_choice or DisputeChoice.NEGOTIATE
        comp_choice = (
            competitor_actions[0].dispute_choice
            if competitor_actions and competitor_actions[0].dispute_choice is not None
            else DisputeChoice.NEGOTIATE
        )
        result = self._regulator.adjudicate_dispute(
            agent_choice,
            comp_choice,
            agent_id="op-0",
            competitor_id="op-1",
        )
        return result

    def _adjudicate_coalition(
        self,
        learner_action: MultiAgentAction,
        competitor_actions: List[MultiAgentAction],
    ) -> Dict[str, Any]:
        """Resolve one coalition round across all three operators."""
        learner_choice = learner_action.cooperation_flag or CooperationChoice.ABSTAIN

        cooperations: Dict[str, CooperationChoice] = {"op-0": learner_choice}
        for i, comp in enumerate(competitor_actions, start=1):
            cooperations[f"op-{i}"] = (
                comp.cooperation_flag or CooperationChoice.ABSTAIN
            )

        priors = {op_id: st.reputation for op_id, st in self._operator_states.items()}
        result = self._regulator.evaluate_coalition(cooperations, priors)

        for op_id, new_rep in result["posterior_reputations"].items():
            self._operator_states[op_id].reputation = float(new_rep)

        return result

    def _compute_reward_components(
        self,
        learner_action: MultiAgentAction,
        observation: MultiAgentObservation,
    ) -> Dict[str, float]:
        gt = self._multi_ground_truth
        comps = {
            "revenue": reward_revenue(learner_action, observation, gt, self._regulator),
            "interference": reward_interference(
                learner_action, observation, gt, self._regulator
            ),
            "compliance": reward_compliance(
                learner_action, observation, gt, self._regulator
            ),
            "justification": reward_justification(
                learner_action, observation, gt, self._regulator
            ),
        }
        return comps

    def _build_multi_agent_observation(
        self,
        done: bool,
        reward: Optional[float],
        error: Optional[str] = None,
        include_this_step_bids: bool = True,
        reward_components: Optional[Dict[str, float]] = None,
        regulator_result: Optional[Dict[str, Any]] = None,
    ) -> MultiAgentObservation:
        """Construct a MultiAgentObservation reflecting current state.

        If ``include_this_step_bids`` is False, competitor bid entries
        added by the current step are withheld — this gives policies
        the sealed-bid view during the pre-step observation used to
        query them.
        """
        total_rounds = _MULTI_AGENT_ROUNDS.get(self._task_name, 1)
        learner_state = self._operator_states.get(
            "op-0",
            OperatorState(operator_id="op-0", budget=0.0, reputation=0.5),
        )

        # Competitor bid history copy (optionally trimmed).
        if include_this_step_bids:
            comp_hist = [list(h) for h in self._competitor_bid_history]
        else:
            comp_hist = [list(h) for h in self._competitor_bid_history]

        # Oversight events: serialize to dicts, keep newest 20.
        oversight_dicts: List[Dict[str, Any]] = []
        for ev in self._regulator.event_log[-20:]:
            d = ev.model_dump()
            d["event_type"] = (
                ev.event_type.value if hasattr(ev.event_type, "value") else ev.event_type
            )
            # Attach attributable choice labels so Mimicking competitors
            # can mirror them without peeking at private state.
            owner = ev.operator_id
            if owner in self._last_cooperation:
                d["cooperation_flag"] = self._last_cooperation[owner].value
            if owner in self._last_dispute:
                d["dispute_choice"] = self._last_dispute[owner].value
            oversight_dicts.append(d)

        # Basic grid/occupancy payload (empty for multi-agent tasks since
        # they do not model a shared spectrum grid).
        grid = get_spectrum_grid()

        round_index = min(self._state.step_count, total_rounds - 1)

        metadata: Dict[str, Any] = {"reward_components": reward_components or {}}
        if regulator_result is not None:
            serializable = dict(regulator_result)
            if "events" in serializable:
                serializable["events"] = [
                    ev.model_dump() if hasattr(ev, "model_dump") else ev
                    for ev in serializable["events"]
                ]
            metadata["regulator_result"] = serializable
        if reward_components is not None:
            metadata.update(
                reward_revenue=reward_components.get("revenue"),
                reward_interference=reward_components.get("interference"),
                reward_compliance=reward_components.get("compliance"),
                reward_justification=reward_components.get("justification"),
            )

        return MultiAgentObservation(
            spectrum_grid=grid,
            active_allocations=[],
            current_request={
                "task_name": self._task_name,
                "description": self._multi_scenario.description if self._multi_scenario else "",
                "round_index": round_index,
                "total_rounds": total_rounds,
            },
            regulatory_rules=self._rules,
            task_difficulty=self._task_name,
            step_number=self._state.step_count,
            total_steps=total_rounds,
            spectral_efficiency=0.0,
            episode_reward_so_far=round(self._state.accumulated_reward, 4),
            last_action_error=error,
            upcoming_requests=[],
            competitor_bid_history=comp_hist,
            reputation_score=float(learner_state.reputation),
            oversight_events=oversight_dicts,
            remaining_budget=float(learner_state.budget),
            opponent_slot_indices=list(range(1, 1 + len(self._competitor_policies))),
            round_index=round_index,
            total_rounds=total_rounds,
            done=done,
            reward=reward,
            metadata=metadata,
        )

    # ── External oversight log accessor ──────────────────────────────

    def get_oversight_log(self) -> List[Dict[str, Any]]:
        """Return the current episode's oversight log as JSON-ready dicts.

        Surfaced by the ``GET /oversight`` FastAPI endpoint so the demo
        can render the regulator's audit trail live.
        """
        out: List[Dict[str, Any]] = []
        for ev in self._regulator.event_log:
            d = ev.model_dump()
            d["event_type"] = (
                ev.event_type.value if hasattr(ev.event_type, "value") else ev.event_type
            )
            out.append(d)
        return out


# ── Grader ───────────────────────────────────────────────────────────

def grade_episode(rewards: List[float]) -> float:
    """
    Compute the final episode score from per-step rewards.

    Uses mean reward with a penalty for catastrophic failures (steps scoring
    below 0.1). This prevents agents from gaming the score by bombing early
    steps and recovering later.

    Returns a float in [0.0, 1.0].
    """
    if not rewards:
        return 0.0

    mean_reward = sum(rewards) / len(rewards)

    # Penalize catastrophic failures: each step below 0.1 costs 0.05
    catastrophic_count = sum(1 for r in rewards if r < 0.1)
    penalty = catastrophic_count * 0.05

    return round(max(0.0, min(1.0, mean_reward - penalty)), 4)
