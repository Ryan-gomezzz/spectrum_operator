"""
RF Spectrum Allocation - Scenario Generator
=============================================
Generates deterministic scenario pools for all five task difficulties.
Each scenario is a sequence of allocation requests with ground-truth
optimal actions for grading.

Task registry pattern:
  TASK_REGISTRY maps task_name → {builder, rules, description, steps_per_episode}
  Each builder returns List[List[ScenarioRequest]]: a list of episodes,
  each episode being an ordered list of requests for the agent to process.

Ground-truth fields on ScenarioRequest drive the reward function:
  gt_best_band_index   : optimal band index (-1 = reject is correct)
  gt_acceptable_bands  : bands that earn partial credit
  gt_max_power_dbm     : power ceiling for full marks
  gt_should_preempt    : whether preempting existing users is required
  gt_reject_reason     : non-None when rejection is the correct action
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class BandSpec:
    """Specification for a frequency band in the grid."""
    start_mhz: float
    end_mhz: float
    label: str
    band_type: str  # "licensed", "unlicensed", "protected", "shared"
    max_power_dbm: float
    guard_band_mhz: float  # required guard on each side


@dataclass
class ScenarioRequest:
    """A single allocation request within a scenario."""
    request_id: str
    requester_type: str
    requester_id: str
    bandwidth_needed_mhz: float
    preferred_band_index: Optional[int]
    priority: int
    duration_steps: int
    power_dbm: float
    description: str

    # Ground truth for grading
    gt_best_band_index: int        # -1 means reject is correct
    gt_acceptable_bands: List[int]  # any of these earns partial credit
    gt_max_power_dbm: float
    gt_should_preempt: bool         # should this preempt a lower-priority user?
    gt_reject_reason: Optional[str]  # if rejection is correct, why


# ── Band grid (shared across all tasks) ──────────────────────────────

SPECTRUM_GRID: List[BandSpec] = [
    BandSpec(700.0, 710.0,  "Band 0: 700 MHz Public Safety", "protected",  30.0, 1.0),
    BandSpec(710.0, 730.0,  "Band 1: 700 MHz LTE",           "licensed",   43.0, 0.5),
    BandSpec(730.0, 746.0,  "Band 2: 700 MHz LTE-B",         "licensed",   43.0, 0.5),
    BandSpec(850.0, 870.0,  "Band 3: 850 MHz Cellular",      "licensed",   40.0, 0.5),
    BandSpec(870.0, 890.0,  "Band 4: 850 MHz Cellular-B",    "licensed",   40.0, 0.5),
    BandSpec(1700.0, 1720.0, "Band 5: AWS-1 Uplink",         "licensed",   38.0, 1.0),
    BandSpec(1720.0, 1755.0, "Band 6: AWS-1 Extended",       "licensed",   38.0, 1.0),
    BandSpec(2400.0, 2450.0, "Band 7: 2.4 GHz ISM-A",        "unlicensed", 20.0, 0.0),
    BandSpec(2450.0, 2483.5, "Band 8: 2.4 GHz ISM-B",        "unlicensed", 20.0, 0.0),
    BandSpec(3550.0, 3600.0, "Band 9: CBRS PAL",             "shared",     30.0, 0.5),
    BandSpec(3600.0, 3650.0, "Band 10: CBRS GAA",            "shared",     23.0, 0.5),
    BandSpec(5150.0, 5250.0, "Band 11: 5 GHz UNII-1",        "unlicensed", 23.0, 0.0),
]

# ── Regulatory rule sets ──────────────────────────────────────────────

REGULATORY_RULES_BASE = [
    "Emergency services (priority 1) must be allocated within protected or licensed bands.",
    "Protected bands (Band 0) are reserved for public safety; commercial use is prohibited.",
    "Guard bands must be maintained: no allocation may overlap another active allocation's guard band.",
    "Power must not exceed the band's maximum rated power.",
    "Unlicensed bands (ISM/UNII) are open-access but have strict power limits.",
    "CBRS shared bands: PAL users have priority over GAA users.",
    "Higher-priority requests may preempt lower-priority allocations in shared/licensed bands.",
]

REGULATORY_RULES_HARD = REGULATORY_RULES_BASE + [
    "Primary users in CBRS PAL bands cannot be preempted by secondary users.",
    "Military requests (priority 1) may commandeer any non-emergency band with 0-step notice.",
    "Concurrent allocations in adjacent bands must account for aggregate interference.",
    "IoT devices in unlicensed bands must use power ≤ 14 dBm to avoid interference.",
]

REGULATORY_RULES_DISASTER = REGULATORY_RULES_BASE + [
    "During declared disasters, all non-essential commercial allocations may be suspended.",
    "Emergency IoT systems (early warning, flood sensors) receive elevated priority during disasters.",
    "Multiple emergency agencies must share protected spectrum via time-division when Band 0 is at capacity.",
]

REGULATORY_RULES_AUCTION = REGULATORY_RULES_BASE + [
    "When spectrum is constrained, coordinators should consider upcoming demand before committing bands to low-priority requests.",
    "Reservation of bands for anticipated high-priority traffic is permitted when queue visibility is available.",
]


# ── Scenario builders ────────────────────────────────────────────────

def _build_easy_scenarios(seed: int = 42) -> List[List[ScenarioRequest]]:
    """
    Easy: mostly empty spectrum, obvious assignments, no conflicts.

    All requests map to a single preferred band with straightforward
    power compliance. Ground truth is always the requester's preferred band.
    """
    rng = random.Random(seed)
    scenarios = []

    for ep in range(50):
        episode: List[ScenarioRequest] = []
        for step in range(5):
            req_type = rng.choice(["commercial", "iot", "amateur"])

            if req_type == "commercial":
                bw = rng.choice([10.0, 16.0, 20.0])
                best = rng.choice([1, 2, 3, 4])          # licensed cellular bands
                pwr = rng.uniform(30.0, 40.0)
                desc = f"Commercial LTE base station needs {bw} MHz in cellular band."
            elif req_type == "iot":
                bw = rng.choice([5.0, 10.0])
                best = rng.choice([7, 8, 11])             # unlicensed ISM/UNII bands
                pwr = rng.uniform(10.0, 14.0)
                desc = f"IoT sensor network requesting {bw} MHz in ISM band."
            else:  # amateur
                bw = 5.0
                best = rng.choice([7, 8])
                pwr = rng.uniform(5.0, 10.0)
                desc = "Amateur radio operator requesting spectrum for local use."

            episode.append(ScenarioRequest(
                request_id=f"easy-{ep}-{step}",
                requester_type=req_type,
                requester_id=f"{req_type}-{rng.randint(100, 999)}",
                bandwidth_needed_mhz=bw,
                preferred_band_index=best,
                priority=rng.choice([3, 4, 5]),
                duration_steps=rng.randint(2, 5),
                power_dbm=round(pwr, 1),
                description=desc,
                gt_best_band_index=best,
                gt_acceptable_bands=[best],
                gt_max_power_dbm=SPECTRUM_GRID[best].max_power_dbm,
                gt_should_preempt=False,
                gt_reject_reason=None,
            ))
        scenarios.append(episode)
    return scenarios


def _build_medium_scenarios(seed: int = 123) -> List[List[ScenarioRequest]]:
    """
    Medium: moderate occupancy, priority conflicts, guard band awareness.

    Introduces emergency requests, CBRS PAL/GAA priority, and commercial
    users attempting to use the protected band (Band 0).

    Roll probabilities per step:
      <0.15  → emergency request (protected/licensed band, preempt=True)
      <0.35  → commercial trying protected band (should redirect to Band 1)
      <0.55  → CBRS PAL or GAA request (shared band priority awareness)
      else   → standard commercial or IoT assignment
    """
    rng = random.Random(seed)
    scenarios = []

    for ep in range(50):
        episode: List[ScenarioRequest] = []
        for step in range(8):
            roll = rng.random()

            if roll < 0.15:
                # 15% chance: emergency → must go to protected Band 0 with preemption
                episode.append(ScenarioRequest(
                    request_id=f"med-{ep}-{step}",
                    requester_type="emergency",
                    requester_id=f"ems-{rng.randint(10, 99)}",
                    bandwidth_needed_mhz=10.0,
                    preferred_band_index=None,
                    priority=1,
                    duration_steps=rng.randint(3, 8),
                    power_dbm=28.0,
                    description="Emergency dispatch requires immediate spectrum for first responder comms.",
                    gt_best_band_index=0,
                    gt_acceptable_bands=[0, 1],
                    gt_max_power_dbm=30.0,
                    gt_should_preempt=True,
                    gt_reject_reason=None,
                ))
            elif roll < 0.35:
                # 20% chance: commercial wanting Band 0 — should redirect to Band 1
                episode.append(ScenarioRequest(
                    request_id=f"med-{ep}-{step}",
                    requester_type="commercial",
                    requester_id=f"telco-{rng.randint(100, 999)}",
                    bandwidth_needed_mhz=10.0,
                    preferred_band_index=0,
                    priority=3,
                    duration_steps=rng.randint(3, 6),
                    power_dbm=35.0,
                    description="Mobile carrier requesting Band 0 for capacity expansion.",
                    gt_best_band_index=1,       # redirect: Band 1 is correct, not Band 0
                    gt_acceptable_bands=[1, 2, 3, 4],
                    gt_max_power_dbm=43.0,
                    gt_should_preempt=False,
                    gt_reject_reason=None,
                ))
            elif roll < 0.55:
                # 20% chance: CBRS PAL vs GAA priority awareness
                is_pal = rng.random() < 0.5
                episode.append(ScenarioRequest(
                    request_id=f"med-{ep}-{step}",
                    requester_type="commercial" if is_pal else "iot",
                    requester_id=f"cbrs-{rng.randint(100, 999)}",
                    bandwidth_needed_mhz=rng.choice([10.0, 20.0]),
                    preferred_band_index=9 if is_pal else 10,  # PAL→9, GAA→10
                    priority=2 if is_pal else 4,
                    duration_steps=rng.randint(2, 6),
                    power_dbm=28.0 if is_pal else 20.0,
                    description=f"CBRS {'PAL licensee' if is_pal else 'GAA user'} requesting shared spectrum.",
                    gt_best_band_index=9 if is_pal else 10,
                    gt_acceptable_bands=[9, 10],
                    gt_max_power_dbm=30.0 if is_pal else 23.0,
                    gt_should_preempt=is_pal,   # PAL can preempt GAA
                    gt_reject_reason=None,
                ))
            else:
                # 45% chance: standard commercial or IoT with straightforward assignment
                req_type = rng.choice(["commercial", "iot"])
                if req_type == "commercial":
                    best = rng.choice([1, 2, 3, 4, 5, 6])
                    bw = rng.choice([10.0, 16.0, 20.0])
                    pwr = rng.uniform(30.0, 40.0)
                    desc = f"Carrier needs {bw} MHz for urban macro cell."
                else:
                    best = rng.choice([7, 8, 11])
                    bw = rng.choice([5.0, 10.0])
                    pwr = rng.uniform(10.0, 14.0)
                    desc = f"IoT deployment needs {bw} MHz in unlicensed band."

                episode.append(ScenarioRequest(
                    request_id=f"med-{ep}-{step}",
                    requester_type=req_type,
                    requester_id=f"{req_type}-{rng.randint(100, 999)}",
                    bandwidth_needed_mhz=bw,
                    preferred_band_index=best,
                    priority=rng.choice([2, 3, 4]),
                    duration_steps=rng.randint(2, 6),
                    power_dbm=round(pwr, 1),
                    description=desc,
                    gt_best_band_index=best,
                    gt_acceptable_bands=[best],
                    gt_max_power_dbm=SPECTRUM_GRID[best].max_power_dbm,
                    gt_should_preempt=False,
                    gt_reject_reason=None,
                ))
        scenarios.append(episode)
    return scenarios


def _build_disaster_response_scenarios(seed: int = 777) -> List[List[ScenarioRequest]]:
    """
    Disaster Response: two-phase episode simulating a coastal cyclone.

    Phase 1 (steps 0-2): Normal commercial/IoT operations — business as usual.
    Phase 2 (steps 3-9): Cyclone strikes — a burst of emergency and military
    requests requires the agent to preempt its own earlier commercial allocations,
    handle competing agencies on the protected band, upgrade IoT early-warning
    devices to elevated priority, and REJECT non-essential commercial traffic.

    This task tests temporal reasoning (situation changed mid-episode),
    self-correction (preempting own earlier decisions), and the ability to
    recognise that rejection is sometimes the correct action.

    Ground truth design:
      Steps 0-2  → normal assignments, gt_should_preempt=False
      Steps 3,4,7 → emergency → Band 0, gt_should_preempt=True
      Step 5     → military → licensed band, gt_should_preempt=True
      Step 6     → IoT early warning (P2 upgrade) → Band 1, gt_max_power_dbm=30.0
      Steps 8,9  → commercial during disaster → gt_best_band_index=-1 (reject)
    """
    rng = random.Random(seed)
    scenarios = []

    agency_names = ["NDRF", "fire brigade", "police", "ambulance", "coast guard"]
    military_bands = [1, 3, 5, 6]  # licensed bands military can commandeer

    for ep in range(50):
        episode: List[ScenarioRequest] = []

        # ── Phase 1: Normal operations (steps 0-2) ───────────────────
        for step in range(3):
            req_type = rng.choice(["commercial", "iot"])
            if req_type == "commercial":
                bw = rng.choice([10.0, 16.0, 20.0])
                best = rng.choice([1, 2, 3, 4])
                pwr = rng.uniform(30.0, 38.0)
                desc = (f"Commercial LTE station needs {bw} MHz for routine operations. "
                        f"Normal traffic conditions.")
            else:
                bw = rng.choice([5.0, 10.0])
                best = rng.choice([7, 8, 11])
                pwr = rng.uniform(10.0, 14.0)
                desc = (f"IoT sensor network requests {bw} MHz for routine monitoring. "
                        f"Normal operations.")

            episode.append(ScenarioRequest(
                request_id=f"dis-{ep}-{step}",
                requester_type=req_type,
                requester_id=f"{req_type}-{rng.randint(100, 999)}",
                bandwidth_needed_mhz=bw,
                preferred_band_index=best,
                priority=rng.choice([3, 4, 5]),
                duration_steps=rng.randint(3, 6),
                power_dbm=round(pwr, 1),
                description=desc,
                gt_best_band_index=best,
                gt_acceptable_bands=[best],
                gt_max_power_dbm=SPECTRUM_GRID[best].max_power_dbm,
                gt_should_preempt=False,
                gt_reject_reason=None,
            ))

        # ── Phase 2: Disaster strikes (steps 3-9) ────────────────────

        # Step 3: First responder emergency (NDRF) — needs protected Band 0
        agency = rng.choice(agency_names)
        episode.append(ScenarioRequest(
            request_id=f"dis-{ep}-3",
            requester_type="emergency",
            requester_id=f"ndrf-{rng.randint(10, 99)}",
            bandwidth_needed_mhz=10.0,
            preferred_band_index=0,
            priority=1,
            duration_steps=rng.randint(5, 10),
            power_dbm=28.0,
            description=(f"CYCLONE ALERT: {agency} requires immediate spectrum for disaster "
                         f"coordination. Preempt all non-emergency users from Band 0."),
            gt_best_band_index=0,
            gt_acceptable_bands=[0, 1],
            gt_max_power_dbm=30.0,
            gt_should_preempt=True,
            gt_reject_reason=None,
        ))

        # Step 4: Second emergency agency competing for protected band
        agency2 = rng.choice(agency_names)
        episode.append(ScenarioRequest(
            request_id=f"dis-{ep}-4",
            requester_type="emergency",
            requester_id=f"ems-{rng.randint(10, 99)}",
            bandwidth_needed_mhz=10.0,
            preferred_band_index=0,
            priority=1,
            duration_steps=rng.randint(4, 8),
            power_dbm=28.0,
            description=(f"CYCLONE: {agency2} requests spectrum. Band 0 may be at capacity — "
                         f"time-division share or spill to Band 1."),
            gt_best_band_index=0,
            gt_acceptable_bands=[0, 1],
            gt_max_power_dbm=30.0,
            gt_should_preempt=True,
            gt_reject_reason=None,
        ))

        # Step 5: Military commandeers a licensed band for secure communications
        mil_band = rng.choice(military_bands)
        episode.append(ScenarioRequest(
            request_id=f"dis-{ep}-5",
            requester_type="military",
            requester_id=f"mil-{rng.randint(10, 99)}",
            bandwidth_needed_mhz=20.0,
            preferred_band_index=mil_band,
            priority=1,
            duration_steps=rng.randint(5, 10),
            power_dbm=43.0,
            description=(f"CYCLONE: Military commandeers Band {mil_band} for secure "
                         f"disaster-response comms. Immediate exclusive access required."),
            gt_best_band_index=mil_band,
            gt_acceptable_bands=[mil_band],
            gt_max_power_dbm=43.0,
            gt_should_preempt=True,
            gt_reject_reason=None,
        ))

        # Step 6: IoT early-warning system — priority upgrade from P4 to P2,
        # needs a licensed band (not ISM) for better range during disaster
        episode.append(ScenarioRequest(
            request_id=f"dis-{ep}-6",
            requester_type="iot",
            requester_id=f"iot-ews-{rng.randint(10, 99)}",
            bandwidth_needed_mhz=10.0,
            preferred_band_index=1,          # licensed band for wider coverage
            priority=2,                       # upgraded from normal IoT P4
            duration_steps=rng.randint(5, 10),
            power_dbm=25.0,
            description=("CYCLONE: Disaster-response early-warning IoT — flood sensors "
                         "require elevated power and licensed band for coastal coverage. "
                         "Priority upgraded to P2 under disaster protocol."),
            gt_best_band_index=1,
            gt_acceptable_bands=[1, 2, 3],   # any licensed band acceptable
            gt_max_power_dbm=30.0,           # elevated limit for disaster IoT
            gt_should_preempt=False,
            gt_reject_reason=None,
        ))

        # Step 7: Additional ambulance / mutual-aid emergency
        episode.append(ScenarioRequest(
            request_id=f"dis-{ep}-7",
            requester_type="emergency",
            requester_id=f"amb-{rng.randint(10, 99)}",
            bandwidth_needed_mhz=10.0,
            preferred_band_index=0,
            priority=1,
            duration_steps=rng.randint(3, 6),
            power_dbm=28.0,
            description=("CYCLONE: Ambulance mutual-aid coordination needs additional "
                         "spectrum. Protected band or licensed fallback required."),
            gt_best_band_index=0,
            gt_acceptable_bands=[0, 1],
            gt_max_power_dbm=30.0,
            gt_should_preempt=True,
            gt_reject_reason=None,
        ))

        # Steps 8-9: Commercial users requesting spectrum during active disaster
        # Correct action is REJECTION — non-essential traffic suspended
        reject_reason = "Non-essential traffic suspended during disaster operations"
        for sub in range(2):
            pref = rng.choice([1, 2, 3, 4])
            episode.append(ScenarioRequest(
                request_id=f"dis-{ep}-{8 + sub}",
                requester_type="commercial",
                requester_id=f"telco-{rng.randint(100, 999)}",
                bandwidth_needed_mhz=rng.choice([10.0, 20.0]),
                preferred_band_index=pref,
                priority=3,
                duration_steps=rng.randint(2, 5),
                power_dbm=round(rng.uniform(30.0, 38.0), 1),
                description=(f"Mobile carrier requesting Band {pref} for capacity expansion. "
                             f"[DISASTER ACTIVE — commercial moratorium in effect]"),
                gt_best_band_index=-1,     # reject is the correct action
                gt_acceptable_bands=[],
                gt_max_power_dbm=0.0,
                gt_should_preempt=False,
                gt_reject_reason=reject_reason,
            ))

        scenarios.append(episode)
    return scenarios


def _build_hard_scenarios(seed: int = 999) -> List[List[ScenarioRequest]]:
    """
    Hard: dense occupancy, cascading preemptions, military, cognitive radio dynamics.

    Roll probabilities per step:
      <0.12  → military commandeer (any licensed/shared band)
      <0.22  → emergency competing with existing emergency
      <0.35  → overpowered IoT (gt_max_power_dbm capped to 14 dBm)
      <0.50  → cognitive radio secondary user (redirect PAL→GAA)
      <0.65  → adjacent band interference (power reduction required)
      else   → dense general traffic (any band, any type)
    """
    rng = random.Random(seed)
    scenarios = []

    for ep in range(50):
        episode: List[ScenarioRequest] = []
        for step in range(12):
            roll = rng.random()

            if roll < 0.12:
                # 12% chance: military commandeer — preempts everything on target band
                target = rng.choice([1, 2, 3, 5, 6, 9])
                episode.append(ScenarioRequest(
                    request_id=f"hard-{ep}-{step}",
                    requester_type="military",
                    requester_id=f"mil-{rng.randint(10, 99)}",
                    bandwidth_needed_mhz=20.0,
                    preferred_band_index=target,
                    priority=1,
                    duration_steps=rng.randint(5, 12),
                    power_dbm=43.0,
                    description="Military operation requires immediate exclusive spectrum access.",
                    gt_best_band_index=target,
                    gt_acceptable_bands=[target],
                    gt_max_power_dbm=43.0,
                    gt_should_preempt=True,
                    gt_reject_reason=None,
                ))
            elif roll < 0.22:
                # 10% chance: emergency competing with existing emergency for Band 0
                episode.append(ScenarioRequest(
                    request_id=f"hard-{ep}-{step}",
                    requester_type="emergency",
                    requester_id=f"ems-{rng.randint(10, 99)}",
                    bandwidth_needed_mhz=10.0,
                    preferred_band_index=0,
                    priority=1,
                    duration_steps=rng.randint(3, 8),
                    power_dbm=28.0,
                    description="Multiple-alarm fire; dispatch needs additional spectrum for mutual aid.",
                    gt_best_band_index=0,
                    gt_acceptable_bands=[0, 1],
                    gt_max_power_dbm=30.0,
                    gt_should_preempt=True,
                    gt_reject_reason=None,
                ))
            elif roll < 0.35:
                # 13% chance: overpowered IoT — hard rules cap power at 14 dBm
                episode.append(ScenarioRequest(
                    request_id=f"hard-{ep}-{step}",
                    requester_type="iot",
                    requester_id=f"iot-{rng.randint(100, 999)}",
                    bandwidth_needed_mhz=10.0,
                    preferred_band_index=rng.choice([7, 8]),
                    priority=5,
                    duration_steps=rng.randint(2, 4),
                    power_dbm=25.0,              # exceeds hard-mode IoT limit of 14 dBm
                    description="Smart city sensor grid requesting ISM band at elevated power.",
                    gt_best_band_index=rng.choice([7, 8]),
                    gt_acceptable_bands=[7, 8],
                    gt_max_power_dbm=14.0,       # agent must cap power to 14 dBm
                    gt_should_preempt=False,
                    gt_reject_reason=None,
                ))
            elif roll < 0.50:
                # 15% chance: cognitive radio secondary user must avoid PAL primary
                episode.append(ScenarioRequest(
                    request_id=f"hard-{ep}-{step}",
                    requester_type="commercial",
                    requester_id=f"cr-{rng.randint(100, 999)}",
                    bandwidth_needed_mhz=rng.choice([10.0, 20.0]),
                    preferred_band_index=9,      # wants PAL, but primary may be active
                    priority=3,
                    duration_steps=rng.randint(2, 5),
                    power_dbm=23.0,
                    description="Cognitive radio secondary user sensing CBRS band for opportunistic access.",
                    gt_best_band_index=10,       # redirect to GAA instead of PAL
                    gt_acceptable_bands=[10, 7, 8],
                    gt_max_power_dbm=23.0,
                    gt_should_preempt=False,
                    gt_reject_reason=None,
                ))
            elif roll < 0.65:
                # 15% chance: adjacent band interference — power reduction required
                adj = rng.choice([(1, 2), (3, 4), (7, 8), (5, 6)])
                episode.append(ScenarioRequest(
                    request_id=f"hard-{ep}-{step}",
                    requester_type="commercial",
                    requester_id=f"telco-{rng.randint(100, 999)}",
                    bandwidth_needed_mhz=20.0,
                    preferred_band_index=adj[0],
                    priority=2,
                    duration_steps=rng.randint(3, 8),
                    power_dbm=43.0,              # requested at max, but adjacency requires reduction
                    description=f"High-power deployment in Band {adj[0]}, adjacent Band {adj[1]} is active.",
                    gt_best_band_index=adj[0],
                    gt_acceptable_bands=[adj[0]],
                    gt_max_power_dbm=38.0,       # must reduce power due to adjacent-band interference
                    gt_should_preempt=False,
                    gt_reject_reason=None,
                ))
            else:
                # 35% chance: dense general traffic — any band type
                req_type = rng.choice(["commercial", "commercial", "iot", "amateur"])
                best = rng.choice(list(range(len(SPECTRUM_GRID))))
                bw = rng.choice([5.0, 10.0, 16.0, 20.0])
                pwr = rng.uniform(15.0, 40.0)
                episode.append(ScenarioRequest(
                    request_id=f"hard-{ep}-{step}",
                    requester_type=req_type,
                    requester_id=f"{req_type}-{rng.randint(100, 999)}",
                    bandwidth_needed_mhz=bw,
                    preferred_band_index=best,
                    priority=rng.choice([2, 3, 4, 5]),
                    duration_steps=rng.randint(1, 4),
                    power_dbm=round(pwr, 1),
                    description=f"{req_type.title()} user needs {bw} MHz; spectrum congested.",
                    gt_best_band_index=best,
                    gt_acceptable_bands=[best] + [
                        b for b in range(len(SPECTRUM_GRID))
                        if SPECTRUM_GRID[b].band_type == SPECTRUM_GRID[best].band_type
                        and b != best
                    ],
                    gt_max_power_dbm=SPECTRUM_GRID[best].max_power_dbm,
                    gt_should_preempt=False,
                    gt_reject_reason=None,
                ))
        scenarios.append(episode)
    return scenarios


def _build_spectrum_auction_scenarios(seed: int = 555) -> List[List[ScenarioRequest]]:
    """
    Spectrum Auction: forward-looking allocation with queue visibility.

    The observation exposes the next 2 upcoming requests so the agent can
    plan globally optimal assignments rather than greedy local ones.

    Each episode follows a fixed 8-step structure with three recurring patterns
    designed so greedy allocation (always pick the obvious band) scores lower
    than look-ahead allocation:

    Pattern A — Save Band 1 (steps 0-1):
      Step 0: Commercial P3 prefers Band 1. BUT upcoming step 1 is an
              emergency P1 that needs Band 1. Optimal: send commercial to
              Band 3 now (gt_best=3), keeping Band 1 free for the emergency.
              Greedy (assign to Band 1) earns only partial credit.
      Step 1: Emergency P1 → Band 0 or Band 1. gt_best=0.

    Pattern B — Spread IoT across ISM (steps 2-4):
      Three consecutive IoT P4 requests all prefer Band 7. Optimal is to
      spread across Bands 7, 8, 11 (ISM-A, ISM-B, UNII-1) for diversity.
      gt_best cycles: 7 → 8 → 11. Piling all on Band 7 earns only partial
      credit for steps 1 and 2.

    Pattern C — Reject low-priority to save spectrum (steps 5-6):
      Step 5: Commercial P5 requests a licensed band. Upcoming is military
              P1 that will need any licensed band. Optimal: reject now
              (gt_best=-1). Assigning earns 0 band-selection credit.
      Step 6: Military P1 → commandeers a licensed band with preemption.

    Pattern D — Standard filler (step 7):
      Normal assignment with no look-ahead advantage.
    """
    rng = random.Random(seed)
    scenarios = []

    for ep in range(50):
        episode: List[ScenarioRequest] = []

        # ── Pattern A: Save Band 1 for incoming emergency ─────────────

        # Step 0: Commercial P3 — locally Band 1 is fine, but globally Band 3 is better
        commercial_bw = rng.choice([10.0, 16.0, 20.0])
        commercial_pwr = rng.uniform(30.0, 38.0)
        episode.append(ScenarioRequest(
            request_id=f"auc-{ep}-0",
            requester_type="commercial",
            requester_id=f"telco-{rng.randint(100, 999)}",
            bandwidth_needed_mhz=commercial_bw,
            preferred_band_index=1,          # prefers Band 1
            priority=3,
            duration_steps=rng.randint(3, 6),
            power_dbm=round(commercial_pwr, 1),
            description=(f"Commercial LTE needs {commercial_bw} MHz — Band 1 or Band 3 "
                         f"both acceptable. Check queue: emergency incoming."),
            gt_best_band_index=3,            # globally optimal: leave Band 1 free for emergency
            gt_acceptable_bands=[1, 2, 3, 4, 5, 6],  # Band 1 earns partial credit
            gt_max_power_dbm=SPECTRUM_GRID[3].max_power_dbm,
            gt_should_preempt=False,
            gt_reject_reason=None,
        ))

        # Step 1: Emergency P1 arrives — needs Band 0 or Band 1
        episode.append(ScenarioRequest(
            request_id=f"auc-{ep}-1",
            requester_type="emergency",
            requester_id=f"ems-{rng.randint(10, 99)}",
            bandwidth_needed_mhz=10.0,
            preferred_band_index=0,
            priority=1,
            duration_steps=rng.randint(3, 8),
            power_dbm=28.0,
            description=("Emergency dispatch requires immediate spectrum. "
                         "Protected Band 0 preferred; Band 1 as fallback."),
            gt_best_band_index=0,
            gt_acceptable_bands=[0, 1],
            gt_max_power_dbm=30.0,
            gt_should_preempt=True,
            gt_reject_reason=None,
        ))

        # ── Pattern B: Spread IoT across ISM bands ────────────────────
        ism_spread = [7, 8, 11]  # optimal assignment order for three IoT requests
        iot_bw = rng.choice([5.0, 10.0])
        for sub, target_band in enumerate(ism_spread):
            episode.append(ScenarioRequest(
                request_id=f"auc-{ep}-{2 + sub}",
                requester_type="iot",
                requester_id=f"iot-{rng.randint(100, 999)}",
                bandwidth_needed_mhz=iot_bw,
                preferred_band_index=7,      # all three prefer Band 7 (greedy trap)
                priority=4,
                duration_steps=rng.randint(2, 5),
                power_dbm=round(rng.uniform(10.0, 14.0), 1),
                description=(f"IoT sensor cluster {sub + 1}/3 requesting ISM band. "
                             f"All clusters prefer Band 7 — spread across ISM/UNII for efficiency."),
                gt_best_band_index=target_band,     # 7 → 8 → 11 for diversity
                gt_acceptable_bands=[7, 8, 11],     # any ISM/UNII earns partial credit
                gt_max_power_dbm=SPECTRUM_GRID[target_band].max_power_dbm,
                gt_should_preempt=False,
                gt_reject_reason=None,
            ))

        # ── Pattern C: Reject low-priority to reserve for military ────

        # Step 5: Commercial P5 — upcoming military P1 will need licensed band
        lic_band = rng.choice([3, 4, 5, 6])
        episode.append(ScenarioRequest(
            request_id=f"auc-{ep}-5",
            requester_type="commercial",
            requester_id=f"telco-{rng.randint(100, 999)}",
            bandwidth_needed_mhz=rng.choice([10.0, 20.0]),
            preferred_band_index=lic_band,
            priority=5,                      # lowest priority
            duration_steps=rng.randint(2, 4),
            power_dbm=round(rng.uniform(25.0, 35.0), 1),
            description=(f"Low-priority commercial P5 requesting Band {lic_band}. "
                         f"Military demand visible in queue — consider deferring."),
            gt_best_band_index=-1,           # reject: preserve spectrum for incoming military
            gt_acceptable_bands=[],
            gt_max_power_dbm=0.0,
            gt_should_preempt=False,
            gt_reject_reason="Reserve band for anticipated high-priority traffic",
        ))

        # Step 6: Military P1 commandeers the licensed band
        episode.append(ScenarioRequest(
            request_id=f"auc-{ep}-6",
            requester_type="military",
            requester_id=f"mil-{rng.randint(10, 99)}",
            bandwidth_needed_mhz=20.0,
            preferred_band_index=lic_band,   # same band commercial wanted
            priority=1,
            duration_steps=rng.randint(4, 8),
            power_dbm=43.0,
            description=(f"Military requires Band {lic_band} for secure operations. "
                         f"Immediate exclusive access with preemption authority."),
            gt_best_band_index=lic_band,
            gt_acceptable_bands=[lic_band],
            gt_max_power_dbm=43.0,
            gt_should_preempt=True,
            gt_reject_reason=None,
        ))

        # ── Pattern D: Standard filler (step 7) ──────────────────────
        filler_type = rng.choice(["commercial", "iot"])
        if filler_type == "commercial":
            filler_band = rng.choice([2, 4, 6])
            filler_bw = rng.choice([10.0, 16.0])
            filler_pwr = rng.uniform(28.0, 38.0)
            filler_desc = f"Standard commercial request for {filler_bw} MHz."
        else:
            filler_band = rng.choice([8, 11])
            filler_bw = rng.choice([5.0, 10.0])
            filler_pwr = rng.uniform(10.0, 14.0)
            filler_desc = f"Standard IoT request for {filler_bw} MHz in unlicensed band."

        episode.append(ScenarioRequest(
            request_id=f"auc-{ep}-7",
            requester_type=filler_type,
            requester_id=f"{filler_type}-{rng.randint(100, 999)}",
            bandwidth_needed_mhz=filler_bw,
            preferred_band_index=filler_band,
            priority=rng.choice([3, 4]),
            duration_steps=rng.randint(2, 4),
            power_dbm=round(filler_pwr, 1),
            description=filler_desc,
            gt_best_band_index=filler_band,
            gt_acceptable_bands=[filler_band],
            gt_max_power_dbm=SPECTRUM_GRID[filler_band].max_power_dbm,
            gt_should_preempt=False,
            gt_reject_reason=None,
        ))

        scenarios.append(episode)
    return scenarios


# ── Public API ───────────────────────────────────────────────────────

TASK_REGISTRY = {
    "easy": {
        "builder": _build_easy_scenarios,
        "rules": REGULATORY_RULES_BASE,
        "description": "Low-occupancy spectrum with straightforward allocation requests.",
        "steps_per_episode": 5,
    },
    "medium": {
        "builder": _build_medium_scenarios,
        "rules": REGULATORY_RULES_BASE,
        "description": "Moderate occupancy with priority conflicts and guard band awareness.",
        "steps_per_episode": 8,
    },
    "disaster_response": {
        "builder": _build_disaster_response_scenarios,
        "rules": REGULATORY_RULES_DISASTER,
        "description": (
            "Natural disaster scenario: normal operations interrupted by cascading "
            "emergency requests requiring preemption and priority overrides."
        ),
        "steps_per_episode": 10,
    },
    "hard": {
        "builder": _build_hard_scenarios,
        "rules": REGULATORY_RULES_HARD,
        "description": (
            "Dense urban spectrum: cascading preemptions, cognitive radio dynamics, "
            "military operations."
        ),
        "steps_per_episode": 12,
    },
    "spectrum_auction": {
        "builder": _build_spectrum_auction_scenarios,
        "rules": REGULATORY_RULES_AUCTION,
        "description": (
            "Forward-looking allocation with queue visibility: agent sees upcoming "
            "requests and must plan globally optimal assignments."
        ),
        "steps_per_episode": 8,
    },
}


def get_scenarios(task_name: str, seed: int | None = None) -> List[List[ScenarioRequest]]:
    """
    Return the list of episodes for a given task.

    Args:
        task_name: One of the keys in TASK_REGISTRY.
        seed: Optional RNG seed override. If None, the builder's default seed is used.

    Returns:
        List of episodes, each episode being an ordered list of ScenarioRequests.
    """
    if task_name not in TASK_REGISTRY:
        raise ValueError(
            f"Unknown task: {task_name!r}. Choose from {list(TASK_REGISTRY.keys())}"
        )
    builder = TASK_REGISTRY[task_name]["builder"]
    return builder(seed=seed) if seed is not None else builder()


def get_rules(task_name: str) -> List[str]:
    """Return the list of regulatory rule strings for a given task."""
    return TASK_REGISTRY[task_name]["rules"]


def get_spectrum_grid() -> List[Dict[str, Any]]:
    """Return the spectrum grid as a list of serializable dicts, one per band."""
    return [
        {
            "index": i,
            "start_mhz": b.start_mhz,
            "end_mhz": b.end_mhz,
            "bandwidth_mhz": b.end_mhz - b.start_mhz,
            "label": b.label,
            "band_type": b.band_type,
            "max_power_dbm": b.max_power_dbm,
            "guard_band_mhz": b.guard_band_mhz,
        }
        for i, b in enumerate(SPECTRUM_GRID)
    ]


# ═══════════════════════════════════════════════════════════════════════
#  Multi-agent (Round 2) scenario generators
# ═══════════════════════════════════════════════════════════════════════
#
# These generators produce scenarios for three game-theoretic tasks:
#
#   * Task A — sealed-bid auction with partial observability
#   * Task B — one-shot dispute-resolution with opponent-type inference
#   * Task C — iterated prisoner's dilemma with a reputation-tracking referee
#
# All ground-truth computation uses abstract game-theoretic constructs
# (Bayesian Nash approximations, best-response over posterior, cooperative
# equilibrium simplifications). Telecom flavor is confined to per-scenario
# description strings. See docs/multi_agent_design.md for the full spec.


# ── Archetype rotation (shared across all Round 2 generators) ────────

_ARCHETYPES: List[str] = ["Aggressive", "Conservative", "Mimicking"]


def _rotate_archetypes(seed: int, num_slots: int) -> List[str]:
    """Deterministic assignment of opponent archetypes to competitor slots.

    Produces a rotation of the archetype list that varies with seed so
    that, across the training and evaluation seed ranges, each archetype
    appears in each slot position in roughly equal proportion. The
    returned list has length `num_slots`.
    """
    base = list(_ARCHETYPES)
    rot = seed % len(base)
    ordered = base[rot:] + base[:rot]
    if seed % 2 == 0:
        ordered = list(reversed(ordered))
    return ordered[:num_slots]


# ── Scenario / GroundTruth data containers ───────────────────────────


@dataclass
class Scenario:
    """Generic container for a Round 2 scenario.

    Holds the task identifier, RNG seed used to generate it, any static
    parameters specific to the task (budgets, valuations, payoff matrix,
    etc.), per-slot opponent archetypes (not exposed to the learner),
    and flavor text.
    """
    task_name: str
    seed: int
    num_rounds: int
    params: Dict[str, Any]
    opponent_archetypes: List[str]
    description: str


@dataclass
class GroundTruth:
    """Ground-truth reference outputs for grading a Round 2 episode.

    `optimal_actions` is an ordered per-round list of reference action
    payloads. `notes` holds free-form metadata explaining the solution
    method (e.g. which equilibrium approximation was used).
    """
    optimal_actions: List[Dict[str, Any]]
    notes: Dict[str, Any] = field(default_factory=dict)


# ── Task A: sealed-bid auction ───────────────────────────────────────

def _first_price_sbne_bid(valuation: float, num_bidders: int) -> float:
    """Symmetric Bayesian Nash equilibrium bid for first-price sealed-bid
    auctions with independent uniform-[0, v_max] private values.

    Under the symmetric BNE of a first-price sealed-bid auction with
    `n` risk-neutral bidders and uniform priors, the equilibrium bid is:

        b_i(v_i) = v_i * (n - 1) / n

    This is a standard textbook result (see, e.g., Krishna, Auction
    Theory, §2.3). We use this as our ground-truth approximation rather
    than solving the full 3-player correlated-valuation game, which is
    not tractable inside a generator.
    """
    if num_bidders <= 1:
        return valuation
    return valuation * (num_bidders - 1) / num_bidders


def generate_auction_scenario(seed: int) -> Tuple[Scenario, GroundTruth]:
    """Generate a sealed-bid auction scenario with partial observability.

    Game structure:
      * 3 strategic players (learner + 2 scripted opponents)
      * 4 indivisible resource items contested over 6 sequential rounds
      * Per-player budgets and per-round valuations are fixed per seed
      * Each round is a first-price sealed-bid auction for one item
      * Bids are revealed to all players after each round

    Ground truth:
      An ordered list of 6 bids computed by the symmetric Bayesian Nash
      equilibrium approximation `b = v * (n-1) / n`, with a light
      backward-induction adjustment that *reduces* late-round bids when
      remaining supply exceeds the learner's remaining demand (i.e. the
      auction is resolving in the learner's favor and there is no point
      over-paying). This is a documented approximation, not the true
      Nash equilibrium.
    """
    rng = random.Random(seed * 1_000_003 + 17)

    num_players = 3
    num_licenses = 4
    num_rounds = 4

    # Deterministic budgets per player (learner is player 0).
    budgets = [round(50.0 + rng.uniform(0.0, 50.0), 2) for _ in range(num_players)]

    # Per-round private valuations for the learner, in [5, 20].
    learner_valuations = [round(5.0 + rng.uniform(0.0, 15.0), 2) for _ in range(num_rounds)]

    # Opponent archetypes assigned via deterministic rotation.
    archetypes = _rotate_archetypes(seed, num_players - 1)

    params: Dict[str, Any] = {
        "num_players": num_players,
        "num_licenses": num_licenses,
        "budgets": budgets,
        "learner_valuations": learner_valuations,
        "learner_budget": budgets[0],
    }

    # Backward-induction adjustment: remaining supply vs remaining demand.
    # If by round k the learner has already won `won_k` items, its
    # remaining demand is max(0, num_licenses_desired - won_k). When
    # remaining rounds exceed that, reduce the bid linearly.
    desired = min(num_licenses, num_rounds // 2 + 1)  # the learner wants up to `desired` items

    optimal_actions: List[Dict[str, Any]] = []
    running_budget = budgets[0]
    for r in range(num_rounds):
        v = learner_valuations[r]
        sbne_bid = _first_price_sbne_bid(v, num_players)
        # Light backward-induction scaling:
        rounds_left = num_rounds - r
        demand_left = max(1, desired)  # avoid division by zero; at least 1 item still desired
        if rounds_left > demand_left:
            scale = demand_left / rounds_left
            adjusted = sbne_bid * scale
        else:
            adjusted = sbne_bid
        adjusted = min(adjusted, running_budget)
        adjusted = round(max(0.0, adjusted), 2)
        optimal_actions.append({"bid_amount": adjusted, "round_index": r})
        # Assume the reference bidder wins on average `desired / num_rounds`
        # rounds; we do not simulate a winner here — we only need a
        # self-consistent reference path for budget accounting.
        expected_spend = adjusted * (desired / num_rounds)
        running_budget = max(0.0, running_budget - expected_spend)

    description = (
        f"Sealed-bid auction, 3 operators bidding over 4 rounds for 4 spectrum licenses "
        f"(flavor: CBRS PAL). Each round reveals the winning bid publicly. "
        f"Learner budget: {budgets[0]}; total rounds: {num_rounds}."
    )

    scenario = Scenario(
        task_name="auction",
        seed=seed,
        num_rounds=num_rounds,
        params=params,
        opponent_archetypes=archetypes,
        description=description,
    )
    gt = GroundTruth(
        optimal_actions=optimal_actions,
        notes={
            "method": "symmetric_bne_first_price_with_backward_induction_supply_adjustment",
            "formula": "b = v * (n-1)/n, scaled down by demand_left/rounds_left when supply > demand",
            "approximation": True,
        },
    )
    return scenario, gt


# ── Task B: one-shot dispute resolution ──────────────────────────────

# Payoff matrix for the dispute-resolution game.
#   rows    = learner action (concede, negotiate, escalate, audit)
#   columns = opponent archetype (Aggressive, Conservative, Mimicking)
#   entries = learner's expected payoff given the archetype's mixed strategy.
_DISPUTE_PAYOFFS: Dict[str, Dict[str, float]] = {
    "concede":   {"Aggressive":  0.1, "Conservative": 0.3, "Mimicking":  0.3},
    "negotiate": {"Aggressive":  0.2, "Conservative": 0.8, "Mimicking":  0.7},
    "escalate":  {"Aggressive":  0.3, "Conservative": 0.9, "Mimicking":  0.5},
    "audit":     {"Aggressive":  0.5, "Conservative": 0.4, "Mimicking":  0.5},
}

_DISPUTE_ACTIONS: List[str] = ["concede", "negotiate", "escalate", "audit"]


def generate_dispute_scenario(seed: int) -> Tuple[Scenario, GroundTruth]:
    """Generate a one-shot dispute-resolution scenario.

    Game structure:
      * 2 players (learner + 1 scripted opponent)
      * Single simultaneous-move stage game
      * Learner chooses from {concede, negotiate, escalate, audit}
      * Opponent plays a mixed strategy drawn from its archetype

    Ground truth:
      The learner's best response is the action that maximizes expected
      payoff given a uniform prior over opponent archetypes, computed as
      `argmax_a (1/K) * sum_t payoff(a, t)` where K is the number of
      archetypes. The prior is uniform because this is the *first* round
      of play — by construction, no observations of this opponent exist
      yet. Later stages of episode play (out of scope for this prompt)
      could update the posterior from observed play.
    """
    rng = random.Random(seed * 7_919 + 101)

    archetypes = _rotate_archetypes(seed, 1)
    opponent_archetype = archetypes[0]

    # Uniform prior over archetypes for best-response computation.
    prior = {a: 1.0 / len(_ARCHETYPES) for a in _ARCHETYPES}

    def expected_payoff(action: str) -> float:
        return sum(prior[t] * _DISPUTE_PAYOFFS[action][t] for t in _ARCHETYPES)

    scored = [(a, expected_payoff(a)) for a in _DISPUTE_ACTIONS]
    best_action = max(scored, key=lambda x: x[1])[0]

    # Per-seed deterministic context to give each seed a distinct scenario.
    band_a = rng.randint(1, 6)
    band_b = band_a + 1
    stake_value = round(rng.uniform(1.0, 10.0), 2)
    audit_cost = round(rng.uniform(0.1, 1.0), 2)
    complaint_id = f"dispute-{seed:05d}-{rng.randint(0, 9_999):04d}"

    description = (
        f"Dispute-resolution game {complaint_id}: learner and 1 competitor "
        f"occupy adjacent bands {band_a} and {band_b} (flavor: interference "
        f"complaint filed, stake {stake_value}). Learner must choose a "
        f"response without knowing the competitor's type; payoff depends on "
        f"whether the competitor is aggressive, conservative, or mimicking."
    )

    params: Dict[str, Any] = {
        "num_players": 2,
        "payoff_matrix": _DISPUTE_PAYOFFS,
        "prior_over_types": prior,
        "expected_payoffs_per_action": {a: expected_payoff(a) for a in _DISPUTE_ACTIONS},
        "adjacent_band_indices": [band_a, band_b],
        "stake_value": stake_value,
        "audit_cost": audit_cost,
        "scenario_id": complaint_id,
    }

    scenario = Scenario(
        task_name="dispute",
        seed=seed,
        num_rounds=1,
        params=params,
        opponent_archetypes=archetypes,
        description=description,
    )
    gt = GroundTruth(
        optimal_actions=[{"dispute_choice": best_action, "round_index": 0}],
        notes={
            "method": "best_response_over_uniform_prior",
            "true_opponent_archetype": opponent_archetype,  # held out — not in observation
        },
    )
    return scenario, gt


# ── Task C: iterated coalition game (stateful) ───────────────────────

# Module-level cache of coalition state keyed by seed. Iterated play
# across repeated calls with the same seed appends to the history.
_COALITION_STATE: Dict[int, Dict[str, Any]] = {}


def _coalition_reset(seed: int) -> None:
    """Drop any persisted state for a coalition seed (testing hook)."""
    _COALITION_STATE.pop(seed, None)


def generate_coalition_scenario(seed: int) -> Tuple[Scenario, GroundTruth]:
    """Generate an iterated coalition (prisoner's-dilemma-style) scenario.

    Game structure:
      * 3 players total, learner + 2 scripted opponents
      * Each episode is one stage of an iterated game; repeated calls
        with the same seed advance the stage index and retain the
        running reputation
      * Learner chooses from {cooperate, defect, abstain}
      * A reputation-tracking referee updates every player's reputation
        after each stage using the mechanics in docs/multi_agent_design.md §6

    Ground truth simplification:
      The reference action is `cooperate` whenever the learner's current
      reputation is below 0.7; otherwise the reference accepts either
      `cooperate` or `defect`. This is NOT the full cooperative
      equilibrium of the iterated game (which would require folk-theorem
      trigger strategies and full opponent-type inference), but it is a
      tractable and well-defined benchmark for hackathon scope.
    """
    rng = random.Random(seed * 314_159 + 2_718)

    state = _COALITION_STATE.get(seed)
    if state is None:
        state = {
            "stage_index": 0,
            "learner_reputation": 0.5,
            "action_history": [],
        }
        _COALITION_STATE[seed] = state

    archetypes = _rotate_archetypes(seed, 2)

    rep = state["learner_reputation"]

    if rep < 0.7:
        optimal_actions = [{"cooperation_flag": "cooperate", "round_index": state["stage_index"]}]
        gt_notes = {
            "method": "simplification_cooperate_when_reputation_below_0_7",
            "acceptable_actions": ["cooperate"],
        }
    else:
        optimal_actions = [{"cooperation_flag": "cooperate", "round_index": state["stage_index"]}]
        gt_notes = {
            "method": "simplification_cooperate_when_reputation_below_0_7",
            "acceptable_actions": ["cooperate", "defect"],
        }

    severity = rng.choice(["regional flooding", "wildfire", "earthquake", "severe storm"])
    resource_pool_size = round(rng.uniform(10.0, 50.0), 2)
    per_player_demand = round(rng.uniform(2.0, 8.0), 2)
    emergency_id = f"coalition-{seed:05d}-{rng.randint(0, 9_999):04d}"

    description = (
        f"Iterated coalition game {emergency_id}, stage {state['stage_index']}: "
        f"3 operators asked to share a common resource pool of size "
        f"{resource_pool_size} (flavor: emergency spectrum sharing during "
        f"{severity}). Current learner reputation: {rep:.2f}. Referee "
        f"updates reputations after each stage."
    )

    params: Dict[str, Any] = {
        "num_players": 3,
        "stage_index": state["stage_index"],
        "learner_reputation": rep,
        "reputation_threshold": 0.7,
        "reputation_delta_defect": -0.10,
        "reputation_delta_cooperate": 0.05,
        "resource_pool_size": resource_pool_size,
        "per_player_demand": per_player_demand,
        "scenario_id": emergency_id,
    }

    scenario = Scenario(
        task_name="coalition",
        seed=seed,
        num_rounds=1,  # one stage exposed per generator call
        params=params,
        opponent_archetypes=archetypes,
        description=description,
    )
    gt = GroundTruth(
        optimal_actions=optimal_actions,
        notes=gt_notes,
    )
    return scenario, gt
