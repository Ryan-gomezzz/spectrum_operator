"""
RF Spectrum Allocation - FastAPI Server
"""

import os
import sys
from pathlib import Path

# Ensure project root is on the path so models/scenarios are importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from openenv.core.env_server import create_fastapi_app

from models import SpectrumAction, MultiAgentAction, SpectrumObservation, MultiAgentObservation
from server.spectrum_environment import (
    SpectrumEnvironment,
    _MULTI_AGENT_ROUNDS,
    _MULTI_AGENT_TASKS,
)

app = create_fastapi_app(SpectrumEnvironment, MultiAgentAction, MultiAgentObservation)

# The OpenEnv harness creates environment instances lazily per-session.
# For endpoints that need access to "the most recent" episode (the demo
# oversight view), we maintain a shared singleton that the REST endpoints
# share with HTTP clients.
_shared_env = SpectrumEnvironment()

# Static assets for the /visualize SPA. The directory is committed in this
# repo at server/static/ and copied into the container by the Dockerfile's
# `COPY . .` step.
_STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")


@app.get("/")
async def root():
    """
    Hugging Face Spaces and browsers hit `/` by default. OpenEnv registers
    `/health`, `/docs`, `/reset`, etc., but not `/`, which otherwise returns 404.
    """
    return {
        "name": "rf_spectrum_env",
        "status": "running",
        "openenv": True,
        "endpoints": {
            "health": "/health",
            "metadata": "/metadata",
            "schema": "/schema",
            "docs": "/docs",
            "openapi": "/openapi.json",
            "reset": "POST /reset",
            "step": "POST /step",
            "state": "GET /state",
            "oversight": "GET /oversight",
            "visualize": "GET /visualize",
            "episode_state": "GET /api/episode_state",
            "start_episode": "POST /api/start_episode",
            "run_step": "POST /api/run_step",
            "websocket": "/ws",
        },
    }


@app.get("/oversight")
async def get_oversight_log():
    """Return the current episode's structured oversight events as JSON.

    Used by the demo UI to visualize the scalable-oversight angle: every
    adjudication the regulator makes is logged here with a severity, an
    operator id, an explanation, and a step number. A GET over this
    endpoint is safe to poll at frame rate.
    """
    return {
        "events": _shared_env.get_oversight_log(),
        "episode_id": _shared_env.state.episode_id,
        "task_name": _shared_env.state.task_name,
        "step_count": _shared_env.state.step_count,
    }
@app.post("/env/reset")
async def env_reset(payload: dict = {}):
    """Stateful reset using the shared environment."""
    seed = payload.get("seed")
    task_name = payload.get("task_name")
    kwargs = {}
    if seed is not None: kwargs["seed"] = int(seed)
    if task_name is not None: kwargs["task_name"] = task_name
    obs = _shared_env.reset(**kwargs)
    return {"observation": obs.model_dump(), "reward": None, "done": False}


@app.post("/env/step")
async def env_step(payload: dict = {}):
    """Stateful step using the shared environment."""
    from models import MultiAgentAction
    action_data = payload.get("action", payload)
    action = MultiAgentAction(**action_data)
    obs = _shared_env.step(action)
    return {"observation": obs.model_dump(), "reward": obs.reward, "done": obs.done}


# ── /visualize SPA + JSON snapshot used by the visualizer ─────────────
#
# The visualizer drives _shared_env directly via /api/start_episode and
# /api/run_step, then polls /api/episode_state for the rendered snapshot.
# We do *not* re-use the OpenEnv POST /reset / POST /step routes here:
# those are stateless single-shot calls (the harness builds a fresh
# SpectrumEnvironment per request and closes it afterwards), so they
# cannot maintain a multi-step episode that the GET /api/episode_state
# poll could read from. The OpenEnv contract endpoints remain untouched
# and continue to work identically for inference.py and the training
# pipeline (both of which use direct in-process calls anyway).


@app.get("/visualize", response_class=HTMLResponse)
async def visualize_page() -> HTMLResponse:
    """Serve the single-file Alpine.js visualizer."""
    html_path = _STATIC_DIR / "visualize.html"
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="visualize.html not found")
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


@app.post("/api/start_episode")
async def api_start_episode(payload: dict) -> dict:
    """Start a new multi-agent episode on the shared env.

    Body: ``{"task_name": "auction" | "dispute" | "coalition", "seed": int}``.
    The seed is clamped into the training range [0, 199] for parity with
    the seed distribution the learner has seen.
    """
    task_name = str(payload.get("task_name", "auction"))
    if task_name not in _MULTI_AGENT_TASKS:
        raise HTTPException(
            status_code=400,
            detail=f"task_name must be one of {sorted(_MULTI_AGENT_TASKS)}",
        )
    seed = int(payload.get("seed", 0)) % 200
    _shared_env.reset(task_name=task_name, seed=seed, episode_index=seed)
    return {
        "ok": True,
        "task_name": task_name,
        "seed": seed,
        "episode_id": _shared_env.state.episode_id,
        "total_rounds": int(_MULTI_AGENT_ROUNDS.get(task_name, 1)),
    }


@app.post("/api/run_step")
async def api_run_step(payload: dict) -> dict:
    """Advance the shared env by one step.

    Body is a ``MultiAgentAction`` JSON: any of ``bid_amount``,
    ``dispute_choice``, ``cooperation_flag``, plus ``justification``.
    """
    if _shared_env._task_name not in _MULTI_AGENT_TASKS:
        raise HTTPException(status_code=400, detail="no multi-agent episode active")
    try:
        action = MultiAgentAction(**payload)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"invalid action: {exc}")
    obs = _shared_env.step(action)
    return {
        "done": bool(getattr(obs, "done", False)),
        "reward": getattr(obs, "reward", None),
        "round_index": int(_shared_env.state.step_count),
    }


@app.get("/api/episode_state")
async def api_episode_state() -> dict:
    """JSON snapshot the visualizer polls every ~1.5s."""
    env = _shared_env
    if not env._operator_states or env._task_name not in _MULTI_AGENT_TASKS:
        return {"status": "idle"}

    learner = env._operator_states.get("op-0")
    competitors = [
        {"slot": i + 1, "bid_history": list(hist)}
        for i, hist in enumerate(env._competitor_bid_history)
    ]

    last_components = env._multi_round_rewards[-1] if env._multi_round_rewards else {}
    last_total = env._step_rewards[-1] if env._step_rewards else 0.0

    return {
        "round_index": int(env._state.step_count),
        "total_rounds": int(_MULTI_AGENT_ROUNDS.get(env._task_name, 1)),
        "task_name": env._task_name,
        "agent": {
            "operator_id": learner.operator_id if learner else "op-0",
            "remaining_budget": float(learner.budget) if learner else 0.0,
            "reputation": float(learner.reputation) if learner else 0.5,
            "licenses_held": list(learner.licenses_held) if learner else [],
            "last_action": (
                learner.action_history[-1]
                if learner and learner.action_history
                else None
            ),
        },
        "competitors": competitors,
        "oversight_events": env.get_oversight_log()[-10:],
        "last_step_rewards": {
            "revenue": float(last_components.get("revenue", 0.0)),
            "interference": float(last_components.get("interference", 0.0)),
            "compliance": float(last_components.get("compliance", 0.0)),
            "justification": float(last_components.get("justification", 0.0)),
            "total": float(last_total),
        },
    }


def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
