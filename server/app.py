"""
RF Spectrum Allocation - FastAPI Server
"""

import asyncio
import os
import sys
from pathlib import Path
from typing import Optional

# Ensure project root is on the path so models/scenarios are importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from openenv.core.env_server import create_fastapi_app

from models import (
    CooperationChoice,
    DisputeChoice,
    MultiAgentAction,
    MultiAgentObservation,
    SpectrumAction,
    SpectrumObservation,
)
from server.spectrum_environment import (
    SpectrumEnvironment,
    _MULTI_AGENT_ROUNDS,
    _MULTI_AGENT_TASKS,
)

# Reuse the *exact* prompt-formatting and JSON-parsing logic from training /
# eval. Any drift here means the trained model generates garbage actions.
from inference import (
    _describe_multi_agent,
    _parse_multi_agent_action,
)

app = create_fastapi_app(SpectrumEnvironment, MultiAgentAction, MultiAgentObservation)

# The OpenEnv harness creates environment instances lazily per-session.
# For endpoints that need access to "the most recent" episode (the demo
# oversight view), we maintain a shared singleton that the REST endpoints
# share with HTTP clients.
_shared_env = SpectrumEnvironment()
_last_obs: Optional[MultiAgentObservation] = None  # cached for trained-step prompt building


def _call_reset(**kwargs) -> MultiAgentObservation:
    global _last_obs
    _last_obs = _shared_env.reset(**kwargs)
    return _last_obs


def _call_step(action: MultiAgentAction) -> MultiAgentObservation:
    global _last_obs
    _last_obs = _shared_env.step(action)
    return _last_obs

# Static assets for the /visualize SPA. The directory is committed in this
# repo at server/static/ and copied into the container by the Dockerfile's
# `COPY . .` step.
_STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")


# ── Trained-model loader (eager at startup) ───────────────────────────
#
# The hackathon demo wires a Qwen2.5-0.5B-Instruct + LoRA checkpoint into
# the visualizer's "Run trained" path. The model is loaded eagerly in a
# worker thread so /health stays responsive during the 30-60s cold start;
# until it's ready, /api/trained_step returns 503 with detail "model
# loading" and the frontend shows a "warming up" ribbon.

_BASE_MODEL = os.environ.get("TRAINED_BASE_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
_ADAPTER_REPO = os.environ.get("TRAINED_ADAPTER_REPO", "ren9087/rf-spectrum-auction-trained")

_model_state: dict = {
    "tokenizer": None,
    "model": None,
    "device": None,
    "status": "loading",  # "loading" | "ready" | "error"
    "error": None,
}

# Verbatim from training/grpo_multiagent_fixed.ipynb cell 16. Do not edit.
SYSTEM_PROMPT_BY_TASK = {
    "auction": (
        "You are a strategic bidder in a sealed-bid auction with two competitors. "
        "Your goal: maximise long-run reward while respecting your remaining budget. "
        "Lower bids preserve budget for future rounds; higher bids increase the chance "
        "of winning this round. Competitor bid history is revealed after each round. "
        'Respond with ONLY a JSON object: {"bid_amount": <float>, "justification": "<str>"}.'
    ),
    "dispute": (
        "You are a player in a one-shot dispute game. Pick one of "
        '{"concede","negotiate","escalate","audit"}. Payoff depends on your '
        "opponent's type, which you must infer from observable behavior. "
        'Respond with ONLY a JSON object: {"dispute_choice": "<one of four>", "justification": "<str>"}.'
    ),
    "coalition": (
        "You are a player in an iterated coalition game with a reputation-tracking referee. "
        'Pick one of {"cooperate","defect","abstain"}. Cooperating raises your reputation; '
        "defecting lowers it and risks a regulator warning if your reputation is high. "
        'Respond with ONLY a JSON object: {"cooperation_flag": "<one of three>", "justification": "<str>"}.'
    ),
}


def _load_trained_model_blocking() -> None:
    """Eager loader called from startup. Runs in a worker thread."""
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel

        tok = AutoTokenizer.from_pretrained(_ADAPTER_REPO, trust_remote_code=True)
        base = AutoModelForCausalLM.from_pretrained(
            _BASE_MODEL,
            device_map="auto",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(base, _ADAPTER_REPO)
        model.eval()
        _model_state.update(
            tokenizer=tok,
            model=model,
            device=next(model.parameters()).device,
            status="ready",
        )
    except Exception as exc:
        _model_state.update(status="error", error=f"{type(exc).__name__}: {exc}")


def _trained_action_for(task_name: str, obs: MultiAgentObservation) -> tuple[MultiAgentAction, str]:
    """Run the trained model on `obs` and return (parsed_action, raw_text)."""
    import torch

    tok = _model_state["tokenizer"]
    model = _model_state["model"]
    device = _model_state["device"]

    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT_BY_TASK[task_name]},
        {"role": "user", "content": _describe_multi_agent(obs, task_name)},
    ]
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    inputs = tok(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            pad_token_id=tok.eos_token_id,
        )
    raw = tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return _parse_multi_agent_action(raw, task_name), raw


def _baseline_fallback_action(task_name: str) -> MultiAgentAction:
    """Trivial fallback used when the trained model isn't applicable (dispute / coalition)."""
    if task_name == "auction":
        return MultiAgentAction(bid_amount=12.0, justification="baseline fallback: small fixed bid")
    if task_name == "dispute":
        return MultiAgentAction(dispute_choice=DisputeChoice.NEGOTIATE,
                                justification="baseline fallback: always negotiate")
    return MultiAgentAction(cooperation_flag=CooperationChoice.COOPERATE,
                            justification="baseline fallback: always cooperate")


@app.on_event("startup")
async def _kickoff_model_load() -> None:
    """Kick off the trained-model load in a worker thread so the event loop stays free."""
    asyncio.get_event_loop().run_in_executor(None, _load_trained_model_blocking)


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
            "trained_step": "POST /api/trained_step",
            "model_status": "GET /api/model_status",
            "env_reset": "POST /env/reset",
            "env_step": "POST /env/step",
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
    _call_reset(task_name=task_name, seed=seed, episode_index=seed)
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
    obs = _call_step(action)
    return {
        "done": bool(getattr(obs, "done", False)),
        "reward": getattr(obs, "reward", None),
        "round_index": int(_shared_env.state.step_count),
    }


# ── /env/* — renya-test-parity wrappers ──────────────────────────────
#
# These are thin equivalents of /api/start_episode and /api/run_step that
# return the full observation in the response body (matching the shape any
# external client built against the renya-test branch expects). The
# visualizer keeps using /api/* and is unaffected.


@app.post("/env/reset")
async def env_reset(payload: dict = {}) -> dict:
    """Stateful reset using the shared environment (renya-test parity)."""
    kwargs = {}
    seed = payload.get("seed")
    if seed is not None:
        kwargs["seed"] = int(seed)
    task_name = payload.get("task_name")
    if task_name is not None:
        kwargs["task_name"] = task_name
    obs = _call_reset(**kwargs)
    return {"observation": obs.model_dump(), "reward": None, "done": False}


@app.post("/env/step")
async def env_step(payload: dict = {}) -> dict:
    """Stateful step using the shared environment (renya-test parity)."""
    action_data = payload.get("action", payload)
    try:
        action = MultiAgentAction(**action_data)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"invalid action: {exc}")
    obs = _call_step(action)
    return {
        "observation": obs.model_dump(),
        "reward": obs.reward,
        "done": obs.done,
    }


# ── /api/trained_step — drives env with the trained LoRA ─────────────


@app.post("/api/trained_step")
async def api_trained_step(payload: dict) -> dict:
    """Generate an action with the trained LoRA, then advance the env one step.

    Body: ``{"task_name": "auction" | "dispute" | "coalition"}``.
    For dispute/coalition we fall back to the baseline fixed action because
    those tasks showed no training-time improvement.
    """
    task_name = str(payload.get("task_name", "auction"))
    if task_name not in _MULTI_AGENT_TASKS:
        raise HTTPException(
            status_code=400,
            detail=f"task_name must be one of {sorted(_MULTI_AGENT_TASKS)}",
        )
    if _shared_env._task_name not in _MULTI_AGENT_TASKS:
        raise HTTPException(status_code=400, detail="no multi-agent episode active")

    if task_name != "auction":
        action = _baseline_fallback_action(task_name)
        obs = _call_step(action)
        return {
            "done": bool(obs.done),
            "reward": obs.reward,
            "round_index": int(_shared_env.state.step_count),
            "mode": "baseline_fallback",
            "raw_text": "",
            "action": action.model_dump(),
        }

    if _model_state["status"] == "loading":
        raise HTTPException(status_code=503, detail="model loading")
    if _model_state["status"] == "error":
        raise HTTPException(status_code=503, detail=f"model unavailable: {_model_state['error']}")

    if _last_obs is None:
        raise HTTPException(status_code=400, detail="no observation cached; call /api/start_episode or /env/reset first")

    # Generation runs synchronously on the worker thread; the per-call ~0.3s
    # on T4 is small enough we don't need to thread-pool it. If we ever go
    # back to CPU and steps take >1s, wrap this in run_in_executor.
    action, raw = _trained_action_for(task_name, _last_obs)
    stepped = _call_step(action)
    return {
        "done": bool(stepped.done),
        "reward": stepped.reward,
        "round_index": int(_shared_env.state.step_count),
        "mode": "trained",
        "raw_text": raw,
        "action": action.model_dump(),
    }


@app.get("/api/model_status")
async def api_model_status() -> dict:
    """Lightweight status endpoint for the visualizer's 'model warming up' ribbon."""
    return {"status": _model_state["status"], "error": _model_state["error"]}


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
