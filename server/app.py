"""
RF Spectrum Allocation - FastAPI Server
"""

import os
import sys

# Ensure project root is on the path so models/scenarios are importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openenv.core.env_server import create_fastapi_app

from models import SpectrumAction, SpectrumObservation
from server.spectrum_environment import SpectrumEnvironment

app = create_fastapi_app(SpectrumEnvironment, SpectrumAction, SpectrumObservation)

# The OpenEnv harness creates environment instances lazily per-session.
# For endpoints that need access to "the most recent" episode (the demo
# oversight view), we maintain a shared singleton that the REST endpoints
# share with HTTP clients.
_shared_env = SpectrumEnvironment()


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


def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
