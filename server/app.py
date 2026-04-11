# Copyright (c) 2026 TRIBUNAL Team.
# Smart Factory Assembly — FastAPI Server Application.

"""
FastAPI application for the Smart Factory Environment.

Exposes the SmartFactoryEnvironment over HTTP and WebSocket endpoints
using OpenEnv's ``create_app()`` helper.  This provides:

- ``POST /reset``  — reset endpoint
- ``POST /step``   — step endpoint
- ``GET /health``  — health check
- ``GET /schema``  — JSON schemas for Action / Observation / State
- ``WebSocket /ws`` — persistent session endpoint
- Web interface (when ``ENABLE_WEB_INTERFACE=true``)

Usage:
    uvicorn server.app:app --host 0.0.0.0 --port 7860
"""

import sys
import os

# Ensure parent directory is on path for model imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from openenv.core.env_server.http_server import create_app
except ImportError:
    from openenv.core.env_server.http_server import create_app

from models import FactoryAction, FactoryObservation
from server.environment import SmartFactoryEnvironment

# Create the OpenEnv HTTP app.
# Pass the CLASS (factory) — not an instance — so each WebSocket session
# gets its own isolated environment instance.
app = create_app(
    SmartFactoryEnvironment,
    FactoryAction,
    FactoryObservation,
    env_name="smart_factory",
)


def main():
    """Entry point for ``uv run --project . server`` or direct execution."""
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()