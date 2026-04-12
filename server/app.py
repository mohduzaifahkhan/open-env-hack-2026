# Copyright (c) 2026 TRIBUNAL Team.
# Smart Factory Assembly — FastAPI Server Application.

"""
FastAPI application for the Smart Factory Environment.

Exposes the SmartFactoryEnvironment over HTTP and WebSocket endpoints
using OpenEnv's ``create_app()`` helper, plus:

- ``GET /``         — visual renderer (web UI)
- ``POST /reset``   — reset endpoint
- ``POST /step``    — step endpoint
- ``GET /health``   — health check
- ``GET /schema``   — JSON schemas for Action / Observation / State
- ``GET /metrics``  — grading rubric metrics
- ``WebSocket /ws`` — persistent session endpoint
- ``WebSocket /ws/demo`` — live demo viewer endpoint

Usage:
    uvicorn server.app:app --host 0.0.0.0 --port 7860
"""

import sys
import os
import json
import asyncio
import pathlib

# Ensure parent directory is on path for model imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from openenv.core.env_server.http_server import create_app

from models import FactoryAction, FactoryObservation
from server.environment import SmartFactoryEnvironment

# Directory for static files
STATIC_DIR = pathlib.Path(__file__).parent / "static"

# Create the OpenEnv HTTP app.
app = create_app(
    SmartFactoryEnvironment,
    FactoryAction,
    FactoryObservation,
    env_name="smart_factory",
)


# ─────────────────────────────────────────────────────────────────────────────
# Visual Renderer (Root Page)
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the visual renderer web UI."""
    html_path = STATIC_DIR / "index.html"
    if html_path.exists():
        return HTMLResponse(html_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>🏭 TRIBUNAL Smart Factory</h1><p>Environment is running.</p>")


# ─────────────────────────────────────────────────────────────────────────────
# Metrics Endpoint
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/metrics")
async def metrics():
    """Return grading rubric metrics from the most recent demo run."""
    return {
        "name": "SmartFactoryAssembly",
        "version": "2.0.0",
        "rubric_weights": {
            "completion_rate": 0.40,
            "efficiency": 0.25,
            "collision_avoidance": 0.15,
            "hazard_avoidance": 0.10,
            "speed_score": 0.10,
        },
        "difficulty_tiers": ["easy", "medium", "hard"],
        "features": [
            "assembly_line_sequence",
            "quality_inspection",
            "grading_rubric",
            "stochastic_events",
        ],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Demo WebSocket — runs a heuristic agent and streams state to viewer
# ─────────────────────────────────────────────────────────────────────────────

def _bfs_next_step(y, x, ty, tx, grid, grid_size):
    """BFS from (y,x) to (ty,tx) avoiding walls. Returns first move action."""
    from collections import deque
    if y == ty and x == tx:
        return None
    visited = {(y, x)}
    queue = deque()
    moves = [(5, -1, 0), (6, 1, 0), (3, 0, -1), (4, 0, 1)]
    for action, dy, dx in moves:
        ny, nx = y + dy, x + dx
        if 0 <= ny < grid_size and 0 <= nx < grid_size and grid[ny][nx] != 1:
            if ny == ty and nx == tx:
                return action
            visited.add((ny, nx))
            queue.append((ny, nx, action))
    while queue:
        cy, cx, first_action = queue.popleft()
        for _, dy, dx in moves:
            ny, nx = cy + dy, cx + dx
            if (ny, nx) not in visited and 0 <= ny < grid_size and 0 <= nx < grid_size:
                if grid[ny][nx] != 1:
                    if ny == ty and nx == tx:
                        return first_action
                    visited.add((ny, nx))
                    queue.append((ny, nx, first_action))
    return None


def _demo_heuristic_action(observation: dict) -> tuple:
    """BFS-powered heuristic agent for demo visualization. Returns (action, thought)."""
    y, x = observation["robot_pos"]
    carrying = observation["carrying"]
    grid = observation.get("grid_layout", [])
    grid_size = len(grid) if grid else 5
    distance = observation.get("distance_to_target", 99)
    metadata = observation.get("metadata", {})
    next_required = metadata.get("next_required")
    pickup_parts_map = metadata.get("pickup_parts", {})

    # At target
    if distance == 0:
        if carrying == 0:
            return 1, "At pickup — grabbing part"
        else:
            return 2, "At dropoff — placing part"

    # Build pickup list from metadata (reliable even when robot covers the cell)
    pickups_from_meta = []
    for pos_str in pickup_parts_map:
        coords = pos_str.split(",")
        pickups_from_meta.append((int(coords[0]), int(coords[1])))

    # Find targets from grid
    pickups = []
    dropoffs = []
    for gy in range(grid_size):
        for gx in range(grid_size):
            if grid[gy][gx] == 3:
                pickups.append((gy, gx))
            elif grid[gy][gx] == 4:
                dropoffs.append((gy, gx))

    all_pickups = list(set(pickups_from_meta + pickups))

    if carrying == 0:
        # Target the correct pickup for the next required part
        if next_required and pickup_parts_map:
            correct_pickups = []
            for pos_str, part_type in pickup_parts_map.items():
                if part_type == next_required:
                    coords = pos_str.split(",")
                    correct_pickups.append((int(coords[0]), int(coords[1])))
            if correct_pickups:
                target = min(correct_pickups, key=lambda p: abs(p[0] - y) + abs(p[1] - x))
            elif all_pickups:
                target = min(all_pickups, key=lambda p: abs(p[0] - y) + abs(p[1] - x))
            else:
                target = (0, 0)
        elif all_pickups:
            target = min(all_pickups, key=lambda p: abs(p[0] - y) + abs(p[1] - x))
        else:
            target = (0, 0)
        target_name = f"pickup ({next_required or 'part'})"
    else:
        if dropoffs:
            target = min(dropoffs, key=lambda p: abs(p[0] - y) + abs(p[1] - x))
        else:
            target = (grid_size - 1, grid_size - 1)
        target_name = "dropoff"

    ty, tx = target

    # Use BFS for wall-aware pathfinding
    bfs_action = _bfs_next_step(y, x, ty, tx, grid, grid_size)
    action_names = {3: "LEFT", 4: "RIGHT", 5: "UP", 6: "DOWN"}

    if bfs_action is not None:
        return bfs_action, f"Moving {action_names.get(bfs_action, '?')} toward {target_name}"

    # Fallback: try any safe direction
    def is_safe(ny, nx):
        return 0 <= ny < grid_size and 0 <= nx < grid_size and grid[ny][nx] != 1

    for act, ny, nx, name in [(5, y-1, x, "UP"), (6, y+1, x, "DOWN"),
                               (3, y, x-1, "LEFT"), (4, y, x+1, "RIGHT")]:
        if is_safe(ny, nx):
            return act, f"Navigating {name} (avoiding obstacle)"

    return 0, "Stuck — waiting"


@app.websocket("/ws/demo")
async def demo_websocket(websocket: WebSocket):
    """Run a demo episode and stream each step to the connected viewer."""
    await websocket.accept()

    try:
        # Wait for client to send difficulty selection
        data = await asyncio.wait_for(websocket.receive_text(), timeout=10.0)
        config = json.loads(data)
        task = config.get("task", "smart_factory_easy")
    except Exception:
        task = "smart_factory_easy"

    max_steps_map = {
        "smart_factory_easy": 50,
        "smart_factory_medium": 75,
        "smart_factory_hard": 120,
    }
    max_steps = max_steps_map.get(task, 50)

    env = SmartFactoryEnvironment()
    obs = env.reset(task=task)
    total_reward = 0.0

    try:
        # Send initial state
        obs_dict = obs.model_dump()
        await websocket.send_text(json.dumps({
            "type": "reset",
            "data": {
                **obs_dict,
                "thought": "Starting episode...",
                "action": -1,
                "total_reward": 0.0,
            }
        }))
        await asyncio.sleep(0.5)

        for step in range(1, max_steps + 1):
            action_val, thought = _demo_heuristic_action(obs_dict)
            obs = env.step(FactoryAction(action=action_val))
            obs_dict = obs.model_dump()
            reward = obs.reward or 0.0
            total_reward += reward

            await websocket.send_text(json.dumps({
                "type": "step",
                "data": {
                    **obs_dict,
                    "thought": thought,
                    "action": action_val,
                    "step": step,
                    "total_reward": round(total_reward, 3),
                }
            }))

            await asyncio.sleep(0.3)  # Animation speed

            if obs.done:
                break

        # Send final results
        rubric = env.get_rubric()
        await websocket.send_text(json.dumps({
            "type": "done",
            "data": {
                "total_reward": round(total_reward, 3),
                "steps": step,
                "deliveries_made": obs.deliveries_made,
                "deliveries_required": obs.deliveries_required,
                "rubric": rubric,
                "assembly_progress": obs.assembly_progress,
            }
        }))

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_text(json.dumps({
                "type": "error",
                "data": {"message": str(e)}
            }))
        except Exception:
            pass
    finally:
        env.close()


def main():
    """Entry point for ``uv run --project . server`` or direct execution."""
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()