#!/usr/bin/env python3
# Copyright (c) 2026 TRIBUNAL Team.
# Smart Factory Assembly — Inference Agent.

"""
Smart Factory Assembly — LLM-Powered Inference Agent.

Uses Meta's Llama-3.2-1B-Instruct model via HuggingFace Router to solve
the Smart Factory environment across 3 difficulty tiers.

The agent uses Chain-of-Thought reasoning with structured JSON output
to handle complex grid navigation, part picking, and delivery ordering.

Usage:
    export API_KEY=your_hf_token_here
    python inference.py
"""

import json
import os
import sys
import time
import re
import requests
from openai import OpenAI

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

API_BASE_URL = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1/")
API_KEY = os.environ.get("API_KEY")
MODEL_NAME = os.environ.get("MODEL_NAME", "meta-llama/Llama-3.2-1B-Instruct")
ENV_API_URL = os.environ.get(
    "ENV_API_URL", "https://uzaif1-meta-hack-openenv-26.hf.space"
)

if not API_KEY:
    print("❌ ERROR: API_KEY is missing! Run: set API_KEY=your_hf_token")
    sys.exit(1)

client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

# ─────────────────────────────────────────────────────────────────────────────
# Grid cell legend (matches models.py)
# ─────────────────────────────────────────────────────────────────────────────

CELL_LEGEND = {
    0: "empty",
    1: "wall",
    2: "hazard",
    3: "pickup",
    4: "dropoff",
    5: "conveyor",
    6: "robot",
}

# ─────────────────────────────────────────────────────────────────────────────
# System Prompt — Chain-of-Thought with JSON output
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert robotic controller for a factory grid environment.

GRID CELL CODES: 0=empty, 1=wall, 2=hazard, 3=pickup, 4=dropoff, 5=conveyor, 6=robot

ACTIONS:
  0=NOOP, 1=GRAB, 2=PLACE, 3=LEFT, 4=RIGHT, 5=UP, 6=DOWN, 7=WAIT, 8=INSPECT

RULES:
- You can GRAB (1) only when at a pickup station (3) and not carrying anything.
- You can PLACE (2) only when at a dropoff station (4) and carrying a part.
- Moving into a wall (1) causes collision penalty. Don't move into walls.
- Hazard zones (2) cost a penalty each step you're on them. Avoid them.
- Conveyors (5) may push you. Be careful.
- If carrying=0, navigate to the nearest pickup station.
- If carrying=1, navigate to the nearest dropoff station.
- Prefer shortest Manhattan-distance path while avoiding walls and hazards.

THINK STEP BY STEP:
1. Note your position (y, x) and carrying status.
2. Identify your target (pickup if not carrying, dropoff if carrying).
3. Plan the best move toward the target, avoiding walls (1) and hazards (2).
4. Output your decision.

OUTPUT FORMAT — respond with ONLY a JSON object, nothing else:
{"thought": "brief reasoning", "action": <integer 0-8>}
"""


# ─────────────────────────────────────────────────────────────────────────────
# LLM Action Selection
# ─────────────────────────────────────────────────────────────────────────────


def get_ai_action(observation: dict) -> int:
    """Query the LLM for an action given the current observation."""
    y, x = observation["robot_pos"]
    carrying = observation["carrying"]
    carrying_type = observation.get("carrying_type", None)
    grid = observation.get("grid_layout", [])
    time_remaining = observation.get("time_remaining", 0)
    inventory = observation.get("inventory", {})
    deliveries_made = observation.get("deliveries_made", 0)
    deliveries_required = observation.get("deliveries_required", 1)
    distance = observation.get("distance_to_target", 0)

    # Build a compact grid representation (show nearby 5x5 window for large grids)
    grid_size = len(grid) if grid else 0
    if grid_size <= 7:
        grid_str = "\n".join(str(row) for row in grid)
    else:
        # Show 5x5 window around robot
        y_min = max(0, y - 2)
        y_max = min(grid_size, y + 3)
        x_min = max(0, x - 2)
        x_max = min(grid_size, x + 3)
        window = [row[x_min:x_max] for row in grid[y_min:y_max]]
        grid_str = f"5x5 window around robot (full grid is {grid_size}x{grid_size}):\n"
        grid_str += "\n".join(str(row) for row in window)

    # Identify pickup and dropoff locations in grid
    pickups = []
    dropoffs = []
    for gy in range(grid_size):
        for gx in range(grid_size):
            if grid[gy][gx] == 3:
                pickups.append(f"({gy},{gx})")
            elif grid[gy][gx] == 4:
                dropoffs.append(f"({gy},{gx})")

    user_msg = (
        f"STATE:\n"
        f"  Position: y={y}, x={x}\n"
        f"  Carrying: {carrying} (type: {carrying_type})\n"
        f"  Time remaining: {time_remaining}\n"
        f"  Deliveries: {deliveries_made}/{deliveries_required}\n"
        f"  Distance to target: {distance}\n"
        f"  Inventory needed: {inventory}\n"
        f"  Pickups at: {', '.join(pickups)}\n"
        f"  Dropoffs at: {', '.join(dropoffs)}\n"
        f"  Grid:\n{grid_str}\n\n"
        f"Choose your action as JSON:"
    )

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            max_tokens=200,
            temperature=0.01,
        )
        ai_text = response.choices[0].message.content.strip()
        print(f"🤖 AI: {ai_text}")

        # Parse JSON
        action = _parse_action(ai_text)
        return action

    except Exception as e:
        print(f"⚠️ LLM Error: {e}")
        return _fallback_action(observation)


def _parse_action(text: str) -> int:
    """Extract action integer from LLM response (JSON or fallback)."""
    # Try JSON parse
    try:
        # Find JSON object in text
        match = re.search(r'\{[^}]+\}', text)
        if match:
            data = json.loads(match.group())
            action = int(data.get("action", 0))
            if 0 <= action <= 8:
                return action
    except (json.JSONDecodeError, ValueError, TypeError):
        pass

    # Fallback: look for action number patterns
    match = re.search(r'"action"\s*:\s*(\d)', text)
    if match:
        return int(match.group(1))

    match = re.search(r'Action:\s*\[?(\d)\]?', text, re.IGNORECASE)
    if match:
        return int(match.group(1))

    # Last resort: any single digit
    digits = re.findall(r'\d', text)
    if digits:
        d = int(digits[-1])
        if 0 <= d <= 8:
            return d

    return 0  # NOOP


def _fallback_action(observation: dict) -> int:
    """Deterministic fallback when LLM fails — simple heuristic."""
    y, x = observation["robot_pos"]
    carrying = observation["carrying"]
    grid = observation.get("grid_layout", [])
    grid_size = len(grid) if grid else 5

    if carrying == 0:
        # Move toward (0,0) — default pickup
        if y > 0:
            return 5  # UP
        if x > 0:
            return 3  # LEFT
        return 1  # GRAB
    else:
        # Move toward bottom-right — default dropoff
        if y < grid_size - 1:
            return 6  # DOWN
        if x < grid_size - 1:
            return 4  # RIGHT
        return 2  # PLACE


# ─────────────────────────────────────────────────────────────────────────────
# Main Loop
# ─────────────────────────────────────────────────────────────────────────────


def main():
    """Run the inference agent against all 3 difficulty tiers."""

    # Handshake with LLM (warm up connection)
    try:
        client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": "ready"}],
            max_tokens=1,
        )
    except Exception:
        pass

    tasks = ["smart_factory_easy", "smart_factory_medium", "smart_factory_hard"]

    for task_name in tasks:
        print(f"\n{'='*60}")
        print(f"[START] task={task_name}", flush=True)
        print(f"{'='*60}")

        # Reset environment
        try:
            payload = {"task": task_name}
            res = requests.post(f"{ENV_API_URL}/reset", json=payload, timeout=15)
            res.raise_for_status()
            result = res.json()
            observation = result["observation"]
        except Exception as e:
            print(f"❌ Failed to reset: {e}")
            print(f"[END] task={task_name} score=0.01 steps=0", flush=True)
            continue

        step_count = 0
        done = False
        total_reward = 0.0
        max_steps = 120  # safety cap

        while not done and step_count < max_steps:
            step_count += 1
            action = get_ai_action(observation)

            try:
                step_payload = {
                    "action": {"action": action},
                }
                res = requests.post(
                    f"{ENV_API_URL}/step",
                    json=step_payload,
                    timeout=15,
                )
                result = res.json()

                observation = result["observation"]
                reward = result.get("reward", 0)
                done = result.get("done", False)
                total_reward += reward

                print(
                    f"  [STEP] step={step_count} action={action} "
                    f"reward={reward:.3f} total={total_reward:.3f} "
                    f"pos={observation.get('robot_pos')} "
                    f"carrying={observation.get('carrying')} "
                    f"deliveries={observation.get('deliveries_made', 0)}/{observation.get('deliveries_required', 1)}",
                    flush=True,
                )

            except Exception as e:
                print(f"  ⚠️ API Error: {e}")
                break

            time.sleep(0.05)

        # Normalize score
        raw_score = (total_reward + 5.0) / 15.0
        normalized_score = max(0.01, min(0.99, raw_score))

        print(f"\n📊 Results: total_reward={total_reward:.3f} steps={step_count}")
        print(f"[END] task={task_name} score={normalized_score:.4f} steps={step_count}", flush=True)


if __name__ == "__main__":
    main()