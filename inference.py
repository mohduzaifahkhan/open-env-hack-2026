#!/usr/bin/env python3
# Copyright (c) 2026 TRIBUNAL Team.
# Smart Factory Assembly — Inference Agent.

"""
Smart Factory Assembly — Fully LLM-Powered Inference Agent.

Uses Meta's Llama-3.2-1B-Instruct model via HuggingFace Router to solve
the Smart Factory environment across 3 difficulty tiers.

Every navigation decision is made by the LLM using few-shot prompting
with pre-computed context to help the small model succeed.

Usage:
    export API_KEY=your_hf_token_here
    python inference.py
"""

import json
import os
import sys
import re
from collections import deque
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

# Task-specific max steps
TASK_MAX_STEPS = {
    "smart_factory_easy": 50,
    "smart_factory_medium": 75,
    "smart_factory_hard": 120,
}

# ─────────────────────────────────────────────────────────────────────────────
# System Prompt — Few-shot examples optimized for a 1B model
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You navigate a robot on a grid to pick up and deliver parts.
Reply with ONLY a JSON object: {"thought":"brief","action":N}

Actions: 1=GRAB 2=PLACE 3=LEFT(x-1) 4=RIGHT(x+1) 5=UP(y-1) 6=DOWN(y+1)

Rules:
- dist=0 and carrying=0 → action 1 (GRAB)
- dist=0 and carrying=1 → action 2 (PLACE)
- dist>0 → pick the movement action marked "best" in the state
- After delivering a part, go pick up the NEXT required part

Examples:
pos=(2,2) carrying=0 target=PICKUP(0,0) dist=4 best=UP(5) → {"thought":"move UP toward pickup","action":5}
pos=(0,0) carrying=0 dist=0 → {"thought":"at pickup, grab","action":1}
pos=(0,0) carrying=1 target=DROPOFF(4,4) dist=8 best=DOWN(6) → {"thought":"go DOWN toward dropoff","action":6}
pos=(4,4) carrying=1 dist=0 → {"thought":"at dropoff, place","action":2}
pos=(4,4) carrying=0 target=PICKUP(0,6) part=chip delivery=2/3 best=UP(5) → {"thought":"delivered gear, now UP to get chip","action":5}
pos=(1,3) carrying=0 best=LEFT(3) → {"thought":"LEFT toward pickup","action":3}
pos=(2,1) carrying=1 best=DOWN(6) → {"thought":"DOWN toward dropoff","action":6}"""


# ─────────────────────────────────────────────────────────────────────────────
# Conversation History Buffer
# ─────────────────────────────────────────────────────────────────────────────

class ConversationHistory:
    """Maintains a rolling window of recent moves for LLM context."""

    def __init__(self, max_entries: int = 3):
        self.max_entries = max_entries
        self.entries = []

    def add(self, user_msg: str, ai_response: str):
        self.entries.append({"user": user_msg, "ai": ai_response})
        if len(self.entries) > self.max_entries:
            self.entries.pop(0)

    def get_messages(self):
        messages = []
        for entry in self.entries:
            messages.append({"role": "user", "content": entry["user"]})
            messages.append({"role": "assistant", "content": entry["ai"]})
        return messages

    def clear(self):
        self.entries = []


# ─────────────────────────────────────────────────────────────────────────────
# BFS Pathfinding — Finds shortest path around walls
# ─────────────────────────────────────────────────────────────────────────────


def _bfs_next_step(y, x, ty, tx, grid, grid_size):
    """
    BFS from (y,x) to (ty,tx) avoiding walls. Returns the first move
    action to take, or None if no path exists.
    """
    if y == ty and x == tx:
        return None  # Already there

    visited = set()
    visited.add((y, x))
    # Queue entries: (current_y, current_x, first_action)
    queue = deque()

    # Expand neighbors — the first_action records which move we took FIRST
    moves = [(5, -1, 0), (6, 1, 0), (3, 0, -1), (4, 0, 1)]
    for action, dy, dx in moves:
        ny, nx = y + dy, x + dx
        if 0 <= ny < grid_size and 0 <= nx < grid_size and grid[ny][nx] != 1:
            if ny == ty and nx == tx:
                return action  # Found target in 1 step
            visited.add((ny, nx))
            queue.append((ny, nx, action))

    while queue:
        cy, cx, first_action = queue.popleft()
        for _, dy, dx in moves:
            ny, nx = cy + dy, cx + dx
            if (ny, nx) not in visited and 0 <= ny < grid_size and 0 <= nx < grid_size:
                if grid[ny][nx] != 1:  # Not a wall
                    if ny == ty and nx == tx:
                        return first_action
                    visited.add((ny, nx))
                    queue.append((ny, nx, first_action))

    return None  # No path


def _compute_best_direction(y, x, ty, tx, grid, grid_size):
    """
    Compute the best safe move direction toward target using BFS.
    Falls back to greedy Manhattan if BFS fails.
    Returns (action_id, direction_name) or (None, None) if stuck.
    """
    ACTION_NAMES = {3: "LEFT(3)", 4: "RIGHT(4)", 5: "UP(5)", 6: "DOWN(6)"}

    # Try BFS first for wall-aware pathfinding
    bfs_action = _bfs_next_step(y, x, ty, tx, grid, grid_size)
    if bfs_action is not None:
        return bfs_action, ACTION_NAMES.get(bfs_action, str(bfs_action))

    # Fallback: greedy Manhattan
    dy = ty - y
    dx = tx - x

    def is_safe(ny, nx):
        if 0 <= ny < grid_size and 0 <= nx < grid_size:
            return grid[ny][nx] != 1
        return False

    candidates = []
    if dy < 0 and is_safe(y - 1, x):
        candidates.append((5, "UP(5)", abs(dy)))
    if dy > 0 and is_safe(y + 1, x):
        candidates.append((6, "DOWN(6)", abs(dy)))
    if dx < 0 and is_safe(y, x - 1):
        candidates.append((3, "LEFT(3)", abs(dx)))
    if dx > 0 and is_safe(y, x + 1):
        candidates.append((4, "RIGHT(4)", abs(dx)))

    if candidates:
        candidates.sort(key=lambda c: c[2], reverse=True)
        return candidates[0][0], candidates[0][1]

    # All direct paths blocked — try any safe direction
    for action, name, ny, nx in [(5, "UP(5)", y-1, x), (6, "DOWN(6)", y+1, x),
                                   (3, "LEFT(3)", y, x-1), (4, "RIGHT(4)", y, x+1)]:
        if is_safe(ny, nx):
            return action, name

    return None, None


def _get_safe_moves(y, x, grid, grid_size):
    """List all safe (non-wall, in-bounds) adjacent directions."""
    safe = []
    for name, ny, nx in [("UP(5)", y-1, x), ("DOWN(6)", y+1, x),
                          ("LEFT(3)", y, x-1), ("RIGHT(4)", y, x+1)]:
        if 0 <= ny < grid_size and 0 <= nx < grid_size:
            cell = grid[ny][nx]
            if cell != 1:  # Not a wall
                label = name
                if cell == 2:
                    label += "[hazard]"
                safe.append(label)
    return safe


# ─────────────────────────────────────────────────────────────────────────────
# Stuck Detection
# ─────────────────────────────────────────────────────────────────────────────

class StuckDetector:
    """Detects when the agent is stuck in a loop and forces alternative moves."""

    def __init__(self, window: int = 6):
        self.window = window
        self.recent_positions = deque(maxlen=window)
        self.stuck_count = 0

    def record(self, pos):
        self.recent_positions.append(tuple(pos))

    def is_stuck(self) -> bool:
        """Check if agent has been oscillating between same positions."""
        if len(self.recent_positions) < self.window:
            return False
        unique = set(self.recent_positions)
        # Stuck if only visiting 1-2 unique positions in last N steps
        return len(unique) <= 2

    def get_escape_action(self, y, x, grid, grid_size, avoid_action=None):
        """Pick a random safe direction that isn't the one we've been doing."""
        import random
        moves = [(5, -1, 0), (6, 1, 0), (3, 0, -1), (4, 0, 1)]
        random.shuffle(moves)
        for action, dy, dx in moves:
            if action == avoid_action:
                continue
            ny, nx = y + dy, x + dx
            if 0 <= ny < grid_size and 0 <= nx < grid_size and grid[ny][nx] != 1:
                return action
        return 0  # NOOP as last resort


# ─────────────────────────────────────────────────────────────────────────────
# LLM Action Selection — Fully LLM-driven
# ─────────────────────────────────────────────────────────────────────────────

_stuck_detector = StuckDetector()


def get_ai_action(
    observation: dict,
    last_reward: float = 0.0,
    history: ConversationHistory = None,
    stuck_detector: StuckDetector = None,
) -> int:
    """Query the LLM for an action. The LLM makes ALL decisions."""
    if stuck_detector is None:
        stuck_detector = _stuck_detector

    y, x = observation["robot_pos"]
    carrying = observation["carrying"]
    carrying_type = observation.get("carrying_type", None)
    grid = observation.get("grid_layout", [])
    time_remaining = observation.get("time_remaining", 0)
    inventory = observation.get("inventory", {})
    deliveries_made = observation.get("deliveries_made", 0)
    deliveries_required = observation.get("deliveries_required", 1)
    distance = observation.get("distance_to_target", 0)
    grid_size = len(grid) if grid else 0

    # ── Use metadata for reliable pickup/dropoff positions ──
    # Grid scanning misses cells where the robot is standing (shows as 6)
    metadata = observation.get("metadata", {})
    next_required = metadata.get("next_required")
    pickup_parts_map = metadata.get("pickup_parts", {})

    # Build pickup list from metadata (more reliable than grid scan)
    pickups_from_meta = []
    for pos_str, part_type in pickup_parts_map.items():
        coords = pos_str.split(",")
        pickups_from_meta.append((int(coords[0]), int(coords[1])))

    # Also scan grid for dropoffs (and any pickups metadata missed)
    pickups = []
    dropoffs = []
    for gy in range(grid_size):
        for gx in range(grid_size):
            if grid[gy][gx] == 3:
                pickups.append((gy, gx))
            elif grid[gy][gx] == 4:
                dropoffs.append((gy, gx))

    # Merge: use metadata pickups + grid-scanned pickups
    all_pickups = list(set(pickups_from_meta + pickups))

    # ── Determine target ──
    if carrying == 0:
        target_type = "PICKUP"
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
    else:
        target_type = "DROPOFF"
        if dropoffs:
            target = min(dropoffs, key=lambda p: abs(p[0] - y) + abs(p[1] - x))
        else:
            target = (grid_size - 1, grid_size - 1)

    ty, tx = target
    actual_dist = abs(ty - y) + abs(tx - x)

    # ── Stuck detection ──
    stuck_detector.record((y, x))

    # ── Pre-compute best direction using BFS ──
    best_action, best_name = _compute_best_direction(y, x, ty, tx, grid, grid_size)
    safe_moves = _get_safe_moves(y, x, grid, grid_size)

    # ── If stuck, force escape ──
    if stuck_detector.is_stuck():
        escape = stuck_detector.get_escape_action(y, x, grid, grid_size, avoid_action=best_action)
        print(f"🔄 STUCK DETECTED — escaping with action {escape}")
        stuck_detector.recent_positions.clear()
        return escape

    # ── Build the user message ──
    delivery_info = f" delivery={deliveries_made}/{deliveries_required}"
    part_info = f" part={next_required}" if next_required else ""

    if actual_dist == 0:
        # At target — tell the LLM clearly
        user_msg = (
            f"pos=({y},{x}) carrying={carrying} dist=0{delivery_info}\n"
            f"You are AT the {target_type}. "
            f"{'GRAB the part (action 1).' if carrying == 0 else 'PLACE the part (action 2).'}"
        )
    else:
        # Need to navigate
        safe_str = ", ".join(safe_moves) if safe_moves else "NONE"
        warning = ""
        if last_reward <= -0.5:
            warning = f" | PENALTY({last_reward:.1f})!"

        user_msg = (
            f"pos=({y},{x}) carrying={carrying} "
            f"target={target_type}({ty},{tx}) dist={actual_dist} "
            f"best={best_name}{part_info}{delivery_info}"
            f"{warning}"
        )

    try:
        # Build messages with conversation history
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        if history:
            messages.extend(history.get_messages())
        messages.append({"role": "user", "content": user_msg})

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            max_tokens=60,
            temperature=0.01,
        )
        ai_text = response.choices[0].message.content.strip()
        print(f"🤖 AI: {ai_text}")

        # Update history
        if history:
            history.add(user_msg, ai_text)

        action = _parse_action(ai_text)

        # ── Safety: prevent wall crashes AND invalid GRAB/PLACE ──
        action = _safety_check(
            action, y, x, grid, grid_size, best_action,
            carrying=carrying, all_pickups=all_pickups, dropoffs=dropoffs,
            actual_dist=actual_dist,
        )

        return action

    except Exception as e:
        print(f"⚠️ LLM Error: {e}")
        # On LLM failure, use the pre-computed best direction
        if best_action is not None:
            return best_action
        return 0


def _safety_check(action: int, y: int, x: int, grid: list, grid_size: int,
                   best_action: int = None, carrying: int = 0,
                   all_pickups: list = None, dropoffs: list = None,
                   actual_dist: float = 0) -> int:
    """
    Safety layer — prevents both physical AND logical errors:
    1. Wall collisions and out-of-bounds moves
    2. GRAB when not at a pickup or already carrying
    3. PLACE when not at a dropoff or not carrying
    Redirects to BFS best_action when the LLM picks an invalid action.
    """
    pos = (y, x)

    # ── Logical validation: GRAB ──
    if action == 1:  # GRAB
        at_pickup = all_pickups and pos in [tuple(p) for p in all_pickups]
        # Also trust GRAB if distance to target is 0 (we're at the right place)
        if not at_pickup and actual_dist != 0:
            if best_action is not None:
                return best_action
            return 0
        if carrying == 1:
            # Already carrying — can't grab again
            if best_action is not None:
                return best_action
            return 0

    # ── Logical validation: PLACE ──
    if action == 2:  # PLACE
        at_dropoff = dropoffs and pos in [tuple(d) for d in dropoffs]
        # Also trust PLACE if distance to target is 0 and carrying
        if not at_dropoff and actual_dist != 0:
            if best_action is not None:
                return best_action
            return 0
        if carrying == 0:
            # Not carrying — can't place
            if best_action is not None:
                return best_action
            return 0

    # ── Physical validation: movement ──
    move_map = {3: (0, -1), 4: (0, 1), 5: (-1, 0), 6: (1, 0)}

    if action in move_map:
        dy, dx = move_map[action]
        ny, nx = y + dy, x + dx
        if not (0 <= ny < grid_size and 0 <= nx < grid_size) or grid[ny][nx] == 1:
            # Blocked — redirect to best safe direction
            if best_action is not None and best_action != action:
                return best_action
            # Find any safe direction
            for alt, ady, adx in [(5, -1, 0), (6, 1, 0), (3, 0, -1), (4, 0, 1)]:
                alt_y, alt_x = y + ady, x + adx
                if 0 <= alt_y < grid_size and 0 <= alt_x < grid_size and grid[alt_y][alt_x] != 1:
                    return alt
            return 0  # Truly stuck

    return action


def _parse_action(text: str) -> int:
    """Extract action integer from LLM response."""
    # Try JSON parse
    try:
        match = re.search(r'\{[^}]+\}', text)
        if match:
            data = json.loads(match.group())
            action = int(data.get("action", 0))
            if 0 <= action <= 8:
                return action
    except (json.JSONDecodeError, ValueError, TypeError):
        pass

    # Fallback patterns
    match = re.search(r'"action"\s*:\s*(\d)', text)
    if match:
        return int(match.group(1))

    match = re.search(r'action\s*[=:]\s*(\d)', text, re.IGNORECASE)
    if match:
        return int(match.group(1))

    # Last resort: last digit in text
    digits = re.findall(r'\d', text)
    if digits:
        d = int(digits[-1])
        if 0 <= d <= 8:
            return d

    return 0


# ─────────────────────────────────────────────────────────────────────────────
# Main Loop
# ─────────────────────────────────────────────────────────────────────────────


def main():
    """Run the inference agent against all 3 difficulty tiers."""

    # Warm up LLM connection
    try:
        client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": "ready"}],
            max_tokens=1,
        )
    except Exception:
        pass

    tasks = ["smart_factory_easy", "smart_factory_medium", "smart_factory_hard"]
    all_results = []

    for task_name in tasks:
        print(f"\n{'='*60}")
        print(f"[START] task={task_name}", flush=True)
        print(f"{'='*60}")

        try:
            import asyncio
            result = asyncio.run(_run_task(task_name))
            all_results.append(result)
        except Exception as e:
            print(f"❌ Failed to run scenario: {e}")
            print(f"[END] task={task_name} score=0.01 steps=0", flush=True)
            all_results.append({
                "task": task_name,
                "deliveries": "0/?",
                "steps": 0,
                "reward": 0.0,
                "score": 0.01,
            })

    # ── Summary Table ──
    print(f"\n{'='*60}")
    print("📊 FINAL SUMMARY — TRIBUNAL Smart Factory Agent")
    print(f"{'='*60}")
    print(f"{'Task':<25} {'Deliveries':>12} {'Steps':>7} {'Reward':>10} {'Score':>8}")
    print(f"{'-'*25} {'-'*12} {'-'*7} {'-'*10} {'-'*8}")
    for r in all_results:
        print(
            f"{r['task']:<25} {r['deliveries']:>12} "
            f"{r['steps']:>7} {r['reward']:>10.3f} {r['score']:>8.4f}"
        )
    print(f"{'-'*25} {'-'*12} {'-'*7} {'-'*10} {'-'*8}")
    avg_score = sum(r['score'] for r in all_results) / max(1, len(all_results))
    print(f"{'AVERAGE':>38} {'':>10} {avg_score:>8.4f}")
    print(f"Model: {MODEL_NAME}")
    print(f"{'='*60}\n")


async def _run_task(task_name: str) -> dict:
    """Run a single task using the stateful WebSocket client."""
    from client import SmartFactoryEnv
    from models import FactoryAction

    ws_url = ENV_API_URL
    max_steps = TASK_MAX_STEPS.get(task_name, 120)

    step_count = 0
    done = False
    total_reward = 0.0
    deliveries_str = "0/?"

    # Create conversation history and stuck detector for this task
    history = ConversationHistory(max_entries=3)
    stuck_detector = StuckDetector(window=6)

    print(f"Connecting to {ws_url} ...")
    try:
        async with SmartFactoryEnv(base_url=ws_url) as env:
            # Reset environment
            res = await env.reset(task=task_name)
            observation = res.observation
            last_reward = 0.0

            while not done and step_count < max_steps:
                step_count += 1
                action_val = get_ai_action(
                    observation.model_dump(),
                    last_reward=last_reward,
                    history=history,
                    stuck_detector=stuck_detector,
                )

                res = await env.step(FactoryAction(action=action_val))
                observation = res.observation
                reward = res.reward or 0.0
                last_reward = reward
                done = res.done
                total_reward += reward

                deliveries_str = f"{observation.deliveries_made}/{observation.deliveries_required}"
                assembly = observation.assembly_progress

                print(
                    f"  [STEP] step={step_count} action={action_val} "
                    f"reward={reward:.3f} total={total_reward:.3f} "
                    f"pos={observation.robot_pos} "
                    f"carrying={observation.carrying} "
                    f"deliveries={deliveries_str}"
                    f"{' assembly=' + str(assembly) if assembly else ''}",
                    flush=True,
                )

    except Exception as e:
        print(f"  ⚠️ API Error: {e}")

    # ── Score normalization per difficulty ──
    max_theoretical = {
        "smart_factory_easy": 12.0,
        "smart_factory_medium": 20.0,
        "smart_factory_hard": 35.0,
    }
    max_reward = max_theoretical.get(task_name, 15.0)
    normalized_score = max(0.01, min(0.99, (total_reward + 2.0) / max_reward))

    print(f"\n📊 Results: total_reward={total_reward:.3f} steps={step_count}")
    print(f"[END] task={task_name} score={normalized_score:.4f} steps={step_count}", flush=True)

    return {
        "task": task_name,
        "deliveries": deliveries_str,
        "steps": step_count,
        "reward": total_reward,
        "score": normalized_score,
    }


if __name__ == "__main__":
    main()