# Copyright (c) 2026 TRIBUNAL Team.
# Smart Factory Assembly Environment — OpenEnv Compliant Environment.

"""
Smart Factory Assembly — Environment Logic.

A multi-difficulty manufacturing RL environment where an AI agent must
navigate a grid, pick up parts from stations, and deliver them to dropoff
zones while avoiding obstacles and hazard zones.

Difficulty tiers
-----------------
- **easy** (5×5):   1 pickup, 1 dropoff, no obstacles, 1 part type, 50 steps.
- **medium** (7×7): 2 pickups, 1 dropoff, walls + hazards, 2 part types,
                     ordered delivery, 75 steps.
- **hard** (10×10): 3 pickups, 2 dropoffs, walls + hazards + conveyors,
                     3 part types, ordered delivery, stochastic events, 120 steps.
"""

from __future__ import annotations

import copy
import math
import random
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import EnvironmentMetadata

import sys
import os

# Add parent dir to path so models can be imported
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import FactoryAction, FactoryObservation, FactoryState

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Cell types for grid_layout
EMPTY = 0
WALL = 1
HAZARD = 2
PICKUP = 3
DROPOFF = 4
CONVEYOR = 5
ROBOT = 6

# Action IDs
A_NOOP = 0
A_GRAB = 1
A_PLACE = 2
A_LEFT = 3
A_RIGHT = 4
A_UP = 5
A_DOWN = 6
A_WAIT = 7
A_INSPECT = 8

# Part types
PART_TYPES = ["gear", "chip", "frame"]

# Difficulty configs
DIFFICULTY_CONFIG = {
    "smart_factory_easy": {
        "grid_size": 5,
        "max_steps": 50,
        "num_part_types": 1,
        "walls": [],
        "hazards": [],
        "conveyors": [],
        "pickups": [(0, 0)],
        "dropoffs": [(4, 4)],
        "deliveries_required": 1,
        "ordered_delivery": False,
        "stochastic": False,
        "breakdown_chance": 0.0,
    },
    "smart_factory_medium": {
        "grid_size": 7,
        "max_steps": 75,
        "num_part_types": 2,
        "walls": [(2, 3), (3, 3), (4, 3)],
        "hazards": [(1, 5), (2, 5)],
        "conveyors": [],
        "pickups": [(0, 0), (0, 6)],
        "dropoffs": [(6, 3)],
        "deliveries_required": 2,
        "ordered_delivery": True,
        "stochastic": False,
        "breakdown_chance": 0.0,
    },
    "smart_factory_hard": {
        "grid_size": 10,
        "max_steps": 120,
        "num_part_types": 3,
        "walls": [
            (2, 2), (2, 3), (2, 4),
            (5, 6), (5, 7), (5, 8),
            (7, 1), (7, 2),
        ],
        "hazards": [(3, 7), (3, 8), (4, 7), (8, 4), (8, 5)],
        "conveyors": [(4, 1), (4, 2), (4, 3)],
        "pickups": [(0, 0), (0, 9), (9, 0)],
        "dropoffs": [(9, 9), (5, 4)],
        "deliveries_required": 3,
        "ordered_delivery": True,
        "stochastic": True,
        "breakdown_chance": 0.05,
    },
}


def _manhattan(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class SmartFactoryEnvironment(
    Environment[FactoryAction, FactoryObservation, FactoryState]
):
    """
    OpenEnv-compliant Smart Factory Assembly environment.

    Follows the Gymnasium-style API: ``reset()``, ``step(action)``, ``state``.
    """

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)

        # Internal state (set properly during reset)
        self._grid_size: int = 5
        self._max_steps: int = 50
        self._grid: List[List[int]] = []
        self._robot_pos: List[int] = [0, 0]
        self._carrying: int = 0
        self._carrying_type: Optional[str] = None
        self._step_count: int = 0
        self._total_reward: float = 0.0
        self._done: bool = False
        self._deliveries_made: int = 0
        self._deliveries_required: int = 1
        self._collisions: int = 0
        self._task_name: str = "smart_factory_easy"
        self._difficulty: str = "easy"
        self._episode_id: str = str(uuid4())

        # Delivery tracking
        self._delivery_order: List[str] = []
        self._next_delivery_idx: int = 0
        self._ordered_delivery: bool = False

        # Part availability at pickups
        self._pickup_parts: Dict[Tuple[int, int], str] = {}
        self._pickup_positions: List[Tuple[int, int]] = []
        self._dropoff_positions: List[Tuple[int, int]] = []

        # Stochastic elements
        self._stochastic: bool = False
        self._breakdown_chance: float = 0.0
        self._broken_pickups: set = set()
        self._conveyor_positions: List[Tuple[int, int]] = []

        # Previous distance for reward shaping
        self._prev_distance: float = 0.0

        # Inventory tracking (what's still needed)
        self._inventory: Dict[str, int] = {}

        # Config reference
        self._config: Dict[str, Any] = {}

    # -----------------------------------------------------------------------
    # reset
    # -----------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> FactoryObservation:
        """Reset the environment and return the initial observation."""
        if hasattr(self, '_reset_rubric'):
            self._reset_rubric()

        if seed is not None:
            random.seed(seed)

        self._episode_id = episode_id or str(uuid4())

        # Determine task / difficulty
        task = kwargs.get("task", "smart_factory_easy")
        if isinstance(task, str):
            task = task.lower().strip()
        if task not in DIFFICULTY_CONFIG:
            task = "smart_factory_easy"

        self._task_name = task
        self._difficulty = task.replace("smart_factory_", "")
        self._config = DIFFICULTY_CONFIG[task]

        cfg = self._config
        self._grid_size = cfg["grid_size"]
        self._max_steps = cfg["max_steps"]
        self._deliveries_required = cfg["deliveries_required"]
        self._ordered_delivery = cfg["ordered_delivery"]
        self._stochastic = cfg["stochastic"]
        self._breakdown_chance = cfg["breakdown_chance"]

        # Build grid
        gs = self._grid_size
        self._grid = [[EMPTY for _ in range(gs)] for _ in range(gs)]

        for (y, x) in cfg["walls"]:
            self._grid[y][x] = WALL
        for (y, x) in cfg["hazards"]:
            self._grid[y][x] = HAZARD
        for (y, x) in cfg["conveyors"]:
            self._grid[y][x] = CONVEYOR
        for (y, x) in cfg["pickups"]:
            self._grid[y][x] = PICKUP
        for (y, x) in cfg["dropoffs"]:
            self._grid[y][x] = DROPOFF

        self._pickup_positions = list(cfg["pickups"])
        self._dropoff_positions = list(cfg["dropoffs"])
        self._conveyor_positions = list(cfg["conveyors"])

        # Assign part types to pickups
        num_types = cfg["num_part_types"]
        parts = PART_TYPES[:num_types]
        self._pickup_parts = {}
        for i, pos in enumerate(self._pickup_positions):
            self._pickup_parts[pos] = parts[i % num_types]

        # Delivery order
        self._delivery_order = parts[:self._deliveries_required]
        self._next_delivery_idx = 0

        # Inventory = what still needs to be delivered
        self._inventory = {}
        for p in self._delivery_order:
            self._inventory[p] = self._inventory.get(p, 0) + 1

        # Robot starts near center
        center = gs // 2
        self._robot_pos = [center, center]
        # Make sure we don't start on a wall
        while self._grid[self._robot_pos[0]][self._robot_pos[1]] == WALL:
            self._robot_pos[0] = (self._robot_pos[0] + 1) % gs

        self._carrying = 0
        self._carrying_type = None
        self._step_count = 0
        self._total_reward = 0.0
        self._done = False
        self._deliveries_made = 0
        self._collisions = 0
        self._broken_pickups = set()

        # Initial distance
        self._prev_distance = self._compute_target_distance()

        return self._make_observation(reward=0.0)

    # -----------------------------------------------------------------------
    # step
    # -----------------------------------------------------------------------

    def step(
        self,
        action: FactoryAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> FactoryObservation:
        """Execute one step in the environment."""
        if self._done:
            return self._make_observation(reward=0.0)

        # Safety guard for stateless HTTP calls
        if not self._grid:
            self.reset(task=self._task_name)

        self._step_count += 1
        act = action.action
        reward = -0.05  # base step cost

        y, x = self._robot_pos
        gs = self._grid_size

        # --- Movement actions ---
        new_y, new_x = y, x
        if act == A_LEFT:
            new_x = x - 1
        elif act == A_RIGHT:
            new_x = x + 1
        elif act == A_UP:
            new_y = y - 1
        elif act == A_DOWN:
            new_y = y + 1

        if act in (A_LEFT, A_RIGHT, A_UP, A_DOWN):
            # Boundary check
            if 0 <= new_y < gs and 0 <= new_x < gs:
                cell = self._grid[new_y][new_x]
                if cell == WALL:
                    # Collision
                    reward += -1.0
                    self._collisions += 1
                else:
                    self._robot_pos = [new_y, new_x]
                    # Hazard penalty
                    if cell == HAZARD:
                        reward += -0.5
            else:
                # Hit boundary — treat as collision
                reward += -1.0
                self._collisions += 1

        # --- GRAB action ---
        elif act == A_GRAB:
            pos_tuple = (self._robot_pos[0], self._robot_pos[1])
            if (
                self._carrying == 0
                and pos_tuple in self._pickup_parts
                and pos_tuple not in self._broken_pickups
            ):
                part_type = self._pickup_parts[pos_tuple]
                self._carrying = 1
                self._carrying_type = part_type

                # Quality multiplier: correct order = 1.0, wrong = 0.5
                if self._ordered_delivery and self._next_delivery_idx < len(self._delivery_order):
                    expected = self._delivery_order[self._next_delivery_idx]
                    quality = 1.0 if part_type == expected else 0.5
                else:
                    quality = 1.0

                reward += 2.0 * quality
            else:
                reward += -0.1  # useless grab

        # --- PLACE action ---
        elif act == A_PLACE:
            pos_tuple = (self._robot_pos[0], self._robot_pos[1])
            if self._carrying == 1 and pos_tuple in [
                tuple(d) for d in self._dropoff_positions
            ]:
                part_type = self._carrying_type

                # Check delivery order
                correct_order = True
                if self._ordered_delivery and self._next_delivery_idx < len(self._delivery_order):
                    expected = self._delivery_order[self._next_delivery_idx]
                    if part_type != expected:
                        correct_order = False

                if correct_order:
                    self._carrying = 0
                    self._carrying_type = None
                    self._deliveries_made += 1
                    self._next_delivery_idx += 1

                    # Decrement inventory
                    if part_type and part_type in self._inventory:
                        self._inventory[part_type] = max(0, self._inventory[part_type] - 1)
                        if self._inventory[part_type] == 0:
                            del self._inventory[part_type]

                    # Speed bonus
                    steps_fraction = self._step_count / self._max_steps
                    speed_bonus = max(0.0, 2.0 * (1.0 - steps_fraction))
                    reward += 5.0 + speed_bonus

                    # Check completion
                    if self._deliveries_made >= self._deliveries_required:
                        efficiency = self._deliveries_made / max(1, self._step_count)
                        reward += 3.0 * efficiency * 10.0  # scale up
                        self._done = True
                else:
                    reward += -0.5  # wrong order penalty
            else:
                reward += -0.1  # useless place

        # --- NOOP / WAIT ---
        elif act in (A_NOOP, A_WAIT):
            reward += -0.1  # idle penalty

        # --- INSPECT ---
        elif act == A_INSPECT:
            # Costs a step but reveals useful info via metadata
            pass  # no extra reward

        # --- Distance-based reward shaping ---
        curr_distance = self._compute_target_distance()
        distance_delta = self._prev_distance - curr_distance
        reward += 0.1 * distance_delta
        self._prev_distance = curr_distance

        # --- Stochastic events (hard mode) ---
        if self._stochastic:
            self._apply_stochastic_events()

        # --- Timeout ---
        time_remaining = self._max_steps - self._step_count
        if time_remaining <= 0:
            self._done = True
            # Partial completion bonus
            if self._deliveries_made > 0:
                completion_frac = self._deliveries_made / self._deliveries_required
                reward += 1.0 * completion_frac

        self._total_reward += reward

        return self._make_observation(reward=reward)

    # -----------------------------------------------------------------------
    # state property
    # -----------------------------------------------------------------------

    @property
    def state(self) -> FactoryState:
        """Return the current internal state."""
        efficiency = (
            self._deliveries_made / max(1, self._step_count)
            if self._step_count > 0
            else 0.0
        )
        return FactoryState(
            episode_id=self._episode_id,
            step_count=self._step_count,
            difficulty=self._difficulty,
            total_reward=round(self._total_reward, 4),
            deliveries_completed=self._deliveries_made,
            collisions=self._collisions,
            efficiency_score=round(efficiency, 4),
        )

    # -----------------------------------------------------------------------
    # metadata
    # -----------------------------------------------------------------------

    def get_metadata(self) -> EnvironmentMetadata:
        return EnvironmentMetadata(
            name="SmartFactoryAssembly",
            description=(
                "Multi-difficulty manufacturing RL environment. "
                "Navigate a grid, pick up parts, deliver them in order "
                "while avoiding obstacles, hazards, and stochastic disruptions."
            ),
            version="1.0.0",
            author="TRIBUNAL Team",
        )

    # -----------------------------------------------------------------------
    # close
    # -----------------------------------------------------------------------

    def close(self) -> None:
        """Clean up (nothing to clean in this env)."""
        pass

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _make_observation(self, reward: float) -> FactoryObservation:
        """Build a FactoryObservation from current internal state."""
        # Build the visible grid (copy + overlay robot)
        vis_grid = [row[:] for row in self._grid]
        ry, rx = self._robot_pos
        vis_grid[ry][rx] = ROBOT

        return FactoryObservation(
            robot_pos=list(self._robot_pos),
            carrying=self._carrying,
            carrying_type=self._carrying_type,
            grid_layout=vis_grid,
            inventory=dict(self._inventory),
            time_remaining=max(0, self._max_steps - self._step_count),
            task_name=self._task_name,
            deliveries_made=self._deliveries_made,
            deliveries_required=self._deliveries_required,
            distance_to_target=round(self._compute_target_distance(), 2),
            done=self._done,
            reward=round(reward, 4),
            metadata={
                "step": self._step_count,
                "collisions": self._collisions,
                "broken_pickups": list(self._broken_pickups),
            },
        )

    def _compute_target_distance(self) -> float:
        """Compute Manhattan distance to the current logical target."""
        pos = tuple(self._robot_pos)
        if self._carrying == 0:
            # Target = nearest non-broken pickup
            targets = [
                p for p in self._pickup_positions
                if p not in self._broken_pickups
            ]
            if not targets:
                targets = list(self._pickup_positions)
        else:
            # Target = nearest dropoff
            targets = list(self._dropoff_positions)

        if not targets:
            return 0.0
        return min(_manhattan(pos, t) for t in targets)

    def _apply_stochastic_events(self) -> None:
        """Apply random events (conveyor drift, breakdowns) in hard mode."""
        # Machine breakdown
        if self._breakdown_chance > 0 and random.random() < self._breakdown_chance:
            available = [
                p for p in self._pickup_positions if p not in self._broken_pickups
            ]
            if available:
                broken = random.choice(available)
                self._broken_pickups.add(broken)

        # Conveyor belt movement: shift robot if on conveyor
        ry, rx = self._robot_pos
        if (ry, rx) in self._conveyor_positions:
            # Conveyors push DOWN
            new_ry = ry + 1
            if new_ry < self._grid_size and self._grid[new_ry][rx] != WALL:
                self._robot_pos = [new_ry, rx]

        # Repair breakdowns after some time (10% chance per step)
        if self._broken_pickups and random.random() < 0.10:
            repaired = random.choice(list(self._broken_pickups))
            self._broken_pickups.discard(repaired)
