# Copyright (c) 2026 TRIBUNAL Team.
# Smart Factory Assembly Environment — OpenEnv Compliant Models.
#
# Typed Pydantic contracts for Action, Observation, and State.

"""
Smart Factory Assembly — Data Models.

Defines the typed API contract for the Smart Factory environment using
Pydantic BaseModel, inheriting from openenv.core base types.

Action Space (discrete, 0-8):
    0=NOOP, 1=GRAB, 2=PLACE, 3=LEFT, 4=RIGHT, 5=UP, 6=DOWN, 7=WAIT, 8=INSPECT

Observation Space:
    - robot_pos, carrying, carrying_type, grid_layout, inventory,
      time_remaining, task_name, deliveries_made, deliveries_required,
      distance_to_target

State:
    - episode_id, step_count, difficulty, total_reward,
      deliveries_completed, collisions, efficiency_score
"""

from typing import Any, Dict, List, Optional

from pydantic import Field

from openenv.core.env_server.types import Action, Observation, State


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------

class FactoryAction(Action):
    """Discrete action for the Smart Factory environment.

    Action mapping:
        0 = NOOP   — do nothing
        1 = GRAB   — pick up part at current position
        2 = PLACE  — place carried part at current position
        3 = LEFT   — move left  (x - 1)
        4 = RIGHT  — move right (x + 1)
        5 = UP     — move up    (y - 1)
        6 = DOWN   — move down  (y + 1)
        7 = WAIT   — wait one step (conveyor tick, no move)
        8 = INSPECT — inspect current cell for part quality info
    """

    action: int = Field(
        ...,
        ge=0,
        le=8,
        description="Discrete action ID (0-8). See class docstring for mapping.",
    )


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------

class FactoryObservation(Observation):
    """Rich observation returned after every step/reset.

    Extends the base Observation (which provides ``done``, ``reward``,
    ``metadata``) with factory-specific fields.
    """

    robot_pos: List[int] = Field(
        description="Current robot position as [y, x]."
    )
    carrying: int = Field(
        default=0,
        ge=0,
        le=1,
        description="1 if robot is carrying a part, 0 otherwise.",
    )
    carrying_type: Optional[str] = Field(
        default=None,
        description="Type of part currently carried (e.g. 'gear', 'chip', 'frame'). None if not carrying.",
    )
    grid_layout: List[List[int]] = Field(
        description=(
            "2-D grid encoding: 0=empty, 1=wall, 2=hazard, "
            "3=pickup, 4=dropoff, 5=conveyor, 6=robot."
        ),
    )
    inventory: Dict[str, int] = Field(
        default_factory=dict,
        description="Parts still required at dropoff stations, e.g. {'gear': 1, 'chip': 2}.",
    )
    time_remaining: int = Field(
        description="Number of steps remaining before the episode times out.",
    )
    task_name: str = Field(
        description="Current task / difficulty tier (e.g. 'smart_factory_easy').",
    )
    deliveries_made: int = Field(
        default=0,
        description="Number of parts successfully delivered so far.",
    )
    deliveries_required: int = Field(
        description="Total parts that must be delivered to complete the episode.",
    )
    distance_to_target: float = Field(
        default=0.0,
        description="Manhattan distance from robot to the current target (pickup or dropoff).",
    )


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class FactoryState(State):
    """Full internal episode state for the Smart Factory environment.

    Inherits ``episode_id`` and ``step_count`` from the OpenEnv ``State`` base.
    """

    difficulty: str = Field(
        default="easy",
        description="Current difficulty tier.",
    )
    total_reward: float = Field(
        default=0.0,
        description="Cumulative reward over the episode.",
    )
    deliveries_completed: int = Field(
        default=0,
        description="Parts delivered so far.",
    )
    collisions: int = Field(
        default=0,
        description="Number of wall / obstacle collisions.",
    )
    efficiency_score: float = Field(
        default=0.0,
        description="Efficiency metric: deliveries / steps.",
    )
