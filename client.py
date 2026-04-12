# Copyright (c) 2026 TRIBUNAL Team.
# Smart Factory Assembly — EnvClient.

"""
Smart Factory Environment Client.

Provides ``SmartFactoryEnv``, a WebSocket-based client for interacting
with a running Smart Factory server.

Usage (async):
    >>> async with SmartFactoryEnv(base_url="http://localhost:7860") as env:
    ...     result = await env.reset(task="smart_factory_easy")
    ...     result = await env.step(FactoryAction(action=5))

Usage (sync):
    >>> with SmartFactoryEnv(base_url="http://localhost:7860").sync() as env:
    ...     result = env.reset(task="smart_factory_easy")
    ...     result = env.step(FactoryAction(action=5))
"""

from typing import Any, Dict

from openenv.core.env_client import EnvClient
from openenv.core.client_types import StepResult

from models import FactoryAction, FactoryObservation, FactoryState


class SmartFactoryEnv(EnvClient[FactoryAction, FactoryObservation, FactoryState]):
    """WebSocket client for the Smart Factory environment."""

    def _step_payload(self, action: FactoryAction) -> Dict[str, Any]:
        """Serialize a FactoryAction to the JSON dict expected by the server."""
        return action.model_dump()

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[FactoryObservation]:
        """Deserialize a server response into StepResult[FactoryObservation]."""
        obs_data = payload.get("observation", {})
        reward = payload.get("reward")
        done = payload.get("done", False)

        observation = FactoryObservation(
            robot_pos=obs_data.get("robot_pos", [0, 0]),
            carrying=obs_data.get("carrying", 0),
            carrying_type=obs_data.get("carrying_type"),
            grid_layout=obs_data.get("grid_layout", []),
            inventory=obs_data.get("inventory", {}),
            time_remaining=obs_data.get("time_remaining", 0),
            task_name=obs_data.get("task_name", ""),
            deliveries_made=obs_data.get("deliveries_made", 0),
            deliveries_required=obs_data.get("deliveries_required", 1),
            distance_to_target=obs_data.get("distance_to_target", 0.0),
            assembly_progress=obs_data.get("assembly_progress", []),
            done=done,
            reward=reward,
            metadata=obs_data.get("metadata") or {},
        )

        return StepResult(
            observation=observation,
            reward=reward,
            done=done,
        )

    def _parse_state(self, payload: Dict[str, Any]) -> FactoryState:
        """Deserialize a server state response into FactoryState."""
        return FactoryState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            difficulty=payload.get("difficulty", "easy"),
            total_reward=payload.get("total_reward", 0.0),
            deliveries_completed=payload.get("deliveries_completed", 0),
            collisions=payload.get("collisions", 0),
            efficiency_score=payload.get("efficiency_score", 0.0),
        )
