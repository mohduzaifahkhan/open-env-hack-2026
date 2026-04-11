# Copyright (c) 2026 TRIBUNAL Team.
# Smart Factory Assembly — Package exports.

"""
Smart Factory Assembly Environment for OpenEnv.

A multi-difficulty manufacturing RL environment where an AI agent
navigates a grid, picks up parts, and delivers them while avoiding
obstacles and stochastic disruptions.

Quick start::

    from smart_factory import SmartFactoryEnv, FactoryAction

    with SmartFactoryEnv(base_url="https://your-space.hf.space").sync() as env:
        result = env.reset(task="smart_factory_easy")
        result = env.step(FactoryAction(action=5))
        print(result.observation.robot_pos)
"""

from models import FactoryAction, FactoryObservation, FactoryState
from client import SmartFactoryEnv

__all__ = [
    "SmartFactoryEnv",
    "FactoryAction",
    "FactoryObservation",
    "FactoryState",
]
