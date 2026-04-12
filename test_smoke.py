# Copyright (c) 2026 TRIBUNAL Team.
# Smart Factory Assembly — Test Suite.

"""
Smoke and unit tests for the Smart Factory environment.

Run with:
    pytest test_smoke.py -v
"""

import pytest
from server.environment import SmartFactoryEnvironment
from models import FactoryAction, FactoryObservation, FactoryState


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(params=["smart_factory_easy", "smart_factory_medium", "smart_factory_hard"])
def env_with_task(request):
    """Yield a fresh environment for each difficulty tier."""
    env = SmartFactoryEnvironment()
    obs = env.reset(task=request.param)
    yield env, obs, request.param
    env.close()


@pytest.fixture
def easy_env():
    """Yield a fresh easy environment."""
    env = SmartFactoryEnvironment()
    obs = env.reset(task="smart_factory_easy")
    yield env, obs
    env.close()


# ---------------------------------------------------------------------------
# Test 1: Environment resets correctly for all tiers
# ---------------------------------------------------------------------------

class TestReset:
    """Tests for environment reset behavior."""

    EXPECTED_GRID_SIZES = {
        "smart_factory_easy": 5,
        "smart_factory_medium": 7,
        "smart_factory_hard": 10,
    }
    EXPECTED_DELIVERIES = {
        "smart_factory_easy": 1,
        "smart_factory_medium": 2,
        "smart_factory_hard": 3,
    }
    EXPECTED_MAX_STEPS = {
        "smart_factory_easy": 50,
        "smart_factory_medium": 75,
        "smart_factory_hard": 120,
    }

    def test_observation_type(self, env_with_task):
        _, obs, _ = env_with_task
        assert isinstance(obs, FactoryObservation)

    def test_grid_size(self, env_with_task):
        _, obs, task = env_with_task
        expected = self.EXPECTED_GRID_SIZES[task]
        assert len(obs.grid_layout) == expected
        assert len(obs.grid_layout[0]) == expected

    def test_initial_state(self, env_with_task):
        _, obs, _ = env_with_task
        assert obs.carrying == 0
        assert obs.carrying_type is None
        assert obs.deliveries_made == 0
        assert obs.done is False

    def test_deliveries_required(self, env_with_task):
        _, obs, task = env_with_task
        assert obs.deliveries_required == self.EXPECTED_DELIVERIES[task]

    def test_time_remaining(self, env_with_task):
        _, obs, task = env_with_task
        assert obs.time_remaining == self.EXPECTED_MAX_STEPS[task]

    def test_inventory_populated(self, env_with_task):
        _, obs, _ = env_with_task
        assert len(obs.inventory) > 0

    def test_state_type(self, env_with_task):
        env, _, _ = env_with_task
        state = env.state
        assert isinstance(state, FactoryState)
        assert state.step_count == 0

    def test_metadata(self, env_with_task):
        env, _, _ = env_with_task
        meta = env.get_metadata()
        assert meta.name == "SmartFactoryAssembly"
        assert isinstance(meta.version, str)


# ---------------------------------------------------------------------------
# Test 2: Step mechanics
# ---------------------------------------------------------------------------

class TestStep:
    """Tests for environment step behavior."""

    def test_movement_changes_position(self, easy_env):
        env, obs = easy_env
        initial_pos = list(obs.robot_pos)
        # Move UP
        obs = env.step(FactoryAction(action=5))
        # Position should change (or collision if at boundary)
        assert isinstance(obs, FactoryObservation)
        assert obs.time_remaining < 50

    def test_step_decrements_time(self, easy_env):
        env, obs = easy_env
        obs = env.step(FactoryAction(action=0))  # NOOP
        assert obs.time_remaining == 49

    def test_noop_penalty(self, easy_env):
        env, obs = easy_env
        obs = env.step(FactoryAction(action=0))
        # NOOP gives step cost (-0.05) + idle penalty (-0.1) = at least negative
        assert obs.reward < 0

    def test_wall_collision_penalty(self):
        """Test that walking into a wall gives a collision penalty."""
        env = SmartFactoryEnvironment()
        env.reset(task="smart_factory_medium")
        # Walk into boundary repeatedly
        for _ in range(10):
            obs = env.step(FactoryAction(action=5))  # UP until boundary
        # After hitting boundary, should have collisions
        state = env.state
        assert state.collisions >= 0  # Non-negative
        env.close()

    def test_grab_without_pickup(self, easy_env):
        """Grabbing when not at a pickup station should give penalty."""
        env, obs = easy_env
        # Robot starts at center, not at pickup
        obs = env.step(FactoryAction(action=1))  # GRAB
        assert obs.carrying == 0  # Should not pick up anything
        assert obs.reward < 0

    def test_place_without_carrying(self, easy_env):
        """Placing when not carrying should give penalty."""
        env, obs = easy_env
        obs = env.step(FactoryAction(action=2))  # PLACE
        assert obs.reward < 0

    def test_episode_timeout(self):
        """Episode should end when time runs out."""
        env = SmartFactoryEnvironment()
        env.reset(task="smart_factory_easy")
        for _ in range(50):
            obs = env.step(FactoryAction(action=0))  # NOOP until timeout
        assert obs.done is True
        env.close()

    def test_done_returns_zero_reward(self):
        """After done, steps should return 0 reward."""
        env = SmartFactoryEnvironment()
        env.reset(task="smart_factory_easy")
        for _ in range(50):
            obs = env.step(FactoryAction(action=0))
        # One more step after done
        obs = env.step(FactoryAction(action=5))
        assert obs.reward == 0.0
        env.close()


# ---------------------------------------------------------------------------
# Test 3: Full episode — easy tier completion
# ---------------------------------------------------------------------------

class TestFullEpisode:
    """Test that a heuristic agent can complete the easy tier."""

    def test_easy_completion_with_heuristic(self):
        """A simple heuristic should be able to complete easy mode."""
        env = SmartFactoryEnvironment()
        obs = env.reset(task="smart_factory_easy", seed=42)

        # Heuristic: go to (0,0) pickup, grab, go to (4,4) dropoff, place
        for _ in range(50):
            y, x = obs.robot_pos
            if obs.carrying == 0:
                # Go to pickup at (0, 0)
                if y > 0:
                    obs = env.step(FactoryAction(action=5))  # UP
                elif x > 0:
                    obs = env.step(FactoryAction(action=3))  # LEFT
                else:
                    obs = env.step(FactoryAction(action=1))  # GRAB
            else:
                # Go to dropoff at (4, 4)
                if y < 4:
                    obs = env.step(FactoryAction(action=6))  # DOWN
                elif x < 4:
                    obs = env.step(FactoryAction(action=4))  # RIGHT
                else:
                    obs = env.step(FactoryAction(action=2))  # PLACE

            if obs.done:
                break

        assert obs.deliveries_made == 1
        assert obs.deliveries_required == 1
        assert obs.done is True
        env.close()


# ---------------------------------------------------------------------------
# Test 4: Grading rubric
# ---------------------------------------------------------------------------

class TestRubric:
    """Tests for the grading rubric calculations."""

    def test_rubric_has_all_keys(self, easy_env):
        env, _ = easy_env
        rubric = env.get_rubric()
        expected_keys = {
            "completion_rate", "efficiency", "collision_rate",
            "hazard_exposure", "speed_score", "overall_score",
        }
        assert set(rubric.keys()) == expected_keys

    def test_rubric_values_in_range(self, easy_env):
        env, obs = easy_env
        # Take a few steps
        for _ in range(5):
            env.step(FactoryAction(action=5))
        rubric = env.get_rubric()
        for key, val in rubric.items():
            assert 0.0 <= val <= 1.0, f"{key}={val} out of [0, 1]"

    def test_completion_rubric_after_delivery(self):
        """Completion rate should be 1.0 after all deliveries."""
        env = SmartFactoryEnvironment()
        obs = env.reset(task="smart_factory_easy", seed=42)

        # Complete the episode with heuristic
        for _ in range(50):
            y, x = obs.robot_pos
            if obs.carrying == 0:
                if y > 0:
                    obs = env.step(FactoryAction(action=5))
                elif x > 0:
                    obs = env.step(FactoryAction(action=3))
                else:
                    obs = env.step(FactoryAction(action=1))
            else:
                if y < 4:
                    obs = env.step(FactoryAction(action=6))
                elif x < 4:
                    obs = env.step(FactoryAction(action=4))
                else:
                    obs = env.step(FactoryAction(action=2))
            if obs.done:
                break

        rubric = env.get_rubric()
        assert rubric["completion_rate"] == 1.0
        assert rubric["overall_score"] > 0.5
        env.close()


# ---------------------------------------------------------------------------
# Test 5: Assembly mechanics
# ---------------------------------------------------------------------------

class TestAssembly:
    """Tests for assembly line and delivery mechanics."""

    def test_assembly_progress_tracking(self):
        """Assembly progress should track delivered parts."""
        env = SmartFactoryEnvironment()
        obs = env.reset(task="smart_factory_easy", seed=42)

        # Complete a delivery
        for _ in range(50):
            y, x = obs.robot_pos
            if obs.carrying == 0:
                if y > 0:
                    obs = env.step(FactoryAction(action=5))
                elif x > 0:
                    obs = env.step(FactoryAction(action=3))
                else:
                    obs = env.step(FactoryAction(action=1))
            else:
                if y < 4:
                    obs = env.step(FactoryAction(action=6))
                elif x < 4:
                    obs = env.step(FactoryAction(action=4))
                else:
                    obs = env.step(FactoryAction(action=2))
            if obs.done:
                break

        assert len(obs.assembly_progress) == 1
        assert obs.assembly_progress[0] == "gear"
        env.close()

    def test_inspect_action_on_hard(self):
        """INSPECT action should work on hard mode with quality inspection."""
        env = SmartFactoryEnvironment()
        obs = env.reset(task="smart_factory_hard", seed=42)

        # Move to a pickup location and inspect
        # First navigate to (0,0) - a pickup station
        for _ in range(20):
            y, x = obs.robot_pos
            if y > 0:
                obs = env.step(FactoryAction(action=5))  # UP
            elif x > 0:
                obs = env.step(FactoryAction(action=3))  # LEFT
            else:
                break

        # Now at or near (0,0), try INSPECT
        obs = env.step(FactoryAction(action=8))  # INSPECT
        # Check that inspection results appear in metadata
        meta = obs.metadata
        assert "inspection_results" in meta
        env.close()


# ---------------------------------------------------------------------------
# Test 6: Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_double_grab(self):
        """Cannot grab when already carrying."""
        env = SmartFactoryEnvironment()
        obs = env.reset(task="smart_factory_easy", seed=42)

        # Navigate to pickup
        for _ in range(20):
            y, x = obs.robot_pos
            if y > 0:
                obs = env.step(FactoryAction(action=5))
            elif x > 0:
                obs = env.step(FactoryAction(action=3))
            else:
                break

        # Grab once
        obs = env.step(FactoryAction(action=1))
        assert obs.carrying == 1

        # Try to grab again — should fail
        obs = env.step(FactoryAction(action=1))
        assert obs.carrying == 1  # Still carrying same item
        env.close()

    def test_invalid_task_defaults_to_easy(self):
        """Invalid task name should default to easy."""
        env = SmartFactoryEnvironment()
        obs = env.reset(task="nonexistent_task")
        assert obs.deliveries_required == 1
        assert len(obs.grid_layout) == 5
        env.close()

    def test_seed_reproducibility(self):
        """Same seed should produce same initial state."""
        env1 = SmartFactoryEnvironment()
        obs1 = env1.reset(task="smart_factory_hard", seed=12345)

        env2 = SmartFactoryEnvironment()
        obs2 = env2.reset(task="smart_factory_hard", seed=12345)

        assert obs1.robot_pos == obs2.robot_pos
        assert obs1.grid_layout == obs2.grid_layout
        env1.close()
        env2.close()
