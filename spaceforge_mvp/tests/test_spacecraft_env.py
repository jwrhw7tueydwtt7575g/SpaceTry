"""Tests for SpacecraftEnv Gymnasium environment."""

import sys
from pathlib import Path
import numpy as np
import pytest

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

try:
    from spaceforge_mvp.env.spacecraft_env import SpacecraftEnv
except ImportError:
    from env.spacecraft_env import SpacecraftEnv


def test_observation_and_action_space_shapes():
    """Test 1: obs/action space shapes correct."""
    env = SpacecraftEnv()

    # Observation space must have shape (10,)
    assert env.observation_space.shape == (10,), f"Expected shape (10,), got {env.observation_space.shape}"

    # Action space must have shape (3,) with bounds [-1.0, 1.0]
    assert env.action_space.shape == (3,), f"Expected shape (3,), got {env.action_space.shape}"
    np.testing.assert_allclose(env.action_space.low, -1.0)
    np.testing.assert_allclose(env.action_space.high, 1.0)


def test_reward_includes_bonus_at_close_distance():
    """
    Test 2: reward at dist=0.3 includes +100 docking bonus and proximity-shaping bonus.

    Hand calculation:
      pos = [0.3, 0.0, 0.0] => dist = 0.3
      vel = [0.0, 0.0, 0.0] => speed = 0.0
      action = [0.0, 0.0, 0.0] => action_penalty = 0.0
      docking_bonus = 100.0 (since dist = 0.3 < 0.5)
      proximity_bonus = 5.0 * max(0, 1 - 0.3/10.0) = 5.0 * 0.97 = 4.85
      reward = -0.3 - 0.5*(0) - 0.01*(0) + 100.0 + 4.85 = 104.55
    """
    env = SpacecraftEnv()
    pos = np.array([0.3, 0.0, 0.0], dtype=np.float64)
    vel = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    action = np.array([0.0, 0.0, 0.0], dtype=np.float64)

    expected_reward = -0.3 + 100.0 + (5.0 * (1.0 - 0.3 / 10.0))  # 104.55
    reward = env.compute_reward(pos, vel, action)

    assert reward > 90.0, f"Reward should include +100 bonus, got {reward}"
    assert reward == pytest.approx(expected_reward, abs=1e-7)


def test_reward_at_large_distance_with_large_action():
    """
    Test 3: reward at dist=10 with large action is more negative and has no docking bonus.

    Hand calculation:
      pos = [10.0, 0.0, 0.0] => dist = 10.0
      vel = [0.0, 0.0, 0.0] => speed = 0.0
      docking_bonus = 0.0 (dist >= 0.5)
      proximity_bonus = 5.0 * max(0, 1 - 10.0/10.0) = 0.0

      zero_action = [0.0, 0.0, 0.0] => action_penalty = 0.0
      reward_zero_action = -10.0 + 0 = -10.0

      large_action = [1.0, 1.0, 1.0] => action_penalty = 0.01 * 3.0 = 0.03
      reward_large_action = -10.0 - 0.03 + 0 = -10.03
    """
    env = SpacecraftEnv()
    pos = np.array([10.0, 0.0, 0.0], dtype=np.float64)
    vel = np.array([0.0, 0.0, 0.0], dtype=np.float64)

    zero_action = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    large_action = np.array([1.0, 1.0, 1.0], dtype=np.float64)

    reward_zero_action = env.compute_reward(pos, vel, zero_action)
    reward_large_action = env.compute_reward(pos, vel, large_action)

    assert reward_zero_action == pytest.approx(-10.0, abs=1e-7)
    assert reward_large_action == pytest.approx(-10.03, abs=1e-7)

    assert reward_large_action < 0.0
    assert reward_large_action < reward_zero_action


def test_reset_then_step_real_call():
    """
    Test 4: reset() then step(valid random action) returns correctly-shaped obs
    and a finite (non-NaN/inf) reward — a real call, not mocked.
    """
    env = SpacecraftEnv()
    obs, info = env.reset(seed=42)

    # Verify initial observation shape and finiteness
    assert obs.shape == (10,)
    assert np.all(np.isfinite(obs))
    assert isinstance(info, dict)

    # Sample a valid random action from the action space
    action = env.action_space.sample()
    assert env.action_space.contains(action)

    # Real step execution
    next_obs, reward, terminated, truncated, step_info = env.step(action)

    # Verify step return types, shapes, and finiteness
    assert next_obs.shape == (10,)
    assert np.all(np.isfinite(next_obs))
    assert np.isfinite(reward)
    assert isinstance(terminated, (bool, np.bool_))
    assert isinstance(truncated, (bool, np.bool_))
    assert isinstance(step_info, dict)


def test_fuel_depletion_sets_truncated_not_terminated():
    """Test 5: Fuel depletion sets truncated=True and terminated=False (when dist >= 0.5)."""
    env = SpacecraftEnv()
    env.reset(seed=42)
    env.rel_pos = np.array([10.0, 0.0, 0.0], dtype=np.float64)  # dist = 10 >= 0.5
    env.fuel = 0.0  # Fuel is depleted

    action = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    obs, reward, terminated, truncated, info = env.step(action)

    assert not terminated, "Fuel depletion should not set terminated=True"
    assert truncated, "Fuel depletion should set truncated=True"
