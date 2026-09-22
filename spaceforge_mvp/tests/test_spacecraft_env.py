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


def test_potential_based_reward_shaping_two_steps():
    """
    Test 2: potential-based reward shaping calculated over two sequential steps.

    Hand calculation breakdown:
      Setup:
        pos_0 = [10.0, 0.0, 0.0], vel_0 = [0.0, 0.0, 0.0], dt = 1.0, n = 0.00113

      Step 1 (action_1 = [-0.1, 0.0, 0.0]):
        dist_prev = 10.0
        a_cw = [0.000038307, 0.0, 0.0]
        a_total = [-0.099961693, 0.0, 0.0]
        vel_1 = [-0.099961693, 0.0, 0.0]
        pos_1 = [9.900038307, 0.0, 0.0]
        dist_curr = 9.900038307
        progress_reward = 5.0 * (10.0 - 9.900038307) = 0.499808465
        velocity_penalty = -0.5 * 0.099961693 = -0.0499808465
        action_penalty = -0.01 * 0.1 = -0.001
        reward_1 = 0.499808465 - 0.0499808465 - 0.001 = 0.4488276185

      Step 2 (action_2 = [-0.1, 0.0, 0.0]):
        dist_prev = 9.900038307
        a_cw = [0.000037924077, 0.000225913426, 0.0]
        a_total = [-0.0999620759, 0.000225913426, 0.0]
        vel_2 = [-0.1999237689, 0.0002259134, 0.0]
        pos_2 = [9.700114538, 0.0002259134, 0.0]
        dist_curr = 9.700114541
        progress_reward = 5.0 * (9.900038307 - 9.700114541) = 0.999618831
        velocity_penalty = -0.5 * 0.199923897 = -0.099961948
        action_penalty = -0.01 * 0.1 = -0.001
        reward_2 = 0.999618831 - 0.099961948 - 0.001 = 0.8986568832
    """
    env = SpacecraftEnv()
    env.reset(seed=42)
    env.rel_pos = np.array([10.0, 0.0, 0.0], dtype=np.float64)
    env.rel_vel = np.array([0.0, 0.0, 0.0], dtype=np.float64)

    action = np.array([-0.1, 0.0, 0.0], dtype=np.float32)

    # Step 1
    _, reward_1, _, _, _ = env.step(action)
    expected_reward_1 = 0.4488276185
    assert reward_1 == pytest.approx(expected_reward_1, abs=1e-7)

    # Step 2
    _, reward_2, _, _, _ = env.step(action)
    expected_reward_2 = 0.8986568832
    assert reward_2 == pytest.approx(expected_reward_2, abs=1e-7)


def test_docking_bonus_at_close_distance():
    """Test 3: +100 docking bonus triggers when dist_curr < 0.5."""
    env = SpacecraftEnv()
    env.reset(seed=42)
    env.rel_pos = np.array([0.4, 0.0, 0.0], dtype=np.float64)  # dist_prev = 0.4
    env.rel_vel = np.array([0.0, 0.0, 0.0], dtype=np.float64)

    action = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    _, reward, _, _, info = env.step(action)

    assert info["distance"] < 0.5
    # Should include +100 docking bonus
    assert reward > 90.0


def test_observation_rel_pos_normalization():
    """
    Test 4: assert normalized rel_pos values stay in a roughly [-1.5, 1.5] range
    for a randomly sampled batch of valid initial environment states.
    """
    env = SpacecraftEnv()
    for i in range(50):
        obs, _ = env.reset(seed=100 + i)
        norm_pos = obs[0:3]
        # Starting positions are in [-20, 20]m, normalized by 20.0 -> [-1.0, 1.0]
        assert np.all(norm_pos >= -1.5), f"norm_pos {norm_pos} below -1.5"
        assert np.all(norm_pos <= 1.5), f"norm_pos {norm_pos} above 1.5"


def test_reset_then_step_real_call():
    """
    Test 5: reset() then step(valid random action) returns correctly-shaped obs
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
    """Test 6: Fuel depletion sets truncated=True and terminated=False (when dist >= 0.5)."""
    env = SpacecraftEnv()
    env.reset(seed=42)
    env.rel_pos = np.array([10.0, 0.0, 0.0], dtype=np.float64)  # dist = 10 >= 0.5
    env.fuel = 0.0  # Fuel is depleted

    action = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    obs, reward, terminated, truncated, info = env.step(action)

    assert not terminated, "Fuel depletion should not set terminated=True"
    assert truncated, "Fuel depletion should set truncated=True"
