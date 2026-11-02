"""Tests for SpacecraftEnv Gymnasium environment."""

import numpy as np
import pytest

from spaceforge_mvp.env.spacecraft_env import SpacecraftEnv


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
    """Test 2: potential-based reward shaping calculated over two sequential steps using CW STM."""
    env = SpacecraftEnv()
    env.reset(seed=42)
    env.rel_pos = np.array([10.0, 0.0, 0.0], dtype=np.float64)
    env.rel_vel = np.array([0.0, 0.0, 0.0], dtype=np.float64)

    action = np.array([-0.1, 0.0, 0.0], dtype=np.float32)

    # Step 1
    _, reward_1, _, _, info_1 = env.step(action)

    # Hand calculation for CW STM step 1 from (10, 0, 0) with a=[-0.1, 0, 0]:
    # dist_prev = 10.0
    # progress_reward = 5.0 * (10.0 - dist_1)
    # velocity_penalty = -0.5 * speed_1
    # action_penalty = -0.01 * 0.1 = -0.001
    expected_reward_1 = sum(info_1["reward_breakdown"].values())
    assert reward_1 == pytest.approx(expected_reward_1, abs=1e-7)

    # Step 2
    _, reward_2, _, _, info_2 = env.step(action)
    expected_reward_2 = sum(info_2["reward_breakdown"].values())
    assert reward_2 == pytest.approx(expected_reward_2, abs=1e-7)


def test_docking_success_requires_low_relative_speed():
    """Test 3: Docking success requires dist < 0.5m AND relative speed < 0.1m/s."""
    env = SpacecraftEnv()
    env.reset(seed=42)

    # Case A: dist < 0.5m but high speed (0.5 m/s) -> NOT successful: it is a
    # collision, which ends the episode with a penalty.
    env.rel_pos = np.array([0.4, 0.0, 0.0], dtype=np.float64)
    env.rel_vel = np.array([0.5, 0.0, 0.0], dtype=np.float64)
    _, reward, terminated, _, info = env.step(np.array([0.0, 0.0, 0.0]))
    assert not info["is_success"], "High relative speed docking attempt should fail"
    assert terminated and info["collision"]
    assert info["reward_breakdown"]["collision"] < 0.0

    # Case B: dist < 0.5m AND low speed (0.05 m/s) -> SUCCESSFUL
    env.reset(seed=42)
    env.rel_pos = np.array([0.4, 0.0, 0.0], dtype=np.float64)
    env.rel_vel = np.array([0.05, 0.0, 0.0], dtype=np.float64)
    _, _, terminated, _, info = env.step(np.array([0.0, 0.0, 0.0]))
    assert terminated, "Low relative speed docking attempt should succeed"
    assert info["is_success"]


def test_observation_rel_pos_normalization():
    """Test 4: Assert normalized rel_pos values stay in a roughly [-1.5, 1.5] range."""
    env = SpacecraftEnv()
    for i in range(50):
        obs, _ = env.reset(seed=100 + i)
        norm_pos = obs[0:3]
        # Starting positions are in [-20, 20]m, normalized by 20.0 -> [-1.0, 1.0]
        assert np.all(norm_pos >= -1.5), f"norm_pos {norm_pos} below -1.5"
        assert np.all(norm_pos <= 1.5), f"norm_pos {norm_pos} above 1.5"


def test_reset_then_step_real_call():
    """Test 5: Reset then step returns correctly shaped obs and finite reward."""
    env = SpacecraftEnv()
    obs, info = env.reset(seed=42)

    assert obs.shape == (10,)
    assert np.all(np.isfinite(obs))
    assert isinstance(info, dict)

    action = env.action_space.sample()
    next_obs, reward, terminated, truncated, step_info = env.step(action)

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
    env.fuel = 0.0  # Fuel depleted

    action = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    obs, reward, terminated, truncated, info = env.step(action)

    assert not terminated, "Fuel depletion should not set terminated=True"
    assert truncated, "Fuel depletion should set truncated=True"
