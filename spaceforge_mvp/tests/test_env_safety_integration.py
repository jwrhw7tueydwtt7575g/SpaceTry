"""Integration tests for SpacecraftEnv safety interlock."""

import numpy as np
from spaceforge_mvp.env.spacecraft_env import SpacecraftEnv


def test_fuel_below_floor_forces_emergency_hold_and_zeroes_action():
    """Test 1: Manually force fuel below FUEL_FLOOR, step with nonzero action, assert action zeroed."""
    env = SpacecraftEnv()
    env.reset(seed=42)

    # Force fuel below floor (0.05)
    env.fuel = 0.02

    nonzero_action = np.array([0.5, -0.5, 0.2], dtype=np.float32)
    obs, reward, terminated, truncated, info = env.step(nonzero_action)

    assert info["safety_override"] == "emergency_hold"
    np.testing.assert_allclose(info["applied_action"], [0.0, 0.0, 0.0])


def test_normal_state_step_is_unaffected():
    """Test 2: Normal state step is unaffected (safety_override is None) and action applied."""
    env = SpacecraftEnv()
    env.reset(seed=42)
    env.rel_pos = np.array([10.0, 0.0, 0.0], dtype=np.float64)
    env.fuel = 0.5

    action = np.array([0.2, 0.1, -0.1], dtype=np.float32)
    obs, reward, terminated, truncated, info = env.step(action)

    assert info["safety_override"] is None
    np.testing.assert_allclose(info["applied_action"], [0.2, 0.1, -0.1], atol=1e-6)
