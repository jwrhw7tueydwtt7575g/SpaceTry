"""Tests for safety interlock integration within SpacecraftEnv."""

import sys
from pathlib import Path
import numpy as np
import pytest

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

try:
    from spaceforge_mvp.env.spacecraft_env import SpacecraftEnv
    from spaceforge_mvp.safety.mission_safety import FUEL_FLOOR, SAFE_APPROACH_DIST
except ImportError:
    from env.spacecraft_env import SpacecraftEnv
    from safety.mission_safety import FUEL_FLOOR, SAFE_APPROACH_DIST


def test_fuel_below_floor_forces_emergency_hold_and_zeroes_action():
    """
    Test 1: Manually force fuel below FUEL_FLOOR in the env state, step with a
    nonzero action, and assert the action was zeroed and
    info["safety_override"] == "emergency_hold".
    """
    env = SpacecraftEnv()
    env.reset(seed=42)

    # Place spacecraft at a safe distance but deplete fuel below FUEL_FLOOR
    env.rel_pos = np.array([15.0, 0.0, 0.0], dtype=np.float64)
    env.fuel = FUEL_FLOOR - 0.01  # e.g. 0.04 < 0.05
    initial_fuel = env.fuel

    nonzero_action = np.array([0.6, -0.4, 0.2], dtype=np.float32)
    obs, reward, terminated, truncated, info = env.step(nonzero_action)

    assert info["safety_override"] == "emergency_hold"
    assert np.allclose(info["applied_action"], 0.0)
    # Fuel must not have decreased since commanded action was zeroed out
    assert env.fuel == initial_fuel


def test_normal_state_step_is_unaffected():
    """
    Test 2: A normal-state step is unaffected (safety_override is None) and the
    action is applied as given.
    """
    env = SpacecraftEnv()
    env.reset(seed=42)

    # Set normal state: distance > SAFE_APPROACH_DIST and fuel > FUEL_FLOOR
    env.rel_pos = np.array([12.0, 0.0, 0.0], dtype=np.float64)
    env.rel_vel = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    env.fuel = 0.8
    initial_fuel = env.fuel

    action = np.array([0.3, -0.5, 0.2], dtype=np.float32)
    obs, reward, terminated, truncated, info = env.step(action)

    assert info["safety_override"] is None
    assert np.allclose(info["applied_action"], action)

    # Fuel should be properly decremented by sum(|action|)*0.001
    expected_fuel = initial_fuel - float(np.sum(np.abs(action)) * 0.001)
    assert env.fuel == pytest.approx(expected_fuel, abs=1e-7)
