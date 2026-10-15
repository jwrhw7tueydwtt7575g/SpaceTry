"""Independent numerical reference tests for the analytical CW propagator."""

import numpy as np
import pytest

from spaceforge_mvp.config import ActuatorConfig, ControllerConfig, MissionConfig, PhysicsConfig, RunConfig, SafetyConfig
from spaceforge_mvp.core.cw_dynamics import cw_step
from spaceforge_mvp.env.spacecraft_env import SpacecraftEnv
from spaceforge_mvp.viz.record import record_episode, replay_continuous_min


def _derivative(state: np.ndarray, accel: np.ndarray, n: float) -> np.ndarray:
    """CW ODE independently expressed as first-order state equations."""
    x, _, z, vx, vy, vz = state
    ax, ay, az = accel
    return np.array([
        vx, vy, vz,
        2.0 * n * vy + 3.0 * n * n * x + ax,
        -2.0 * n * vx + ay,
        -n * n * z + az,
    ])


def _rk4(pos: np.ndarray, vel: np.ndarray, accel: np.ndarray, dt: float, n: float) -> tuple[np.ndarray, np.ndarray]:
    state = np.concatenate([pos, vel]).astype(np.float64)
    k1 = _derivative(state, accel, n)
    k2 = _derivative(state + 0.5 * dt * k1, accel, n)
    k3 = _derivative(state + 0.5 * dt * k2, accel, n)
    k4 = _derivative(state + dt * k3, accel, n)
    result = state + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
    return result[:3], result[3:]


@pytest.mark.parametrize("accel", [np.zeros(3), np.array([0.03, -0.02, 0.01])])
def test_exact_stm_matches_independent_rk4_reference(accel):
    pos = np.array([14.0, -8.0, 3.0])
    vel = np.array([0.12, -0.08, 0.04])
    dt, n = 0.25, 0.002
    stm_pos, stm_vel = cw_step(pos, vel, accel, dt=dt, n=n)
    rk_pos, rk_vel = _rk4(pos, vel, accel, dt=dt, n=n)
    np.testing.assert_allclose(stm_pos, rk_pos, atol=2e-10, rtol=0.0)
    np.testing.assert_allclose(stm_vel, rk_vel, atol=2e-10, rtol=0.0)


def _configured_run() -> RunConfig:
    mission = MissionConfig(
        start_pos=(8.0, -2.0, 1.0), target_pos=(3.0, 1.0, -1.0),
        start_vel=(0.2, -0.1, 0.0), target_vel=(0.05, 0.0, -0.02),
        initial_fuel=0.5, duration=2.0, dt=0.25, mean_motion=0.002,
        safety_mode="off",
    )
    return RunConfig(
        mission=mission, physics=PhysicsConfig(n=0.002, dt=0.25),
        actuator=ActuatorConfig(max_cmd_accel=0.4, fuel_rate_scale=0.05),
        controller=ControllerConfig(name="glideslope"), safety_mode="off", seed=17,
    )


def test_custom_physics_target_state_and_fuel_are_shared_by_env_and_recorder():
    config = _configured_run()
    action = np.array([1.0, 0.0, 0.0])
    env = SpacecraftEnv(run_config=config)
    env.reset(seed=config.seed)
    np.testing.assert_allclose(env.rel_pos, [5.0, -3.0, 2.0])
    np.testing.assert_allclose(env.rel_vel, [0.15, -0.1, 0.02])
    expected_pos, expected_vel = cw_step(env.rel_pos, env.rel_vel, [0.4, 0.0, 0.0], dt=0.25, n=0.002)
    _, _, _, _, info = env.step(action)
    np.testing.assert_allclose(env.rel_pos, expected_pos)
    np.testing.assert_allclose(env.rel_vel, expected_vel)
    assert info["fuel"] == pytest.approx(0.45)

    episode = record_episode(lambda pos, vel: action, run_config=config, seed=config.seed, annotate_continuous=True)
    assert episode["frames"][1]["pos"] == pytest.approx(expected_pos, abs=1e-6)
    assert episode["frames"][-1]["t"] == pytest.approx(2.0)
    assert episode["summary"]["fuel_used"] == pytest.approx(0.4)


def test_replay_uses_custom_mean_motion_and_actuator_scaling():
    config = _configured_run()
    episode = record_episode(lambda pos, vel: np.array([1.0, 0.0, 0.0]), run_config=config, seed=17)
    replayed = replay_continuous_min(
        episode["frames"], sub_dt=0.25, interval_dt=0.25,
        mean_motion=config.physics.n, max_cmd_accel=config.actuator.max_cmd_accel,
    )
    assert replayed[0] == pytest.approx(episode["summary"]["min_dist"], abs=1e-4)


def test_fuel_depletion_truncates_before_timeout_with_custom_rate():
    base = _configured_run()
    config = RunConfig(
        mission=base.mission, physics=base.physics, safety=base.safety,
        environment=base.environment, controller=base.controller, safety_mode="off", seed=17,
        actuator=ActuatorConfig(max_cmd_accel=0.4, fuel_rate_scale=0.2),
    )
    env = SpacecraftEnv(run_config=config)
    env.reset(seed=17)
    _, _, terminated, truncated, info = env.step(np.ones(3))
    assert not terminated
    assert truncated
    assert info["fuel"] == pytest.approx(0.0)


def test_resolved_docking_threshold_and_v2_authority_are_applied():
    mission = MissionConfig(
        start_pos=(0.4, 0.0, 0.0), target_pos=(0.0, 0.0, 0.0),
        start_vel=(0.05, 0.0, 0.0), target_vel=(0.0, 0.0, 0.0),
        initial_fuel=0.5, duration=1.0, dt=0.01, mean_motion=0.00113,
        safety_mode="v2",
    )
    config = RunConfig(
        mission=mission, physics=PhysicsConfig(n=0.00113, dt=0.01),
        safety=SafetyConfig(safe_approach_dist=5.0, fuel_floor=0.05,
                            max_safe_closing_speed=0.02, max_docking_speed=0.1),
        controller=ControllerConfig(name="hot"), safety_mode="v2",
    )
    env = SpacecraftEnv(run_config=config)
    env.reset(seed=1)
    _, _, terminated, _, info = env.step(np.array([1.0, 0.0, 0.0]))
    assert info["safety_override"] == "braking_substituted"
    assert terminated
    assert info["is_success"]
