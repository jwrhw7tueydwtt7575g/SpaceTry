"""Transport latency/noise tests: delay line semantics, seeded noise (offline)."""
import numpy as np
import pytest

from spaceforge_mvp.config import MissionConfig, PhysicsConfig, RunConfig
from spaceforge_mvp.env.spacecraft_env import SpacecraftEnv
from spaceforge_rl_env.env import SpacecraftRLEnv
from spaceforge_rl_env.transport import MockTransport


def backend(dt=1.0, seed_run=0):
    mission = MissionConfig(
        start_pos=(5.0, 0.0, 0.0), target_pos=(0.0, 0.0, 0.0),
        start_vel=(0.0, 0.0, 0.0), target_vel=(0.0, 0.0, 0.0),
        initial_fuel=1.0, duration=10.0, dt=dt, mean_motion=0.00113,
        safety_mode="off",
    )
    run = RunConfig(mission=mission, physics=PhysicsConfig(n=mission.mean_motion, dt=dt),
                    seed=seed_run)
    return SpacecraftEnv(run_config=run)


def test_delay_steps_quantize_conservatively():
    assert MockTransport(backend(dt=1.0)).delay_steps == 0
    assert MockTransport(backend(dt=1.0), latency_ms=5.0).delay_steps == 1
    assert MockTransport(backend(dt=1.0), latency_ms=10.0).delay_steps == 1
    assert MockTransport(backend(dt=0.01), latency_ms=5.0).delay_steps == 1
    assert MockTransport(backend(dt=0.01), latency_ms=10.0).delay_steps == 1
    assert MockTransport(backend(dt=0.01), latency_ms=25.0).delay_steps == 3
    with pytest.raises(ValueError):
        MockTransport(backend(), latency_ms=-1.0)
    with pytest.raises(ValueError):
        MockTransport(backend(), thrust_noise_std=-0.1)


def test_zero_latency_matches_legacy_behavior():
    env = SpacecraftRLEnv(backend=backend())
    env.reset(seed=3)
    before = env.transport.backend.fuel
    env.transport.publish_wrench(np.array([1.0, 0.0, 0.0]), np.zeros(3), 0)
    env.transport.wait_for_sensor(0, 0.25)
    assert env.transport.backend.fuel < before  # command applied immediately


def test_latency_delays_execution_by_whole_steps():
    transport = MockTransport(backend(dt=0.01), latency_ms=10.0)
    assert transport.delay_steps == 1
    env = SpacecraftRLEnv(backend=transport.backend, transport=transport, max_steps=10)
    env.reset(seed=3)
    transport.publish_wrench(np.array([1.0, 0.0, 0.0]), np.zeros(3), 0)
    fuel_after_first_wait = transport.backend.fuel
    transport.wait_for_sensor(0, 0.25)
    assert transport.backend.fuel == fuel_after_first_wait == 1.0  # zeros executed first
    transport.publish_wrench(np.zeros(3), np.zeros(3), 1)
    transport.wait_for_sensor(1, 0.25)
    assert transport.backend.fuel < 1.0  # delayed command now executes


def test_transport_noise_is_seeded():
    def run(seed):
        transport = MockTransport(backend(), thrust_noise_std=0.1)
        env = SpacecraftRLEnv(backend=transport.backend, transport=transport)
        env.reset(seed=seed)
        positions = []
        for _ in range(5):
            transport.publish_wrench(np.array([0.5, 0.0, 0.0]), np.zeros(3), 0)
            transport.wait_for_sensor(0, 0.25)
            positions.append(tuple(float(v) for v in transport.backend.rel_pos))
        return positions

    first, second, other = run(4), run(4), run(5)
    assert first == second  # identical seeds replay identically
    assert first != other


def test_transport_mass_scales_response():
    def drift(mass):
        transport = MockTransport(backend(), mass_kg=mass)
        env = SpacecraftRLEnv(backend=transport.backend, transport=transport)
        env.reset(seed=1)
        transport.publish_wrench(np.array([1.0, 0.0, 0.0]), np.zeros(3), 0)
        transport.wait_for_sensor(0, 0.25)
        return float(np.linalg.norm(transport.backend.rel_vel))
    assert drift(1.0) > drift(2.0) > 0.0
