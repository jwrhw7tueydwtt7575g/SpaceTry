"""Phase 6 domain randomization and fault injection tests (offline, fast)."""
import numpy as np
import pytest

from spaceforge_mvp.config import DomainConfig, FaultConfig, RunConfig
from spaceforge_mvp.env.spacecraft_env import SpacecraftEnv


def rollout(env, seed, actions):
    env.reset(seed=seed)
    trace = []
    for action in actions:
        obs, _, terminated, truncated, info = env.step(np.asarray(action, dtype=float))
        trace.append((env.rel_pos.copy(), env.rel_vel.copy(), obs.copy(), dict(info)))
        if terminated or truncated:
            break
    return trace


ACTIONS = [np.array([0.3, -0.2, 0.1]), np.array([-0.1, 0.4, -0.3]), np.zeros(3)]


def test_identity_domain_matches_no_domain_bit_for_bit():
    plain = rollout(SpacecraftEnv(), 7, ACTIONS)
    domed = rollout(SpacecraftEnv(domain=DomainConfig(), faults=FaultConfig()), 7, ACTIONS)
    assert len(plain) == len(domed)
    for (p_pos, p_vel, p_obs, _), (d_pos, d_vel, d_obs, _) in zip(plain, domed):
        np.testing.assert_allclose(d_pos, p_pos)
        np.testing.assert_allclose(d_vel, p_vel)
        np.testing.assert_allclose(d_obs, p_obs)


def test_mass_scaling_and_reproducibility():
    heavy = DomainConfig(mass_low_kg=2.0, mass_high_kg=2.0)
    light = rollout(SpacecraftEnv(), 3, ACTIONS)
    heavy_trace = rollout(SpacecraftEnv(domain=heavy), 3, ACTIONS)
    # Heavier vehicle responds less to the same command.
    assert float(np.linalg.norm(heavy_trace[0][0] - light[0][0])) > 0.0
    assert heavy_trace[0][3]["domain"]["mass_kg"] == pytest.approx(2.0)
    repeat = rollout(SpacecraftEnv(domain=heavy), 3, ACTIONS)
    for (_, _, _, info_a), (_, _, _, info_b) in zip(heavy_trace, repeat):
        np.testing.assert_allclose(info_a["applied_action"], info_b["applied_action"])
    varied = rollout(SpacecraftEnv(domain=DomainConfig(mass_low_kg=0.5, mass_high_kg=2.0)), 3, ACTIONS)
    assert 0.5 <= varied[0][3]["domain"]["mass_kg"] <= 2.0


def test_thrust_noise_is_seeded_and_bounded():
    noisy = DomainConfig(thrust_noise_std=0.05)
    first = rollout(SpacecraftEnv(domain=noisy), 11, ACTIONS)
    second = rollout(SpacecraftEnv(domain=noisy), 11, ACTIONS)
    for (pos_a, _, _, info_a), (pos_b, _, _, info_b) in zip(first, second):
        np.testing.assert_allclose(pos_a, pos_b)  # same seed, same noise
        np.testing.assert_allclose(info_a["executed_action"], info_b["executed_action"])
    clean = rollout(SpacecraftEnv(), 11, ACTIONS)
    assert any(
        not np.allclose(i["executed_action"], c["executed_action"])
        for (_, _, _, i), (_, _, _, c) in zip(first, clean)
    )
    assert all(
        np.all(np.abs(i["executed_action"]) <= 1.0 + 1e-9) for _, _, _, i in first
    )


def test_observation_noise_leaves_truth_exact():
    noisy = DomainConfig(obs_pos_noise_std_m=0.1, obs_vel_noise_std_mps=0.02)
    truth = rollout(SpacecraftEnv(), 5, ACTIONS)
    sensed = rollout(SpacecraftEnv(domain=noisy), 5, ACTIONS)
    for (t_pos, t_vel, t_obs, _), (s_pos, s_vel, s_obs, _) in zip(truth, sensed):
        np.testing.assert_allclose(s_pos, t_pos)  # truth untouched
        np.testing.assert_allclose(s_vel, t_vel)
    assert any(
        not np.allclose(s_obs, t_obs) for (_, _, t_obs, _), (_, _, s_obs, _) in zip(truth, sensed)
    )


def test_fuel_randomization_bounds():
    domain = DomainConfig(fuel_low=0.4, fuel_high=0.6)
    fuels = set()
    for seed in range(10):
        env = SpacecraftEnv(domain=domain)
        env.reset(seed=seed)
        assert 0.4 <= env.fuel <= 0.6
        fuels.add(round(env.fuel, 6))
    assert len(fuels) > 1  # actually randomized, not constant
    fixed = SpacecraftEnv()
    fixed.reset(seed=0)
    assert fixed.fuel == 1.0


@pytest.mark.parametrize("kwargs", [
    {"mass_low_kg": 2.0, "mass_high_kg": 1.0},
    {"mass_low_kg": -1.0},
    {"effectiveness_tol": 1.0},
    {"thrust_noise_std": -0.1},
    {"fuel_low": 0.8, "fuel_high": 0.2},
    {"stuck_axis": 3},
    {"stuck_value": 2.0},
    {"dropout_prob": 1.5},
    {"bias": (float("nan"), 0.0, 0.0)},
])
def test_invalid_domain_and_fault_configs_rejected(kwargs):
    if set(kwargs) <= {"mass_low_kg", "mass_high_kg", "effectiveness_tol", "thrust_noise_std",
                       "obs_pos_noise_std_m", "obs_vel_noise_std_mps", "fuel_low", "fuel_high",
                       "mass_nominal_kg"}:
        base = dict(mass_nominal_kg=1.0, mass_low_kg=1.0, mass_high_kg=1.0,
                    effectiveness_tol=0.0, thrust_noise_std=0.0, obs_pos_noise_std_m=0.0,
                    obs_vel_noise_std_mps=0.0, fuel_low=1.0, fuel_high=1.0)
        base.update(kwargs)
        with pytest.raises(ValueError):
            DomainConfig(**base)
    else:
        base = dict(stuck_axis=None, stuck_value=0.0, bias=(0.0, 0.0, 0.0),
                    dropout_prob=0.0, sensor_freeze_prob=0.0, fuel_leak_rate=0.0)
        base.update(kwargs)
        with pytest.raises(ValueError):
            FaultConfig(**base)


def test_stuck_thruster_bias_dropout_leak_and_freeze():
    stuck_env = SpacecraftEnv(faults=FaultConfig(stuck_axis=1, stuck_value=-0.5))
    stuck = rollout(stuck_env, 9, ACTIONS)
    authority = stuck_env.actuator_config.max_cmd_accel
    assert all(info["thruster_command"][1] == pytest.approx(-0.5) for _, _, _, info in stuck)
    assert all(info["executed_action"][1] == pytest.approx(-0.5 * authority) for _, _, _, info in stuck)
    assert stuck[0][3]["fault"] == "stuck_thruster"

    leak_rate = 0.01
    leaked = rollout(SpacecraftEnv(faults=FaultConfig(fuel_leak_rate=leak_rate)), 9, ACTIONS)
    clean = rollout(SpacecraftEnv(), 9, ACTIONS)
    assert leaked[-1][3]["fuel"] < clean[-1][3]["fuel"]
    assert leaked[0][3]["fault"] == "fuel_leak"

    frozen = SpacecraftEnv(faults=FaultConfig(sensor_freeze_prob=1.0))
    trace = rollout(frozen, 9, ACTIONS)
    assert all(np.allclose(obs, trace[0][2]) for _, _, obs, _ in trace)

    dropped = rollout(SpacecraftEnv(faults=FaultConfig(dropout_prob=1.0)), 9, ACTIONS)
    assert all(info["fault"] == "command_dropout" for _, _, _, info in dropped)


def test_faults_are_deterministic_per_seed():
    faults = FaultConfig(dropout_prob=0.3, sensor_freeze_prob=0.3)
    first = rollout(SpacecraftEnv(faults=faults), 21, ACTIONS * 3)
    second = rollout(SpacecraftEnv(faults=faults), 21, ACTIONS * 3)
    for (_, _, obs_a, info_a), (_, _, obs_b, info_b) in zip(first, second):
        np.testing.assert_allclose(obs_a, obs_b)
        assert (info_a["fault"] is None) == (info_b["fault"] is None)


def test_run_config_round_trip_with_domain_and_faults(tmp_path):
    from spaceforge_mvp.config import MissionConfig, PhysicsConfig
    mission = MissionConfig(
        start_pos=(4, 0, 0), target_pos=(0, 0, 0),
        start_vel=(0, 0, 0), target_vel=(0, 0, 0),
        initial_fuel=0.9, duration=2, dt=1.0, mean_motion=0.00113,
        safety_mode="v1",
    )
    config = RunConfig(
        mission=mission, physics=PhysicsConfig(n=0.00113, dt=1.0),
        domain=DomainConfig(mass_low_kg=0.8, mass_high_kg=1.2, thrust_noise_std=0.01),
        faults=FaultConfig(dropout_prob=0.05), seed=3,
    )
    restored = RunConfig.read_json(config.write_json(tmp_path / "mission.json"))
    assert restored == config
    assert restored.domain.mass_high_kg == pytest.approx(1.2)
    assert restored.faults.dropout_prob == pytest.approx(0.05)
