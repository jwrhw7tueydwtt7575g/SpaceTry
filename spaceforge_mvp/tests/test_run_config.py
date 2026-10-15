"""Configuration contracts shared by simulation, training, and replay."""

import pytest

from spaceforge_mvp.config import (
    ActuatorConfig,
    ControllerConfig,
    MissionConfig,
    PhysicsConfig,
    RunConfig,
)
from spaceforge_mvp.env.spacecraft_env import SpacecraftEnv
from spaceforge_mvp.viz.record import record_episode


def _run_config() -> RunConfig:
    mission = MissionConfig(
        start_pos=(12, -7, 3.5), target_pos=(2, 1, 0.5),
        start_vel=(1, 2, 3), target_vel=(0.2, 0.5, 1),
        initial_fuel=0.8, duration=4, dt=0.5, mean_motion=0.00113,
        safety_mode="v2",
    )
    return RunConfig(
        mission=mission,
        physics=PhysicsConfig(n=0.00113, dt=0.5),
        actuator=ActuatorConfig(max_cmd_accel=0.25, fuel_rate_scale=0.02),
        controller=ControllerConfig(name="glideslope"),
        seed=19,
    )


def test_run_config_round_trip_preserves_nonzero_target(tmp_path):
    config = _run_config()
    path = config.write_json(tmp_path / "mission.json")
    restored = RunConfig.read_json(path)
    assert restored == config
    assert restored.mission.relative_pos == (10.0, -8.0, 3.0)
    assert restored.mission.relative_vel == (0.8, 1.5, 2.0)


@pytest.mark.parametrize("kwargs", [
    {"start_pos": (1, 2)},
    {"initial_fuel": 1.1},
    {"duration": 1.0, "dt": 0.3},
    {"safety_mode": "unsafe"},
])
def test_mission_config_rejects_invalid_values(kwargs):
    base = dict(
        start_pos=(1, 0, 0), target_pos=(0, 0, 0),
        start_vel=(0, 0, 0), target_vel=(0, 0, 0),
        initial_fuel=0.5, duration=1.0, dt=0.5, mean_motion=0.00113,
    )
    base.update(kwargs)
    if kwargs == {"duration": 1.0, "dt": 0.3}:
        with pytest.raises(ValueError, match="integer multiple"):
            _ = MissionConfig(**base).max_steps
    else:
        with pytest.raises(ValueError):
            MissionConfig(**base)


def test_resolved_config_drives_environment_and_replay_reproducibly():
    config = _run_config()
    env = SpacecraftEnv(run_config=config)
    env.reset(seed=config.seed)
    assert env.max_steps == 8
    assert env.fuel == 0.8
    _, _, _, _, info = env.step([1, 0, 0])
    assert info["fuel"] == pytest.approx(0.78)

    zero = lambda pos, vel: [0, 0, 0]
    first = record_episode(zero, run_config=config, seed=config.seed)
    second = record_episode(zero, run_config=config, seed=config.seed)
    assert first == second
