"""Tests for explicit mission inputs and the command-line simulator."""

import json

import numpy as np

from spaceforge_mvp.config import MissionConfig
from spaceforge_mvp.env.spacecraft_env import SpacecraftEnv
from spaceforge_mvp.simulate import build_parser, run


def test_mission_config_derives_target_relative_state():
    mission = MissionConfig(
        start_pos=(12.0, -7.0, 3.5),
        target_pos=(2.0, 1.0, 0.5),
        start_vel=(1.0, 2.0, 3.0),
        target_vel=(0.2, 0.5, 1.0),
        initial_fuel=0.7,
        duration=10.0,
        dt=0.5,
        mean_motion=0.00113,
    )
    env = SpacecraftEnv(mission=mission)
    env.reset(seed=1)
    np.testing.assert_allclose(env.rel_pos, [10.0, -8.0, 3.0])
    np.testing.assert_allclose(env.rel_vel, [0.8, 1.5, 2.0])
    assert env.fuel == 0.7
    assert env.max_steps == 20


def test_simulator_writes_resolved_inputs_and_custom_timestamps(tmp_path):
    args = build_parser().parse_args([
        "--controller", "glideslope",
        "--start-pos", "4", "0", "0",
        "--target-pos", "1", "0", "0",
        "--start-vel", "0", "0", "0",
        "--target-vel", "0", "0", "0",
        "--fuel", "0.8", "--duration", "2", "--dt", "0.5",
        "--mean-motion", "0.00113", "--safety", "v2", "--output-dir", str(tmp_path),
    ])
    episode = run(args)
    mission = json.loads((tmp_path / "mission.json").read_text())
    assert mission["relative_pos"] == [3.0, 0.0, 0.0]
    assert episode["frames"][-1]["t"] == 2.0
    assert (tmp_path / "telemetry.json").exists()
    assert (tmp_path / "metrics.json").exists()
