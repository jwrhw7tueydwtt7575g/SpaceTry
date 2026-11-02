"""Regression tests for defects found in the external review.

Each test pins one previously broken behaviour so it cannot silently return.
"""

import argparse
from pathlib import Path

import numpy as np
import pytest

from spaceforge_mvp.config import (
    ActuatorConfig, FaultConfig, MissionConfig, PhysicsConfig, RunConfig,
)
from spaceforge_mvp.control.glideslope import glideslope_controller, hot_approach_action
from spaceforge_mvp.env.spacecraft_env import SpacecraftEnv
from spaceforge_mvp.viz.record import record_episode

N = 0.00113


def _run(start, duration=300.0, dt=1.0, n=N, accel=1.0, safety="v2", start_vel=(0.0, 0.0, 0.0)):
    mission = MissionConfig(start, (0, 0, 0), start_vel, (0, 0, 0), 1.0, duration, dt, n, safety)
    return RunConfig(mission=mission, physics=PhysicsConfig(n=n, dt=dt),
                     actuator=ActuatorConfig(max_cmd_accel=accel), safety_mode=safety)


# ---------------------------------------------------------------- recorder
def test_recorder_stops_when_the_vehicle_leaves_the_work_volume():
    rc = _run((10.0, 0.0, 0.0), safety="off")
    away = lambda p, v: p / np.linalg.norm(p)  # noqa: E731 - full thrust away from the target
    summary = record_episode(away, seed=0, run_config=rc, safety="off")["summary"]
    assert summary["terminal_reason"] == "out_of_bounds"
    assert summary["steps"] < 30
    assert summary["end_dist"] < 2 * rc.environment.work_volume_bound


def test_fly_through_is_a_collision_not_a_timeout():
    rc = _run((6.0, 0.0, 0.0), duration=30.0, safety="off")
    summary = record_episode(hot_approach_action, seed=0, run_config=rc, safety="off",
                             annotate_continuous=True)["summary"]
    assert summary["terminal_reason"] == "collision"
    assert summary["penetrated"] and not summary["success"]


# ---------------------------------------------------------------- faults
@pytest.mark.parametrize("faults", [
    FaultConfig(stuck_axis=0, stuck_value=1.0),
    FaultConfig(bias=(1.0, 0.0, 0.0)),
])
def test_faults_do_not_alias_the_interlock_decision_and_are_billed(faults):
    mission = MissionConfig((10, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), 1.0, 50, 1.0, N, "off")
    env = SpacecraftEnv(mission=mission, faults=faults)
    env.reset(seed=0)
    for _ in range(10):
        _, _, _, _, info = env.step(np.zeros(3))
    np.testing.assert_allclose(info["applied_action"], 0.0)       # what the filter allowed
    np.testing.assert_allclose(info["thruster_command"], [1, 0, 0])  # what actually fired
    assert env.fuel == pytest.approx(1.0 - 10 * env.actuator_config.fuel_rate_scale)


# ---------------------------------------------------------------- glideslope binding
@pytest.mark.parametrize("kwargs", [
    {"n": 0.05},                                   # feed-forward must use the mission's n
    {"dt": 5.0, "start": (12.0, -7.0, 3.5)},       # gain must stay stable at large dt
    {"accel": 0.01, "duration": 900.0, "start": (15.0, -10.0, 5.0)},  # physical units
])
def test_glideslope_bound_to_run_config_docks(kwargs):
    start = kwargs.pop("start", (40.0, -30.0, 10.0))
    rc = _run(start, **kwargs)
    summary = record_episode(glideslope_controller(rc), seed=0, run_config=rc, safety="v2",
                             annotate_continuous=True)["summary"]
    assert summary["success"], summary
    assert not summary["penetrated"]


# ---------------------------------------------------------------- CLI
def test_cli_reports_a_missing_checkpoint_cleanly(tmp_path):
    from spaceforge_mvp.simulate import build_parser, run

    args = build_parser().parse_args([
        "--controller", "rl", "--checkpoint", str(tmp_path / "missing.zip"),
        "--start-pos", "1", "0", "0", "--target-pos", "0", "0", "0",
        "--start-vel", "0", "0", "0", "--target-vel", "0", "0", "0",
        "--fuel", "1", "--duration", "10", "--dt", "1", "--mean-motion", str(N),
        "--safety", "v1", "--output-dir", str(tmp_path / "out"),
    ])
    with pytest.raises(ValueError, match="checkpoint not found"):
        run(args)


def test_cli_goal_label_cannot_disable_v2(tmp_path):
    from spaceforge_mvp.simulate import run

    args = argparse.Namespace(
        controller="hot", checkpoint=None, start_pos=[6, 0, 0], target_pos=[0, 0, 0],
        start_vel=[0, 0, 0], target_vel=[0, 0, 0], fuel=1.0, duration=120.0, dt=1.0,
        mean_motion=N, safety="v2", goal="DOCK", seed=42, output_dir=tmp_path,
        nondeterministic=False,
    )
    summary = run(args)["summary"]
    assert not summary["penetrated"]
    assert summary["success"]


# ---------------------------------------------------------------- visualisation
def test_broadcast_renderer_draws_a_frame(tmp_path):
    """render.py had 0% coverage and crashed under the old matplotlib pin."""
    from spaceforge_mvp.viz.preview import render_frames

    written = render_frames("docking", [0, 10], Path(tmp_path), dpi=30)
    assert len(written) == 2 and all(p.stat().st_size > 0 for p in written)


def test_dashboard_builds(tmp_path, monkeypatch):
    from spaceforge_mvp.viz import dashboard

    monkeypatch.setattr(dashboard, "OUT", tmp_path / "dashboard.html")
    out = dashboard.build()
    html = Path(out).read_text()
    assert "<html" in html.lower() and len(html) > 10_000


# ---------------------------------------------------------------- shipped policy
def test_shipped_checkpoint_docks():
    """The released PPO checkpoint must dock (the previous one scored 0/50)."""
    from stable_baselines3 import PPO

    model = PPO.load(str(Path(__file__).resolve().parents[2] / "checkpoints" / "ppo_spacecraft_final.zip"))
    env = SpacecraftEnv()
    successes = collisions = 0
    for seed in range(7000, 7010):
        obs, _ = env.reset(seed=seed)
        while True:
            obs, _, terminated, truncated, info = env.step(model.predict(obs, deterministic=True)[0])
            if terminated or truncated:
                break
        successes += int(info["is_success"])
        collisions += int(info["collision"])
    assert successes >= 9 and collisions == 0
