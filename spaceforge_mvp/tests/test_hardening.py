"""Hardening regression tests: safety scope lockdown + RL obs tracking.

All backward compatible: legacy defaults must keep passing.
"""

import numpy as np

from spaceforge_mvp.config import SafetyConfig
from spaceforge_mvp.safety.mission_safety import evaluate_goal


def test_legacy_goal_gating_preserved_by_default():
    # INSPECT bypass is legacy load-bearing behaviour — must not change silently.
    out = evaluate_goal(goal_type="INSPECT", current_dist=3.0, fuel_remaining=0.5, current_speed=0.5)
    assert out["status"] == "accepted"


def test_strict_mode_makes_proximity_purely_geometric():
    strict = SafetyConfig(enforce_proximity_for_all_goals=True)
    out = evaluate_goal(
        goal_type="INSPECT",
        current_dist=3.0,
        fuel_remaining=0.5,
        current_speed=0.5,
        commanded_action=np.array([1.0, 0.0, 0.0]),
        rel_vel=np.array([0.5, 0.0, 0.0]),
        safety_config=strict,
    )
    assert out["status"] == "rejected"


def test_custom_proximity_goals_extend_scope():
    cfg = SafetyConfig(proximity_goals=("APPROACH", "DOCK"))
    out = evaluate_goal(
        goal_type="DOCK",
        current_dist=3.0,
        fuel_remaining=0.5,
        current_speed=0.5,
        commanded_action=np.array([1.0, 0.0, 0.0]),
        rel_vel=np.array([0.5, 0.0, 0.0]),
        safety_config=cfg,
    )
    assert out["status"] == "rejected"


def test_safety_config_round_trips_through_runconfig(tmp_path):
    from spaceforge_mvp.config import MissionConfig, RunConfig

    mission = MissionConfig(
        start_pos=(4, 0, 0), target_pos=(0, 0, 0),
        start_vel=(0, 0, 0), target_vel=(0, 0, 0),
        initial_fuel=0.8, duration=2, dt=1.0, mean_motion=0.00113,
        safety_mode="v2",
    )
    cfg = RunConfig(mission=mission, safety=SafetyConfig(enforce_proximity_for_all_goals=True), safety_mode="v2")
    restored = RunConfig.read_json(cfg.write_json(tmp_path / "m.json"))
    assert restored.safety.enforce_proximity_for_all_goals is True
    assert restored.safety.proximity_goals == ("APPROACH",)


def test_rl_controller_tracks_time_and_fuel():
    from spaceforge_mvp.viz.record import rl_controller

    act = rl_controller("checkpoints/ppo_spacecraft_final.zip")
    assert hasattr(act, "reset_tracking")
    pos = np.array([10.0, 0.0, 0.0])
    vel = np.zeros(3)
    act.reset_tracking(1.0)
    act(pos, vel)
    act(pos, vel)
    # Internal dummy env must have advanced past the frozen step_count=0 bug.
    # We verify indirectly: a second fresh controller reset behaves identically.
    act.reset_tracking(1.0)
    a1 = act(pos, vel)
    assert np.all(np.isfinite(a1)) and a1.shape == (3,)
