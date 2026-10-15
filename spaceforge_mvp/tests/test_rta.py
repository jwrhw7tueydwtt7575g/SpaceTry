"""Tests for the runtime-assurance filter (v2 substitution).

The claim under test: a veto that *zeroes* thrust does not stop an already-fast
vehicle, because doing nothing is not braking. Substituting a certified braking
action instead changes the outcome of the same reckless approach from a
fly-through into a soft capture.
"""

import numpy as np

from spaceforge_mvp.config import ENV_CONFIG, PHYSICS_CONFIG
from spaceforge_mvp.control.glideslope import hot_approach_action
from spaceforge_mvp.control.rta import (
    BRAKING_SUBSTITUTED,
    apply_safety_filter,
    certified_brake_action,
)
from spaceforge_mvp.core.cw_dynamics import cw_step
from spaceforge_mvp.safety.mission_safety import evaluate_goal

SLAM_START = np.array([6.0, 0.0, 0.0])


def _rollout(safety: str, steps: int = 30):
    """Roll the reckless slam under one safety mode; return per-step (dist, speed) and success."""
    pos = SLAM_START.copy()
    vel = np.zeros(3)
    fuel = 1.0
    log = []
    for _ in range(steps):
        cmd = hot_approach_action(pos, vel)
        override = None
        if safety == "v1":
            verdict = evaluate_goal(
                goal_type="APPROACH",
                current_dist=float(np.linalg.norm(pos)),
                fuel_remaining=fuel,
                current_speed=float(np.linalg.norm(vel)),
                commanded_action=cmd,
                rel_vel=vel,
            )
            if verdict["status"] == "accepted":
                applied = cmd
            else:
                applied, override = np.zeros(3), verdict["status"]
        elif safety == "v2":
            decision = apply_safety_filter(cmd, pos, vel, fuel)
            applied, override = decision["applied"], decision["override"]
        else:
            applied = cmd

        prev_speed = float(np.linalg.norm(vel))
        pos, vel = cw_step(pos, vel, applied, dt=PHYSICS_CONFIG.dt)
        fuel = max(0.0, fuel - ENV_CONFIG.fuel_rate_scale * float(np.sum(np.abs(applied))))
        dist, speed = float(np.linalg.norm(pos)), float(np.linalg.norm(vel))
        log.append((dist, speed, prev_speed, override))
        if dist < ENV_CONFIG.docking_success_dist and speed < 0.1:
            return log, True
    return log, False


# --------------------------------------------------------------------- unit rules
def test_accepted_command_passes_through_untouched():
    """Outside the keep-out bubble the policy keeps full authority."""
    cmd = np.array([0.3, -0.2, 0.1])
    out = apply_safety_filter(
        commanded_action=cmd,
        rel_pos=np.array([12.0, 0.0, 0.0]),
        rel_vel=np.zeros(3),
        fuel_remaining=1.0,
    )
    assert out["override"] is None
    np.testing.assert_allclose(out["applied"], cmd)


def test_vetoed_command_is_replaced_by_braking_not_by_zero():
    """Inside the bubble at excessive speed, thrust is revoked -- and replaced."""
    vel = np.array([0.6, 0.0, 0.0])  # closing at 0.6 m/s, limit is 0.2
    cmd = np.array([1.0, 0.0, 0.0])  # still accelerating: unsafe
    out = apply_safety_filter(cmd, np.array([3.0, 0.0, 0.0]), vel, fuel_remaining=0.5)

    assert out["override"] == BRAKING_SUBSTITUTED
    np.testing.assert_allclose(out["applied"], [-1.0, 0.0, 0.0], atol=1e-9)
    assert float(np.dot(out["applied"], vel)) < 0.0


def test_fuel_floor_still_wins_over_everything():
    """Below the fuel floor the craft holds: no thrust, not even a braking one."""
    out = apply_safety_filter(
        commanded_action=np.array([-1.0, -1.0, -1.0]),
        rel_pos=np.array([3.0, 0.0, 0.0]),
        rel_vel=np.array([0.6, 0.0, 0.0]),
        fuel_remaining=0.01,
    )
    assert out["override"] == "emergency_hold"
    np.testing.assert_allclose(out["applied"], [0.0, 0.0, 0.0])


def test_braking_action_never_increases_speed():
    """Certified braking is anti-parallel to velocity for arbitrary directions."""
    for speed in (0.05, 0.5, 3.0):
        for direction in (
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, -1.0, 0.3]),
            np.array([0.2, 0.1, -1.0]),
        ):
            vel = direction / np.linalg.norm(direction) * speed
            brake = certified_brake_action(vel)
            assert float(np.dot(brake, vel)) < 0.0
            np.testing.assert_allclose(np.linalg.norm(brake), 1.0, atol=1e-9)


# ----------------------------------------------------------------- behaviour claims
def test_v2_keeps_the_speed_non_increasing_whenever_it_takes_over():
    """Every substituted step must reduce (or hold) relative speed. Provable property."""
    log, _ = _rollout("v2", steps=30)
    substituted = [row for row in log if row[3] == BRAKING_SUBSTITUTED]
    assert substituted, "expected the filter to take over at least once"
    for _, speed, prev_speed, _ in substituted:
        assert speed <= prev_speed + 1e-9


def test_v2_filter_turns_a_reckless_slam_into_a_soft_capture():
    """Same controller, same physics, same 1 s step: only the veto response differs."""
    log_open, ok_open = _rollout("off", steps=30)
    log_v1, ok_v1 = _rollout("v1", steps=30)
    log_v2, ok_v2 = _rollout("v2", steps=30)

    # No runtime assurance, and the MVP's thrust-denial version: neither docks.
    assert not ok_open and not ok_v1

    # v1 does reach the capture radius -- at a speed that would be a collision.
    closest_v1 = min(log_v1, key=lambda row: row[0])
    assert closest_v1[0] < ENV_CONFIG.docking_success_dist
    assert closest_v1[1] > 1.0, "the coasting pass should still be metres per second"

    # v2 recovers the identical approach into a soft capture.
    assert ok_v2, "certified braking should recover the reckless slam"
    end_dist, end_speed, _, _ = log_v2[-1]
    assert end_dist < ENV_CONFIG.docking_success_dist
    assert end_speed < 0.1
