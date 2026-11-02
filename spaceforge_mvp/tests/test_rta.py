"""Tests for the runtime-assurance filter (v2 substitution).

The claims under test: a veto that *zeroes* thrust does not stop an
already-fast vehicle, because doing nothing is not braking; the v2 filter
predicts each step against a closing-speed envelope, and when it takes over it
never adds speed, never reverses the vehicle, never pushes a retreating vehicle
back toward the target, and never goes quiet below the fuel floor while on a
collision course.
"""

import numpy as np

from spaceforge_mvp.config import ENV_CONFIG, PHYSICS_CONFIG
from spaceforge_mvp.control.glideslope import hot_approach_action
from spaceforge_mvp.control.rta import (
    BRAKING_SUBSTITUTED,
    EMERGENCY_BRAKE,
    apply_safety_filter,
    certified_brake_action,
)
from spaceforge_mvp.core.cw_dynamics import cw_step
from spaceforge_mvp.safety.mission_safety import evaluate_goal

SLAM_START = np.array([6.0, 0.0, 0.0])
# These scenarios model a vehicle with 1 m/s^2 per unit command, stated explicitly
# rather than inherited from the ActuatorConfig default.
A1 = 1.0


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
                max_accel=A1,
            )
            if verdict["status"] == "accepted":
                applied = cmd
            else:
                applied, override = np.zeros(3), verdict["status"]
        elif safety == "v2":
            decision = apply_safety_filter(cmd, pos, vel, fuel, max_accel=A1)
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
        fuel_remaining=1.0, max_accel=A1)
    assert out["override"] is None
    np.testing.assert_allclose(out["applied"], cmd)


def test_vetoed_command_is_replaced_by_braking_not_by_zero():
    """Inside the bubble at excessive closing speed, thrust is revoked -- and replaced."""
    pos = np.array([3.0, 0.0, 0.0])
    vel = np.array([-0.6, 0.0, 0.0])  # closing at 0.6 m/s, envelope is ~0.12 here
    cmd = np.array([-1.0, 0.0, 0.0])  # still accelerating at the target: unsafe
    out = apply_safety_filter(cmd, pos, vel, fuel_remaining=0.5, max_accel=A1)

    assert out["override"] == BRAKING_SUBSTITUTED
    assert out["applied"][0] > 0.0  # pushes away from the target
    assert float(np.dot(out["applied"], vel)) < 0.0
    _, next_vel = cw_step(pos, vel, out["applied"], dt=1.0)
    assert np.linalg.norm(next_vel) < np.linalg.norm(vel)
    assert next_vel[0] <= 0.0, "the substitute must not over-brake into a reversal"


def test_coasting_into_the_target_is_not_accepted_as_braking():
    """A zero command while closing fast is coasting; v2 must take over."""
    out = apply_safety_filter(np.zeros(3), np.array([3.0, 0.0, 0.0]), np.array([-0.6, 0.0, 0.0]), 1.0, max_accel=A1)
    assert out["override"] == BRAKING_SUBSTITUTED
    assert out["applied"][0] > 0.0


def test_lateral_thrust_with_token_braking_is_not_accepted():
    """u.v < 0 but a large lateral component: predicted unsafe, policy loses authority."""
    pos, vel = np.array([2.0, 0.0, 0.0]), np.array([-0.3, 0.0, 0.0])
    out = apply_safety_filter(np.array([0.01, 1.0, 1.0]), pos, vel, 1.0, max_accel=A1)
    assert out["override"] == BRAKING_SUBSTITUTED
    _, next_vel = cw_step(pos, vel, out["applied"], dt=1.0)
    assert np.linalg.norm(next_vel) <= np.linalg.norm(vel) + 1e-9


def test_retreating_vehicle_is_never_pushed_back_toward_the_target():
    """Opening range fast inside the bubble is not a collision risk: no override."""
    for cmd in (np.zeros(3), np.array([1.0, 0.0, 0.0])):
        out = apply_safety_filter(cmd, np.array([3.0, 0.0, 0.0]), np.array([0.6, 0.0, 0.0]), 1.0, max_accel=A1)
        assert out["override"] is None
        np.testing.assert_allclose(out["applied"], cmd)


def test_fuel_floor_holds_when_safe():
    """Below the fuel floor, a vehicle that is not on a collision course holds."""
    out = apply_safety_filter(
        commanded_action=np.array([-1.0, -1.0, -1.0]),
        rel_pos=np.array([3.0, 0.0, 0.0]),
        rel_vel=np.array([0.6, 0.0, 0.0]),  # opening range
        fuel_remaining=0.01, max_accel=A1)
    assert out["override"] == "emergency_hold"
    np.testing.assert_allclose(out["applied"], [0.0, 0.0, 0.0])


def test_fuel_floor_never_disables_collision_avoidance():
    """Below the fuel floor on a collision course, the reserve is spent on braking."""
    out = apply_safety_filter(
        commanded_action=np.zeros(3),
        rel_pos=np.array([3.0, 0.0, 0.0]),
        rel_vel=np.array([-0.6, 0.0, 0.0]),  # closing fast
        fuel_remaining=0.01, max_accel=A1)
    assert out["override"] == EMERGENCY_BRAKE
    assert out["applied"][0] > 0.0


def test_goal_label_does_not_disable_v2():
    """By default no goal label switches the envelope off."""
    out = apply_safety_filter(np.array([-1.0, 0.0, 0.0]), np.array([3.0, 0.0, 0.0]),
                              np.array([-0.6, 0.0, 0.0]), 1.0, goal_type="DOCK", max_accel=A1)
    assert out["override"] == BRAKING_SUBSTITUTED


def test_braking_action_never_increases_speed():
    """Certified braking is anti-parallel, sized to the speed, and never overshoots."""
    for speed in (0.05, 0.25, 0.5, 3.0):
        for direction in (
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, -1.0, 0.3]),
            np.array([0.2, 0.1, -1.0]),
        ):
            vel = direction / np.linalg.norm(direction) * speed
            brake = certified_brake_action(vel, max_accel=A1)
            assert float(np.dot(brake, vel)) < 0.0
            assert np.linalg.norm(brake) <= 1.0 + 1e-12
            np.testing.assert_allclose(np.linalg.norm(brake), min(1.0, speed), atol=1e-9)
            _, next_vel = cw_step(np.array([3.0, 0.0, 0.0]), vel, brake, dt=1.0)
            # Exactly cancelled up to the O(n * v * dt) CW Coriolis coupling.
            coupling = 3.0 * PHYSICS_CONFIG.n * speed * PHYSICS_CONFIG.dt + 1e-4
            assert np.linalg.norm(next_vel) <= speed
            assert np.linalg.norm(next_vel) <= max(0.0, speed - 1.0) + coupling


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
    log_v2, ok_v2 = _rollout("v2", steps=120)

    # No runtime assurance, and the MVP's thrust-denial version: neither docks.
    assert not ok_open and not ok_v1

    # v1 does reach the capture radius -- at a speed that would be a collision.
    closest_v1 = min(log_v1, key=lambda row: row[0])
    assert closest_v1[0] < ENV_CONFIG.docking_success_dist
    assert closest_v1[1] > 1.0, "the coasting pass should still be metres per second"

    # v2 recovers the identical approach into a soft capture, and never lets
    # the vehicle into the capture sphere at collision speed on the way.
    assert ok_v2, "the certified backup should recover the reckless slam"
    assert all(not (d < ENV_CONFIG.docking_success_dist and v >= 0.1) for d, v, _, _ in log_v2)
    end_dist, end_speed, _, _ = log_v2[-1]
    assert end_dist < ENV_CONFIG.docking_success_dist
    assert end_speed < 0.1
