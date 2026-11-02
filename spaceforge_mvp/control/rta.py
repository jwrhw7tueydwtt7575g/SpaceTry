"""Runtime-assurance filter v2 -- what the interlock should do when it says no.

The v1 interlock (``spaceforge_mvp.safety.mission_safety.evaluate_goal``) can
only *deny* a command, and a denied command is zeroed. A vehicle that is
already fast inside the keep-out bubble therefore coasts straight through the
target: denying thrust is not braking.

v2 is a predictive, minimal-intervention filter:

1. It predicts one control step ahead under the commanded thrust and checks the
   whole step against a distance-dependent closing-speed envelope
   (``closing_speed_limit``) plus a capture-sphere fly-through test. Outside
   the keep-out sphere the envelope is a braking curve, so the vehicle always
   reaches the sphere edge at or below the gate speed instead of arriving hot
   and braking inside. The policy keeps authority whenever the predicted step
   is safe.
2. When the predicted step is unsafe, the policy's command is discarded for
   that step and the backup controller flies instead (Simplex
   architecture): the glideslope law, whose approach profile sits inside the
   envelope. If even the backup's command is predicted unsafe (the vehicle is
   already far above the envelope), an envelope-tracking brake is used: the
   closing component of velocity is driven down to the envelope, tangential
   motion is not amplified, and the correction is sized to what one step can
   deliver -- it never over-brakes into a reversal.
3. The fuel floor still stops discretionary thrust, but it never stops
   collision avoidance: below the floor, a vehicle on an unsafe trajectory is
   still braked from the reserve.

The goal label never disables the envelope unless the legacy label-scoped
mode is explicitly selected in ``SafetyConfig``.
"""

from typing import Any, Dict

import numpy as np

from spaceforge_mvp.config import ACTUATOR_CONFIG, PHYSICS_CONFIG, SAFETY_CONFIG, SafetyConfig
from spaceforge_mvp.control.glideslope import glideslope_action
from spaceforge_mvp.safety.mission_safety import (
    closing_speed,
    closing_speed_limit,
    predict_unsafe,
    proximity_gate_applies,
)

# Status strings reported when the filter takes authority.
BRAKING_SUBSTITUTED = "braking_substituted"
EMERGENCY_HOLD = "emergency_hold"
EMERGENCY_BRAKE = "emergency_brake"

_EPS = 1e-9


def certified_brake_action(
    rel_vel: np.ndarray,
    max_accel: float = ACTUATOR_CONFIG.max_cmd_accel,
    dt: float = 1.0,
) -> np.ndarray:
    """Speed-sized deceleration anti-parallel to the relative velocity.

    Returns a normalized command (1.0 == ``max_accel`` m/s^2). Its magnitude is
    ``min(1, speed / (max_accel * dt))``: enough to cancel the velocity within
    one step if authority allows, never more. A full-authority brake applied to
    a slow vehicle would reverse it and *raise* its speed (0.25 m/s becomes
    0.75 m/s at 1 m/s^2 over 1 s); sizing removes that overshoot, so the speed
    after the step is ``max(0, speed - max_accel * dt)`` up to the O(n * dt)
    Clohessy-Wiltshire coupling.
    """
    vel = np.asarray(rel_vel, dtype=np.float64)
    speed = float(np.linalg.norm(vel))
    if speed <= _EPS:
        return np.zeros(3, dtype=np.float64)
    magnitude = min(1.0, speed / (float(max_accel) * float(dt)))
    return ((-vel / speed) * magnitude).astype(np.float64)


def envelope_correction(
    rel_pos: np.ndarray,
    rel_vel: np.ndarray,
    safety_config: SafetyConfig = SAFETY_CONFIG,
    max_accel: float = ACTUATOR_CONFIG.max_cmd_accel,
    dt: float = 1.0,
    margin: float = 0.9,
) -> np.ndarray:
    """Normalized command that brings the closing speed back inside the envelope.

    Only the closing (radial) component of velocity is changed, and only
    downward: a vehicle that is opening range is never pushed back toward the
    target, and tangential motion is not amplified.
    """
    pos = np.asarray(rel_pos, dtype=np.float64)
    vel = np.asarray(rel_vel, dtype=np.float64)
    dist = float(np.linalg.norm(pos))
    if dist < _EPS:
        # Direction to the target is undefined at contact: stop outright.
        return certified_brake_action(vel, max_accel=max_accel, dt=dt)
    away = pos / dist  # unit vector pointing from the target to the chaser
    closing = closing_speed(pos, vel)
    limit_now = closing_speed_limit(dist, safety_config, max_accel)
    look_ahead = max(0.0, dist - limit_now * float(dt))
    target_closing = margin * min(limit_now, closing_speed_limit(look_ahead, safety_config, max_accel))
    if closing <= target_closing:
        return np.zeros(3, dtype=np.float64)
    # Raise the outward radial velocity by (closing - target_closing).
    delta_v = away * (closing - target_closing)
    command = delta_v / (float(max_accel) * float(dt))
    norm = float(np.linalg.norm(command))
    if norm > 1.0:
        command = command / norm
    return command.astype(np.float64)


def apply_safety_filter(
    commanded_action: np.ndarray,
    rel_pos: np.ndarray,
    rel_vel: np.ndarray,
    fuel_remaining: float,
    goal_type: str = "APPROACH",
    safety_config: SafetyConfig = SAFETY_CONFIG,
    dt: float = 1.0,
    max_accel: float = ACTUATOR_CONFIG.max_cmd_accel,
    n: float = PHYSICS_CONFIG.n,
) -> Dict[str, Any]:
    """Return the command that may actually reach the thrusters.

    Parameters
    ----------
    commanded_action : normalized command in [-1, 1]^3 (1.0 == ``max_accel``).
    dt : control step in seconds; ``max_accel`` : authority in m/s^2.
    n : mean motion (rad/s) for the backup controller's CW feed-forward.

    Returns
    -------
    dict with keys:
        ``applied``  -- the safe normalized action to execute
        ``override`` -- None when the policy kept authority, otherwise one of
                       "emergency_hold" | "emergency_brake" | "braking_substituted"
        ``reason``   -- human-readable explanation
    """
    cmd = np.clip(np.asarray(commanded_action, dtype=np.float64).reshape(3), -1.0, 1.0)
    pos = np.asarray(rel_pos, dtype=np.float64)
    vel = np.asarray(rel_vel, dtype=np.float64)
    gated = proximity_gate_applies(goal_type, safety_config)

    if fuel_remaining < safety_config.fuel_floor:
        # Discretionary thrust stops; collision avoidance does not.
        if gated and predict_unsafe(pos, vel, np.zeros(3), dt, safety_config, max_accel=max_accel):
            return {
                "applied": envelope_correction(pos, vel, safety_config, max_accel, dt),
                "override": EMERGENCY_BRAKE,
                "reason": (
                    f"Fuel ({fuel_remaining:.4f}) below floor ({safety_config.fuel_floor}) on a collision "
                    "trajectory: reserve spent on braking."
                ),
            }
        return {
            "applied": np.zeros(3, dtype=np.float64),
            "override": EMERGENCY_HOLD,
            "reason": f"Fuel ({fuel_remaining:.4f}) below floor ({safety_config.fuel_floor}): emergency hold.",
        }

    if not gated or not predict_unsafe(pos, vel, cmd * float(max_accel), dt, safety_config, max_accel=max_accel):
        return {"applied": cmd, "override": None, "reason": "Predicted step satisfies the approach envelope."}

    backup = glideslope_action(pos, vel, n=n, dt=dt, max_accel=max_accel)
    if not predict_unsafe(pos, vel, backup * float(max_accel), dt, safety_config, max_accel=max_accel):
        return {
            "applied": backup,
            "override": BRAKING_SUBSTITUTED,
            "reason": (
                "Commanded thrust predicted to violate the approach envelope: policy authority "
                "revoked for this step, glideslope backup substituted."
            ),
        }
    return {
        "applied": envelope_correction(pos, vel, safety_config, max_accel, dt),
        "override": BRAKING_SUBSTITUTED,
        "reason": (
            "Commanded thrust predicted to violate the approach envelope: policy authority "
            "revoked for this step, envelope-tracking braking substituted."
        ),
    }
