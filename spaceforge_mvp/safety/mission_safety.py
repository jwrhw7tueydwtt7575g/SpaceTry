"""Mission safety interlock layer for spacecraft rendezvous.

Two pieces live here:

* ``evaluate_goal`` -- the v1 interlock. It can only accept or reject a
  command (the environment zeroes rejected thrust). Its proximity gate is
  purely geometric by default, and a command only counts as "braking" when it
  actually lowers the relative speed over the next control step.
* Envelope helpers (``closing_speed``, ``closing_speed_limit``,
  ``predict_unsafe``) shared with the v2 runtime-assurance filter in
  ``spaceforge_mvp.control.rta``.
"""

from typing import Dict, Optional

import numpy as np

from spaceforge_mvp.config import SAFETY_CONFIG, SafetyConfig

SAFE_APPROACH_DIST: float = SAFETY_CONFIG.safe_approach_dist
FUEL_FLOOR: float = SAFETY_CONFIG.fuel_floor
MAX_SAFE_CLOSING_SPEED: float = SAFETY_CONFIG.max_safe_closing_speed

_EPS = 1e-9


# ----------------------------------------------------------------- envelope helpers
def proximity_gate_applies(goal_type: str, safety_config: SafetyConfig = SAFETY_CONFIG) -> bool:
    """True when the proximity speed rule is in force for this goal label."""
    return bool(safety_config.enforce_proximity_for_all_goals or goal_type in safety_config.proximity_goals)


def closing_speed(rel_pos: np.ndarray, rel_vel: np.ndarray) -> float:
    """Rate at which range to the target is shrinking (m/s); negative when opening.

    At (numerically) zero range the direction to the target is undefined, so the
    full speed is treated as closing -- the conservative choice.
    """
    pos = np.asarray(rel_pos, dtype=np.float64)
    vel = np.asarray(rel_vel, dtype=np.float64)
    dist = float(np.linalg.norm(pos))
    if dist < _EPS:
        return float(np.linalg.norm(vel))
    return float(-np.dot(vel, pos) / dist)


def closing_speed_limit(dist: float, safety_config: SafetyConfig = SAFETY_CONFIG) -> float:
    """Maximum permitted closing speed at range ``dist``.

    Outside the keep-out bubble the limit is ``max_safe_closing_speed`` (it is
    only enforced on states predicted to enter the bubble). Inside, it tapers
    linearly with range down to ``capture_speed_fraction * max_docking_speed``,
    so a vehicle that respects it is slow enough to be captured on arrival.
    """
    edge = float(safety_config.max_safe_closing_speed)
    floor = float(safety_config.capture_speed_fraction * safety_config.max_docking_speed)
    if dist >= safety_config.safe_approach_dist:
        return edge
    scaled = edge * float(dist) / float(safety_config.safe_approach_dist)
    return float(min(edge, max(floor, scaled)))


def predict_unsafe(
    rel_pos: np.ndarray,
    rel_vel: np.ndarray,
    accel_mps2: np.ndarray,
    dt: float,
    safety_config: SafetyConfig = SAFETY_CONFIG,
    samples: int = 8,
) -> bool:
    """Would holding ``accel_mps2`` for one step violate the approach envelope?

    The step is sampled at ``samples`` points (constant-acceleration kinematics;
    the CW coupling over one control step is orders of magnitude smaller than
    the actuator authority). A sample is unsafe when it is inside the keep-out
    bubble and closing faster than the envelope allows, or inside the capture
    sphere at or above the docking speed. Sampling the whole step -- not just
    its end point -- catches a fly-through that starts and ends outside.
    """
    pos = np.asarray(rel_pos, dtype=np.float64)
    vel = np.asarray(rel_vel, dtype=np.float64)
    acc = np.asarray(accel_mps2, dtype=np.float64)
    taus = np.linspace(float(dt) / samples, float(dt), samples)[:, None]
    p = pos + vel * taus + 0.5 * acc * taus ** 2
    v = vel + acc * taus
    dists = np.linalg.norm(p, axis=1)
    speeds = np.linalg.norm(v, axis=1)
    for d, pk, vk, s in zip(dists, p, v, speeds):
        if d < safety_config.capture_radius and s >= safety_config.max_docking_speed:
            return True
        if d < safety_config.safe_approach_dist and closing_speed(pk, vk) > closing_speed_limit(d, safety_config) + 1e-12:
            return True
    return False


def state_unsafe(rel_pos: np.ndarray, rel_vel: np.ndarray, safety_config: SafetyConfig = SAFETY_CONFIG) -> bool:
    """Is the vehicle, right now, inside the bubble and closing above the envelope?"""
    dist = float(np.linalg.norm(rel_pos))
    return bool(
        dist < safety_config.safe_approach_dist
        and closing_speed(rel_pos, rel_vel) > closing_speed_limit(dist, safety_config) + 1e-12
    )


# ------------------------------------------------------------------- v1 interlock
def is_braking_command(
    commanded_action: np.ndarray,
    rel_vel: np.ndarray,
    dt: float = 1.0,
    max_accel: float = 1.0,
) -> bool:
    """A command brakes only if it opposes the velocity AND lowers the speed.

    ``u . v <= 0`` alone is not braking: a zero command coasts, and a command
    with a token opposing component plus large lateral thrust speeds the
    vehicle up. Both are rejected here.
    """
    cmd = np.asarray(commanded_action, dtype=np.float64)
    vel = np.asarray(rel_vel, dtype=np.float64)
    speed = float(np.linalg.norm(vel))
    if speed <= 1e-6 or float(np.dot(cmd, vel)) >= 0.0:
        return False
    next_speed = float(np.linalg.norm(vel + cmd * float(max_accel) * float(dt)))
    return next_speed <= speed + 1e-12


def evaluate_goal(
    goal_type: str,
    current_dist: float,
    fuel_remaining: float,
    current_speed: float,
    commanded_action: Optional[np.ndarray] = None,
    rel_vel: Optional[np.ndarray] = None,
    safety_config: SafetyConfig = SAFETY_CONFIG,
    dt: float = 1.0,
    max_accel: float = 1.0,
) -> Dict[str, str]:
    """
    Evaluate a candidate command against the v1 interlock rules.

    Priority order:
    1. fuel_remaining < safety_config.fuel_floor -> emergency_hold.
    2. proximity speed gate (dist < safe_approach_dist and speed > max_safe_closing_speed):
       - Applies to every goal by default (``enforce_proximity_for_all_goals``);
         the legacy label-scoped mode is opt-in.
       - A command passes only if it is genuinely braking (see
         ``is_braking_command``): it opposes the velocity and lowers the speed
         over one step of ``dt`` seconds at ``max_accel`` m/s^2 per unit command.
         Coasting (zero thrust) and lateral thrust are rejected.
    3. else -> accepted.

    v1 can only veto. A vetoed command is zeroed by the environment, so a fast
    vehicle still coasts in -- that limitation is what the v2 filter
    (``spaceforge_mvp.control.rta``) exists to fix.

    Returns
    -------
    dict
        ``status`` ("accepted" | "rejected" | "emergency_hold") and ``reason``.
    """
    if fuel_remaining < safety_config.fuel_floor:
        return {
            "status": "emergency_hold",
            "reason": (
                f"Fuel remaining ({fuel_remaining:.4f}) is below minimum safety floor ({safety_config.fuel_floor})."
            ),
        }

    if (
        proximity_gate_applies(goal_type, safety_config)
        and current_dist < safety_config.safe_approach_dist
        and current_speed > safety_config.max_safe_closing_speed
    ):
        if (
            commanded_action is not None
            and rel_vel is not None
            and is_braking_command(commanded_action, rel_vel, dt=dt, max_accel=max_accel)
        ):
            return {
                "status": "accepted",
                "reason": "Braking/decelerating thrust permitted inside safe approach bubble.",
            }

        return {
            "status": "rejected",
            "reason": (
                f"Goal '{goal_type}' rejected: distance ({current_dist:.2f}m) < {safety_config.safe_approach_dist:.2f}m "
                f"with excessive speed ({current_speed:.2f}m/s > {safety_config.max_safe_closing_speed:.2f}m/s)."
            ),
        }

    return {
        "status": "accepted",
        "reason": "Goal satisfies all safety interlock constraints.",
    }
