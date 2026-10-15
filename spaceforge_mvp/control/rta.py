"""Runtime-assurance filter v2 — what the interlock should do when it says no.

The MVP interlock (``spaceforge_mvp.safety.mission_safety.evaluate_goal``) can
only *deny* a command: when it denies, the environment zeroes the thrust. A
vehicle that is already above the safe closing speed inside the keep-out bubble
therefore coasts straight through the target, because denying thrust is not the
same as applying brakes. Measured in ``hot_v1``: penetration at ~2.0 m/s.

v2 keeps the same veto rules and the same "fuel floor wins" priority, but when
the veto fires it substitutes a *certified braking action* — maximum
deceleration along the current relative velocity — instead of doing nothing.

The resulting authority split is the one a real runtime-assurance layer has:

    the policy may fly anything that is not unsafe;
    when it flies something unsafe, the certified controller flies the
    braking law instead of the policy flying nothing at all.
"""

from typing import Any, Dict

import numpy as np

from spaceforge_mvp.config import SAFETY_CONFIG, SafetyConfig
from spaceforge_mvp.safety.mission_safety import evaluate_goal

# Status string reported when the filter substitutes the braking action.
BRAKING_SUBSTITUTED = "braking_substituted"


def certified_brake_action(
    rel_vel: np.ndarray,
    max_accel: float = 1.0,
) -> np.ndarray:
    """Maximum deceleration along the current relative velocity vector.

    This is the simplest action with a provable property: it can never increase
    the relative speed, so a vehicle that is told to fly it cannot accelerate
    into the target.
    """
    vel = np.asarray(rel_vel, dtype=np.float64)
    speed = float(np.linalg.norm(vel))
    if speed <= 1e-9:
        return np.zeros(3, dtype=np.float64)
    return ((-vel / speed) * float(max_accel)).astype(np.float64)


def apply_safety_filter(
    commanded_action: np.ndarray,
    rel_pos: np.ndarray,
    rel_vel: np.ndarray,
    fuel_remaining: float,
    goal_type: str = "APPROACH",
    safety_config: SafetyConfig = SAFETY_CONFIG,
) -> Dict[str, Any]:
    """Return the command that may actually reach the thrusters.

    Returns
    -------
    dict with keys:
        ``applied``  — the safe action to execute
        ``override`` — None when the policy kept authority, otherwise the
                       interlock status ("emergency_hold" | "braking_substituted")
        ``reason``   — human-readable explanation
    """
    cmd = np.clip(np.asarray(commanded_action, dtype=np.float64).reshape(3), -1.0, 1.0)
    pos = np.asarray(rel_pos, dtype=np.float64)
    vel = np.asarray(rel_vel, dtype=np.float64)

    verdict = evaluate_goal(
        goal_type=goal_type,
        current_dist=float(np.linalg.norm(pos)),
        fuel_remaining=float(fuel_remaining),
        current_speed=float(np.linalg.norm(vel)),
        commanded_action=cmd,
        rel_vel=vel,
        safety_config=safety_config,
    )
    status = str(verdict["status"])

    if status == "accepted":
        return {"applied": cmd, "override": None, "reason": str(verdict["reason"])}

    if status == "emergency_hold":
        # No fuel budget: the only safe action is to stop spending it.
        return {
            "applied": np.zeros(3, dtype=np.float64),
            "override": "emergency_hold",
            "reason": str(verdict["reason"]),
        }

    # Vetoed but still fuelled: take over and brake rather than coast.
    return {
        "applied": certified_brake_action(vel),
        "override": BRAKING_SUBSTITUTED,
        "reason": (
            "Commanded thrust rejected inside the keep-out bubble: policy authority "
            "revoked for this step, certified braking action substituted."
        ),
    }
