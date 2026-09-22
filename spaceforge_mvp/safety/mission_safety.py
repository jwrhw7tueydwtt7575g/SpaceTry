"""Mission safety interlock layer for spacecraft rendezvous."""

from typing import Dict, Optional
import numpy as np
from spaceforge_mvp.config import SAFETY_CONFIG

SAFE_APPROACH_DIST: float = SAFETY_CONFIG.safe_approach_dist
FUEL_FLOOR: float = SAFETY_CONFIG.fuel_floor
MAX_SAFE_CLOSING_SPEED: float = SAFETY_CONFIG.max_safe_closing_speed


def evaluate_goal(
    goal_type: str,
    current_dist: float,
    fuel_remaining: float,
    current_speed: float,
    commanded_action: Optional[np.ndarray] = None,
    rel_vel: Optional[np.ndarray] = None,
) -> Dict[str, str]:
    """
    Evaluate candidate mission goal/action against mission safety constraints.

    Priority order:
    1. fuel_remaining < FUEL_FLOOR -> emergency_hold, regardless of anything else.
    2. goal_type == "APPROACH", current_dist < SAFE_APPROACH_DIST, and
       current_speed > MAX_SAFE_CLOSING_SPEED:
       - If commanded_action and rel_vel are provided, allow braking/decelerating
         thrust (where u dot v < 0 or speed decreases), rejecting only accelerating thrust.
       - Otherwise, reject non-braking approach commands.
    3. else -> accepted.

    Parameters
    ----------
    goal_type : str
        The type of mission goal (e.g. "APPROACH", "INSPECT", "DOCK").
    current_dist : float
        Current distance to target in meters.
    fuel_remaining : float
        Remaining fuel fraction in [0.0, 1.0].
    current_speed : float
        Current relative speed in m/s.
    commanded_action : np.ndarray, optional
        Commanded 3D acceleration vector [ax, ay, az].
    rel_vel : np.ndarray, optional
        Current relative velocity vector [vx, vy, vz].

    Returns
    -------
    dict
        Dictionary with keys 'status' ("accepted" | "rejected" | "emergency_hold")
        and 'reason' (explanation string).
    """
    if fuel_remaining < FUEL_FLOOR:
        return {
            "status": "emergency_hold",
            "reason": (
                f"Fuel remaining ({fuel_remaining:.4f}) is below minimum safety floor ({FUEL_FLOOR})."
            ),
        }

    if (
        goal_type == "APPROACH"
        and current_dist < SAFE_APPROACH_DIST
        and current_speed > MAX_SAFE_CLOSING_SPEED
    ):
        # Check if the commanded action is a braking / decelerating action
        if commanded_action is not None and rel_vel is not None and current_speed > 1e-6:
            cmd = np.asarray(commanded_action, dtype=np.float64)
            vel = np.asarray(rel_vel, dtype=np.float64)
            dot_prod = float(np.dot(cmd, vel))

            # If action opposes velocity vector (u dot v <= 0), it is braking/decelerating -> ALLOW IT
            if dot_prod <= 0.0:
                return {
                    "status": "accepted",
                    "reason": "Braking/decelerating thrust permitted inside safe approach bubble.",
                }

        return {
            "status": "rejected",
            "reason": (
                f"Goal '{goal_type}' rejected: distance ({current_dist:.2f}m) < {SAFE_APPROACH_DIST:.2f}m "
                f"with excessive speed ({current_speed:.2f}m/s > {MAX_SAFE_CLOSING_SPEED:.2f}m/s)."
            ),
        }

    return {
        "status": "accepted",
        "reason": "Goal satisfies all safety interlock constraints.",
    }
