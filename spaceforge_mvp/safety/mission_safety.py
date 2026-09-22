"""Mission safety interlock layer for spacecraft rendezvous."""

from typing import Dict

SAFE_APPROACH_DIST: float = 5.0
FUEL_FLOOR: float = 0.05
MAX_SAFE_CLOSING_SPEED: float = 0.2


def evaluate_goal(
    goal_type: str,
    current_dist: float,
    fuel_remaining: float,
    current_speed: float,
) -> Dict[str, str]:
    """
    Evaluate candidate mission goal against mission safety constraints.

    Priority order:
    1. fuel_remaining < FUEL_FLOOR -> emergency_hold, regardless of anything else.
    2. goal_type == "APPROACH", current_dist < SAFE_APPROACH_DIST, and
       current_speed > MAX_SAFE_CLOSING_SPEED -> rejected.
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
