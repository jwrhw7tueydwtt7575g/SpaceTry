"""Pure goal validation; no actuator filtering occurs in this module."""
import math

import numpy as np

from .schema import ConstraintConfig, GoalType, ReasonCode, Telemetry, ValidationResult


def _finite_vector(values, size):
    try:
        arr = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return arr.shape == (size,) and bool(np.all(np.isfinite(arr)))


def _finite_number(value):
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def validate_goal(goal, constraints: ConstraintConfig, telemetry: Telemetry, now_ns: int) -> ValidationResult:
    """Validate a goal against deterministic mission constraints.

    Every numeric field is checked for finiteness *before* any comparison:
    every ordered comparison against NaN is False, so a NaN distance, closing
    speed or timeout would otherwise slip past each range check and be
    accepted.
    """
    if not isinstance(goal.goal_type, GoalType): return ValidationResult(False, ReasonCode.INVALID_GOAL_TYPE, "goal_type is not a supported GoalType")
    if not _finite_vector(goal.target_position_m, 3): return ValidationResult(False, ReasonCode.INVALID_VECTOR, "target_position_m is not a finite 3-vector")
    if not all(_finite_number(v) for v in (goal.desired_distance_m, goal.desired_closing_speed_mps, goal.timeout_s)):
        return ValidationResult(False, ReasonCode.INVALID_VECTOR, "goal distance, closing speed and timeout must be finite numbers")
    if not _finite_vector(goal.max_force_n, 3) or not _finite_vector(goal.max_torque_nm, 3):
        return ValidationResult(False, ReasonCode.INVALID_VECTOR, "goal actuator limits must be finite 3-vectors")
    if not _finite_vector(telemetry.position_m, 3) or not _finite_vector(telemetry.velocity_mps, 3): return ValidationResult(False, ReasonCode.INVALID_VECTOR, "telemetry vector is not finite or three-dimensional")
    if not _finite_number(telemetry.fuel_fraction) or not 0 <= telemetry.fuel_fraction <= 1 or telemetry.stamp_ns < 0: return ValidationResult(False, ReasonCode.INVALID_VECTOR, "telemetry fuel or timestamp is invalid")
    age_s = (now_ns - telemetry.stamp_ns) / 1e9
    if age_s > constraints.stale_telemetry_timeout_s: return ValidationResult(False, ReasonCode.STALE_TELEMETRY, "telemetry is older than the configured stale bound")
    if age_s < -constraints.stale_telemetry_timeout_s: return ValidationResult(False, ReasonCode.STALE_TELEMETRY, "telemetry timestamp is in the future beyond the allowed clock skew")
    if goal.timeout_s <= 0 or goal.timeout_s > constraints.max_goal_timeout_s: return ValidationResult(False, ReasonCode.INVALID_TIMEOUT, "goal timeout exceeds configured mission bound")
    if goal.desired_distance_m < 0 or goal.desired_closing_speed_mps < 0: return ValidationResult(False, ReasonCode.INVALID_VECTOR, "goal distance and closing speed must be non-negative")
    if telemetry.fuel_fraction < constraints.fuel_floor: return ValidationResult(False, ReasonCode.FUEL_BELOW_FLOOR, "fuel is below the configured safety floor")
    if goal.goal_type in (GoalType.APPROACH, GoalType.INSPECT) and goal.desired_distance_m < constraints.keep_out_radius_m: return ValidationResult(False, ReasonCode.KEEP_OUT_VIOLATION, "goal requests a distance inside the keep-out radius")
    if goal.desired_closing_speed_mps > constraints.max_closing_speed_mps: return ValidationResult(False, ReasonCode.CLOSING_SPEED_EXCEEDED, "goal closing speed exceeds configured limit")
    if any(abs(v) > limit for v, limit in zip(goal.max_force_n, constraints.max_force_n)) or any(abs(v) > limit for v, limit in zip(goal.max_torque_nm, constraints.max_torque_nm)): return ValidationResult(False, ReasonCode.ACTUATOR_LIMIT_EXCEEDED, "goal actuator limit exceeds configured authority")
    return ValidationResult(True, ReasonCode.ACCEPTED, "goal satisfies deterministic mission constraints")
