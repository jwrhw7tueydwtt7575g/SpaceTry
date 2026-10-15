"""Deterministic action filtering, separate from goal validation.

Braking reuses the proven Python RTA law
``spaceforge_mvp.control.rta.certified_brake_action`` (maximum deceleration
anti-parallel to relative velocity, provably never increasing speed), scaled
into mission_ctrl's per-axis actuator authority. Goal validation stays in
``validation.py`` and is never bypassed: every path through this module either
passes validation or emits a safe-hold/braking substitute with a stable reason.
"""
import numpy as np
from spaceforge_mvp.control.rta import certified_brake_action
from .schema import ActuatorCommand, ConstraintConfig, FilterResult, GoalSpec, GoalType, ReasonCode, Telemetry
from .validation import validate_goal

def _brake(velocity, limits):
    """Certified braking direction from the proven RTA law, clipped to authority.

    certified_brake_action returns the unit-maximum anti-parallel vector; the
    per-axis clip maps it into this boundary's force envelope without changing
    its direction (clip of a colinear vector stays colinear).
    """
    direction = certified_brake_action(np.asarray(velocity, dtype=np.float64))
    limits_arr = np.asarray(limits, dtype=np.float64)
    return np.clip(direction * limits_arr, -limits_arr, limits_arr)

def _zero():
    return ActuatorCommand((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))

def _brake_command(telemetry, constraints):
    force = _brake(telemetry.velocity_mps, constraints.max_force_n)
    return ActuatorCommand(tuple(float(v) for v in force), (0.0, 0.0, 0.0))

def filter_action(goal: GoalSpec, telemetry: Telemetry, commanded: ActuatorCommand, constraints: ConstraintConfig, now_ns: int, event_id: str = "OVR-000000") -> FilterResult:
    validation = validate_goal(goal, constraints, telemetry, now_ns)
    force = np.asarray(commanded.force_n, dtype=np.float64); torque = np.asarray(commanded.torque_nm, dtype=np.float64)
    if validation.code in (ReasonCode.FUEL_BELOW_FLOOR, ReasonCode.INVALID_GOAL_TYPE, ReasonCode.INVALID_VECTOR, ReasonCode.INVALID_TIMEOUT, ReasonCode.ACTUATOR_LIMIT_EXCEEDED):
        return FilterResult(_zero(), True, validation.code, validation.reason + "; safe hold enforced", event_id)
    if validation.code in (ReasonCode.STALE_TELEMETRY, ReasonCode.KEEP_OUT_VIOLATION, ReasonCode.CLOSING_SPEED_EXCEEDED):
        return FilterResult(_brake_command(telemetry, constraints), True, validation.code, validation.reason + "; certified braking enforced", event_id)
    if goal.goal_type is GoalType.ABORT:
        return FilterResult(_brake_command(telemetry, constraints), True, ReasonCode.ABORT_ENFORCED, "ABORT goal enforces certified braking", event_id)
    if goal.goal_type is GoalType.SAFE_HOLD:
        return FilterResult(_zero(), True, ReasonCode.SAFE_HOLD_ENFORCED, "SAFE_HOLD goal enforces zero actuator command", event_id)
    if not np.all(np.isfinite(force)) or not np.all(np.isfinite(torque)):
        return FilterResult(_zero(), True, ReasonCode.INVALID_VECTOR, "non-finite command rejected; safe hold enforced", event_id)
    clipped_force = np.clip(force, -np.asarray(constraints.max_force_n), np.asarray(constraints.max_force_n))
    clipped_torque = np.clip(torque, -np.asarray(constraints.max_torque_nm), np.asarray(constraints.max_torque_nm))
    if not np.allclose(force, clipped_force) or not np.allclose(torque, clipped_torque):
        return FilterResult(ActuatorCommand(tuple(clipped_force), tuple(clipped_torque)), True, ReasonCode.COMMAND_CLIPPED, "command exceeded actuator authority and was clipped", event_id)
    distance = float(np.linalg.norm(telemetry.position_m)); speed = float(np.linalg.norm(telemetry.velocity_mps))
    closing = float(np.dot(force, telemetry.velocity_mps)) > 0.0
    if distance < constraints.keep_out_radius_m and speed > constraints.max_closing_speed_mps and closing:
        return FilterResult(_brake_command(telemetry, constraints), True, ReasonCode.CLOSING_SPEED_EXCEEDED, "accelerating command vetoed inside keep-out; certified braking enforced", event_id)
    return FilterResult(commanded, False, ReasonCode.ACCEPTED, "command accepted", "")
