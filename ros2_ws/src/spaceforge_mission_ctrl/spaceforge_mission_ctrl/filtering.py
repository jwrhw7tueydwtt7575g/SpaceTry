"""Deterministic action filtering, separate from goal validation.

Rules, in one sentence each:

* A vehicle inside the keep-out radius and *closing* (range shrinking) faster
  than the limit is braked -- whatever the goal, whatever the command, and
  whether or not the goal was valid. Rejecting a goal never means coasting
  into the target.
* "Closing" is measured along the line of sight (position-aware), so a vehicle
  opening range is never braked back toward the target.
* Braking reuses the Python RTA law
  ``spaceforge_mvp.control.rta.certified_brake_action``: anti-parallel to the
  velocity and sized so it cancels at most the current velocity within one
  control period -- it cannot over-brake into a reversal.
* Commands are clipped to authority first and the clipped command is then
  checked like any other; clipping is never a way around the keep-out rule.
* A command that would push a currently-safe vehicle above the limit inside
  the keep-out radius within the next period is replaced by a hold.
"""
import numpy as np

from spaceforge_mvp.control.rta import certified_brake_action
from spaceforge_mvp.safety.mission_safety import closing_speed

from .schema import ActuatorCommand, ConstraintConfig, FilterResult, GoalSpec, GoalType, ReasonCode, Telemetry
from .validation import validate_goal

_HOLD_CODES = (
    ReasonCode.FUEL_BELOW_FLOOR, ReasonCode.INVALID_GOAL_TYPE, ReasonCode.INVALID_VECTOR,
    ReasonCode.INVALID_TIMEOUT, ReasonCode.ACTUATOR_LIMIT_EXCEEDED, ReasonCode.STALE_TELEMETRY,
    ReasonCode.KEEP_OUT_VIOLATION, ReasonCode.CLOSING_SPEED_EXCEEDED,
)


def _brake(velocity, limits, mass_kg: float = 1.0, period_s: float = 1.0):
    """Speed-sized certified braking force, clipped to authority.

    A single scalar authority (the smallest axis limit) keeps the force exactly
    anti-parallel to the velocity; its magnitude never exceeds what cancels the
    velocity within one control period.
    """
    limits_arr = np.asarray(limits, dtype=np.float64)
    authority = float(np.min(limits_arr))
    direction = certified_brake_action(
        np.asarray(velocity, dtype=np.float64), max_accel=authority / float(mass_kg), dt=float(period_s),
    )
    return np.clip(direction * authority, -limits_arr, limits_arr)


def _zero():
    return ActuatorCommand((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))


def _brake_command(telemetry, constraints):
    force = _brake(telemetry.velocity_mps, constraints.max_force_n,
                   constraints.vehicle_mass_kg, constraints.control_period_s)
    return ActuatorCommand(tuple(float(v) for v in force), (0.0, 0.0, 0.0))


def _telemetry_usable(telemetry) -> bool:
    try:
        pos = np.asarray(telemetry.position_m, dtype=np.float64)
        vel = np.asarray(telemetry.velocity_mps, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return pos.shape == (3,) and vel.shape == (3,) and bool(np.all(np.isfinite(pos)) and np.all(np.isfinite(vel)))


def closing_unsafe(telemetry, constraints: ConstraintConfig) -> bool:
    """Inside the keep-out radius and closing faster than the limit?"""
    if not _telemetry_usable(telemetry):
        return False
    distance = float(np.linalg.norm(telemetry.position_m))
    return bool(
        distance < constraints.keep_out_radius_m
        and closing_speed(telemetry.position_m, telemetry.velocity_mps) > constraints.max_closing_speed_mps
    )


def _hold_or_brake(telemetry, constraints, code, reason, event_id) -> FilterResult:
    """Safe output when the command or goal cannot be trusted."""
    if closing_unsafe(telemetry, constraints):
        return FilterResult(_brake_command(telemetry, constraints), True, code, reason + "; certified braking enforced", event_id)
    return FilterResult(_zero(), True, code, reason + "; safe hold enforced", event_id)


def filter_action(goal: GoalSpec, telemetry: Telemetry, commanded: ActuatorCommand, constraints: ConstraintConfig, now_ns: int, event_id: str = "OVR-000000") -> FilterResult:
    validation = validate_goal(goal, constraints, telemetry, now_ns)
    if validation.code in _HOLD_CODES:
        return _hold_or_brake(telemetry, constraints, validation.code, validation.reason, event_id)
    if goal.goal_type is GoalType.ABORT:
        return FilterResult(_brake_command(telemetry, constraints), True, ReasonCode.ABORT_ENFORCED, "ABORT goal enforces certified braking", event_id)
    if goal.goal_type is GoalType.SAFE_HOLD:
        return _hold_or_brake(telemetry, constraints, ReasonCode.SAFE_HOLD_ENFORCED, "SAFE_HOLD goal enforces zero discretionary thrust", event_id)

    try:
        force = np.asarray(commanded.force_n, dtype=np.float64); torque = np.asarray(commanded.torque_nm, dtype=np.float64)
    except (TypeError, ValueError):
        return _hold_or_brake(telemetry, constraints, ReasonCode.INVALID_VECTOR, "malformed command rejected", event_id)
    if force.shape != (3,) or torque.shape != (3,) or not np.all(np.isfinite(force)) or not np.all(np.isfinite(torque)):
        return _hold_or_brake(telemetry, constraints, ReasonCode.INVALID_VECTOR, "non-finite command rejected", event_id)

    max_force = np.asarray(constraints.max_force_n); max_torque = np.asarray(constraints.max_torque_nm)
    clipped_force = np.clip(force, -max_force, max_force)
    clipped_torque = np.clip(torque, -max_torque, max_torque)
    was_clipped = not np.allclose(force, clipped_force) or not np.allclose(torque, clipped_torque)

    pos = np.asarray(telemetry.position_m, dtype=np.float64); vel = np.asarray(telemetry.velocity_mps, dtype=np.float64)
    period = float(constraints.control_period_s)
    next_vel = vel + clipped_force / float(constraints.vehicle_mass_kg) * period
    next_pos = pos + 0.5 * (vel + next_vel) * period
    closing_next = closing_speed(next_pos, next_vel)

    if closing_unsafe(telemetry, constraints):
        # Already too fast inside the keep-out: only a command that pushes away
        # from the target (never toward it) and does not raise the speed keeps
        # authority. Judged at the current line of sight -- a predicted state
        # that has already flown past the target would look "opening" and
        # wrongly excuse thrust toward it.
        line_of_sight = pos / max(float(np.linalg.norm(pos)), 1e-12)
        improving = (float(np.dot(clipped_force, line_of_sight)) > 0.0
                     and float(np.linalg.norm(next_vel)) <= float(np.linalg.norm(vel)))
        if not improving:
            return FilterResult(_brake_command(telemetry, constraints), True, ReasonCode.CLOSING_SPEED_EXCEEDED,
                                "command does not reduce closing speed inside keep-out; certified braking enforced", event_id)
    elif float(np.linalg.norm(next_pos)) < constraints.keep_out_radius_m and closing_next > constraints.max_closing_speed_mps:
        return FilterResult(_zero(), True, ReasonCode.CLOSING_SPEED_EXCEEDED,
                            "command would exceed the closing-speed limit inside keep-out; safe hold enforced", event_id)

    if was_clipped:
        return FilterResult(ActuatorCommand(tuple(float(v) for v in clipped_force), tuple(float(v) for v in clipped_torque)),
                            True, ReasonCode.COMMAND_CLIPPED, "command exceeded actuator authority and was clipped", event_id)
    return FilterResult(commanded, False, ReasonCode.ACCEPTED, "command accepted", "")
