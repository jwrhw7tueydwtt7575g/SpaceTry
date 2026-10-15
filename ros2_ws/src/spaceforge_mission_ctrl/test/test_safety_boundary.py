"""Exhaustive safety-boundary tests: every reason code, hostile inputs, invariants.

No ROS, no policy, no LLM: the boundary must hold against arbitrary callers.
"""
import json
import numpy as np
import pytest
from spaceforge_mission_ctrl import (
    ActuatorCommand, ConstraintConfig, GoalSpec, GoalType, MissionController,
    ReasonCode, Telemetry, validate_goal,
)
from spaceforge_mission_ctrl.filtering import _brake
from spaceforge_mission_ctrl.node import parse_goal_json
from spaceforge_mission_ctrl.ros_messages import command_payload, to_ros_event, to_ros_wrench

NOW = 1_000_000_000


def fresh_telemetry(**kwargs):
    values = dict(position_m=(10.0, 0.0, 0.0), velocity_mps=(0.0, 0.0, 0.0),
                  fuel_fraction=1.0, stamp_ns=NOW)
    values.update(kwargs)
    return Telemetry(**values)


def fresh_goal(kind="RETREAT", **kwargs):
    return GoalSpec(GoalType(kind), **kwargs)


def hostile_command():
    return ActuatorCommand((float("inf"), float("nan"), 1e300), (0.0, 0.0, 0.0))


# ---------------------------------------------------------- reason-code reachability
@pytest.mark.parametrize("kind", ["APPROACH", "INSPECT", "SAFE_HOLD", "RETREAT", "ABORT"])
def test_every_goal_type_validates_clean(kind):
    result = validate_goal(fresh_goal(kind), ConstraintConfig(), fresh_telemetry(), NOW)
    assert result == result  # frozen dataclass equality
    assert (result.accepted, result.code) == (True, ReasonCode.ACCEPTED)


def test_invalid_goal_type_rejected_with_stable_reason():
    goal = fresh_goal()
    bad = GoalSpec.__new__(GoalSpec)  # hostile caller bypassing the type system
    object.__setattr__(bad, "goal_type", "APPROACH")  # plain str, not GoalType
    for field in ("target_position_m", "desired_distance_m", "desired_closing_speed_mps",
                  "timeout_s", "max_force_n", "max_torque_nm", "goal_id"):
        object.__setattr__(bad, field, getattr(goal, field))
    result = validate_goal(bad, ConstraintConfig(), fresh_telemetry(), NOW)
    assert (result.accepted, result.code) == (False, ReasonCode.INVALID_GOAL_TYPE)


@pytest.mark.parametrize("kwargs,code", [
    ({"target_position_m": (float("inf"), 0.0, 0.0)}, ReasonCode.INVALID_VECTOR),
    ({"desired_distance_m": -1.0}, ReasonCode.INVALID_VECTOR),
    ({"timeout_s": 0.0}, ReasonCode.INVALID_TIMEOUT),
    ({"timeout_s": 301.0}, ReasonCode.INVALID_TIMEOUT),
    ({"desired_distance_m": 2.0}, ReasonCode.KEEP_OUT_VIOLATION),  # APPROACH default kind below
    ({"desired_closing_speed_mps": 5.0}, ReasonCode.CLOSING_SPEED_EXCEEDED),
    ({"max_force_n": (9.0, 1.0, 1.0)}, ReasonCode.ACTUATOR_LIMIT_EXCEEDED),
])
def test_goal_shape_violations(kwargs, code):
    kind = "APPROACH" if code in (ReasonCode.KEEP_OUT_VIOLATION,) else "RETREAT"
    result = validate_goal(fresh_goal(kind, **kwargs), ConstraintConfig(), fresh_telemetry(), NOW)
    assert (result.accepted, result.code) == (False, code)


def test_telemetry_violations():
    cfg = ConstraintConfig()
    r = validate_goal(fresh_goal(), cfg, fresh_telemetry(fuel_fraction=0.01), NOW)
    assert r.code is ReasonCode.FUEL_BELOW_FLOOR
    r = validate_goal(fresh_goal(), cfg, fresh_telemetry(stamp_ns=0), NOW)
    assert r.code is ReasonCode.STALE_TELEMETRY
    r = validate_goal(fresh_goal(), cfg, fresh_telemetry(position_m=(float("nan"), 0, 0)), NOW)
    assert r.code is ReasonCode.INVALID_VECTOR
    r = validate_goal(fresh_goal(), cfg, fresh_telemetry(fuel_fraction=1.5), NOW)
    assert r.code is ReasonCode.INVALID_VECTOR


def test_hostile_command_never_leaks_through():
    ctrl = MissionController()
    result, event = ctrl.step(fresh_goal(), fresh_telemetry(), hostile_command(), NOW)
    assert result.override and result.code is ReasonCode.INVALID_VECTOR
    assert result.applied == ActuatorCommand((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    assert event is not None and event.reason_code == ReasonCode.INVALID_VECTOR.value


def test_keep_out_veto_substitutes_braking_not_zero():
    ctrl = MissionController()
    tele = fresh_telemetry(position_m=(3.0, 0.0, 0.0), velocity_mps=(1.0, 0.0, 0.0))
    result, _ = ctrl.step(fresh_goal(), tele, ActuatorCommand((1.0, 0.0, 0.0), (0.0, 0.0, 0.0)), NOW)
    assert result.code is ReasonCode.CLOSING_SPEED_EXCEEDED and result.override
    applied = np.asarray(result.applied.force_n)
    assert float(np.dot(applied, tele.velocity_mps)) < 0.0  # braking, not coasting


# ---------------------------------------------------------- RTA reuse + invariants
def test_brake_matches_proven_rta_law():
    from spaceforge_mvp.control.rta import certified_brake_action
    rng = np.random.default_rng(11)
    for _ in range(50):
        vel = rng.normal(0, 2, 3)
        got = _brake(vel, (1.0, 1.0, 1.0))
        want = np.clip(certified_brake_action(vel), -1.0, 1.0)
        np.testing.assert_allclose(got, want, atol=1e-12)
        if float(np.linalg.norm(vel)) > 1e-9:
            assert float(np.dot(got, vel)) < 0.0


def test_safety_invariants_over_hostile_space():
    rng = np.random.default_rng(7)
    kinds = ["APPROACH", "INSPECT", "SAFE_HOLD", "RETREAT", "ABORT"]
    limits = np.asarray((1.0, 1.0, 1.0))
    for trial in range(500):
        pos = tuple(rng.normal(0, 8, 3))
        vel = tuple(rng.normal(0, 2, 3))
        fuel = float(rng.uniform(-0.1, 1.1))
        stamp = NOW if rng.random() > 0.1 else 0
        tele = Telemetry(pos, vel, fuel, stamp)
        goal = fresh_goal(rng.choice(kinds))
        force = tuple(rng.normal(0, 5, 3))
        if rng.random() < 0.05:
            force = (float("inf"), float("-inf"), float("nan"))
        ctrl = MissionController()
        result, event = ctrl.step(goal, tele, ActuatorCommand(force, (0.0, 0.0, 0.0)), NOW)
        applied_f = np.asarray(result.applied.force_n, dtype=float)
        applied_t = np.asarray(result.applied.torque_nm, dtype=float)
        assert np.all(np.isfinite(applied_f)) and np.all(np.isfinite(applied_t))
        assert np.all(np.abs(applied_f) <= limits + 1e-12)
        assert (not result.override) or (event is not None)
        if fuel < 0.05 and 0.0 <= fuel <= 1.0 and stamp == NOW:
            # Fresh telemetry below the floor: fuel priority beats everything.
            assert result.applied == ActuatorCommand((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
        # Determinism: same inputs, same verdict and output (event ids excluded).
        again, _ = MissionController().step(goal, tele, ActuatorCommand(force, (0.0, 0.0, 0.0)), NOW)
        assert (again.code, again.override, again.applied) == (result.code, result.override, result.applied)


# ---------------------------------------------------------- events + messages
def test_every_override_is_audited_with_matching_content():
    ctrl = MissionController()
    accepted, no_event = ctrl.step(fresh_goal(), fresh_telemetry(), ActuatorCommand((0.1, 0, 0), (0, 0, 0)), NOW)
    assert not accepted.override and no_event is None
    rejected, event = ctrl.step(fresh_goal("APPROACH", timeout_s=999), fresh_telemetry(),
                                ActuatorCommand((1, 0, 0), (0, 0, 0)), NOW)
    assert rejected.override and event is not None
    assert event.reason_code == rejected.code.value == ReasonCode.INVALID_TIMEOUT.value
    assert tuple(event.applied_force_n) == rejected.applied.force_n
    assert event.event_id == "OVR-000001"
    round_tripped = json.loads(event.to_json())
    assert round_tripped["event_id"] == "OVR-000001" and round_tripped["goal_type"] == "APPROACH"


def test_ros_adapters_fail_explicitly_without_ros():
    try:
        to_ros_wrench(ActuatorCommand((1, 0, 0), (0, 0, 0)), 2_000_000_001)
    except RuntimeError as exc:
        assert "geometry_msgs" in str(exc)
    else:
        raise AssertionError("expected geometry_msgs-dependent conversion to be explicit")
    try:
        to_ros_event(None)
    except (RuntimeError, AttributeError, TypeError):
        pass
    else:
        raise AssertionError("event conversion without ROS/std_msgs must not silently succeed")
    assert set(command_payload(ActuatorCommand((1, 2, 3), (4, 5, 6))).keys()) == {"force_n", "torque_nm"}


# ---------------------------------------------------------- goal wire boundary
def test_goal_dict_round_trip():
    goal = fresh_goal("INSPECT", timeout_s=30.0, goal_id="g-1")
    restored = GoalSpec.from_dict(goal.to_dict())
    assert restored == goal


@pytest.mark.parametrize("text", [
    "not json",
    '{"goal_type": "WARP"}',
    '{"goal_type": "APPROACH", "target_position_m": [0, "x", 0]}',
    '{"goal_type": "APPROACH", "timeout_s": "soon"}',
    '[1, 2, 3]',
])
def test_malformed_wire_goals_rejected_before_validation(text):
    with pytest.raises(ValueError):
        parse_goal_json(text)


def test_wire_goal_defaults_are_explicit():
    goal = parse_goal_json('{"goal_type": "SAFE_HOLD"}')
    assert goal.goal_type is GoalType.SAFE_HOLD and goal.timeout_s == 60.0
    assert goal.max_force_n == (1.0, 1.0, 1.0)
