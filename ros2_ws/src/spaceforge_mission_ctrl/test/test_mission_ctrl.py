import numpy as np
from spaceforge_mission_ctrl import (ActuatorCommand, GoalSpec, GoalType, MissionController, ReasonCode, Telemetry)
from spaceforge_mission_ctrl.ros_messages import command_payload

def telemetry(stamp=1_000_000_000, fuel=1.0, position=(10,0,0), velocity=(0,0,0)):
    return Telemetry(position, velocity, fuel, stamp)

def goal(kind, **kwargs):
    return GoalSpec(GoalType(kind), **kwargs)

def command(force=(0,0,0), torque=(0,0,0)):
    return ActuatorCommand(force, torque)

def test_all_goal_types_have_typed_validation():
    ctrl = MissionController()
    for kind in GoalType:
        result = ctrl.validate_goal(goal(kind), telemetry(), 1_000_000_000)
        assert result.accepted
    assert [g.value for g in GoalType] == ["APPROACH", "INSPECT", "SAFE_HOLD", "RETREAT", "ABORT"]

def test_invalid_goals_have_stable_reasons_and_safe_output():
    ctrl = MissionController()
    bad = goal("APPROACH", timeout_s=999)
    result, event = ctrl.step(bad, telemetry(), command((1,0,0)), 1_000_000_000)
    assert result.code is ReasonCode.INVALID_TIMEOUT and result.override and result.applied == command()
    assert event.event_id == "OVR-000001"
    again, event2 = ctrl.step(bad, telemetry(), command((1,0,0)), 1_000_000_000)
    assert again.code is result.code and event2.event_id == "OVR-000002"

def test_fuel_stale_keepout_closing_and_actuator_constraints():
    ctrl = MissionController()
    r, _ = ctrl.step(goal("RETREAT"), telemetry(fuel=0.01), command((1,0,0)), 1_000_000_000); assert r.code is ReasonCode.FUEL_BELOW_FLOOR and r.applied == command()
    r, _ = ctrl.step(goal("RETREAT"), telemetry(stamp=0), command((1,0,0)), 1_000_000_000); assert r.code is ReasonCode.STALE_TELEMETRY and r.applied.force_n[0] <= 1.0
    r, _ = ctrl.step(goal("APPROACH", desired_distance_m=2), telemetry(), command((1,0,0)), 1_000_000_000); assert r.code is ReasonCode.KEEP_OUT_VIOLATION
    r, _ = ctrl.step(goal("APPROACH", desired_closing_speed_mps=1), telemetry(), command((1,0,0)), 1_000_000_000); assert r.code is ReasonCode.CLOSING_SPEED_EXCEEDED
    r, _ = ctrl.step(goal("APPROACH", max_force_n=(2,1,1)), telemetry(), command((1,0,0)), 1_000_000_000); assert r.code is ReasonCode.ACTUATOR_LIMIT_EXCEEDED and r.applied == command()

def test_safe_hold_abort_and_clipping_are_authoritative():
    ctrl = MissionController()
    hold, _ = ctrl.step(goal("SAFE_HOLD"), telemetry(), command((1,0,0)), 1_000_000_000); assert hold.code is ReasonCode.SAFE_HOLD_ENFORCED and hold.applied == command()
    abort, _ = ctrl.step(goal("ABORT"), telemetry(velocity=(0.2,0,0)), command((1,0,0)), 1_000_000_000); assert abort.code is ReasonCode.ABORT_ENFORCED and abort.applied.force_n[0] < 0
    clipped, _ = ctrl.step(goal("RETREAT"), telemetry(), command((4,0,0),(0,3,0)), 1_000_000_000); assert clipped.code is ReasonCode.COMMAND_CLIPPED and clipped.applied.force_n == (1.0,0.0,0.0)

def test_safety_invariants_over_random_commands():
    rng = np.random.default_rng(4); ctrl = MissionController()
    for _ in range(300):
        force = rng.normal(0, 10, 3); torque = rng.normal(0, 10, 3)
        result, event = ctrl.step(goal("RETREAT"), telemetry(velocity=rng.normal(0, 1, 3)), command(tuple(force), tuple(torque)), 1_000_000_000)
        assert np.all(np.isfinite(result.applied.force_n)) and np.all(np.isfinite(result.applied.torque_nm))
        assert np.all(np.abs(result.applied.force_n) <= 1.0 + 1e-12)
        assert np.all(np.abs(result.applied.torque_nm) <= 1.0 + 1e-12)
        assert (not result.override) or event is not None

def test_message_payloads_are_machine_readable():
    payload = command_payload(command((1,2,3),(4,5,6)))
    assert payload == {"force_n": [1,2,3], "torque_nm": [4,5,6]}
