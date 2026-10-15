"""Tests for mission safety interlock module."""

import numpy as np
from spaceforge_mvp.safety.mission_safety import evaluate_goal


def test_case_1_approach_safe_distance_adequate_fuel():
    """Case 1: fuel=0.5, dist=10, speed=0.1, 'APPROACH' -> accepted."""
    result = evaluate_goal(
        goal_type="APPROACH",
        current_dist=10.0,
        fuel_remaining=0.5,
        current_speed=0.1,
    )
    assert result["status"] == "accepted"
    assert "reason" in result


def test_case_2_approach_below_safe_distance_excessive_speed_accelerating_rejected():
    """Case 2: dist=3m < 5m, speed=0.5m/s > 0.2m/s with accelerating thrust -> rejected."""
    result = evaluate_goal(
        goal_type="APPROACH",
        current_dist=3.0,
        fuel_remaining=0.5,
        current_speed=0.5,
        commanded_action=np.array([1.0, 0.0, 0.0]),
        rel_vel=np.array([0.5, 0.0, 0.0]),
    )
    assert result["status"] == "rejected"
    assert "reason" in result


def test_case_3_fuel_below_floor_triggers_emergency_hold():
    """Case 3: fuel=0.02 < 0.05 -> emergency_hold (fuel check wins regardless)."""
    result = evaluate_goal(
        goal_type="APPROACH",
        current_dist=10.0,
        fuel_remaining=0.02,
        current_speed=0.1,
    )
    assert result["status"] == "emergency_hold"
    assert "reason" in result


def test_case_4_inspect_at_close_distance_accepted():
    """Case 4: fuel=0.5, dist=3, speed=0.5, 'INSPECT' -> accepted (distance check is APPROACH-only)."""
    result = evaluate_goal(
        goal_type="INSPECT",
        current_dist=3.0,
        fuel_remaining=0.5,
        current_speed=0.5,
    )
    assert result["status"] == "accepted"
    assert "reason" in result


def test_case_5_approach_close_distance_safe_speed_accepted():
    """Case 5: dist=3, speed=0.1 <= 0.2 -> accepted."""
    result = evaluate_goal(
        goal_type="APPROACH",
        current_dist=3.0,
        fuel_remaining=0.5,
        current_speed=0.1,
    )
    assert result["status"] == "accepted"
    assert "reason" in result


def test_case_6_approach_close_distance_excessive_speed_braking_thrust_accepted():
    """Case 6: dist=3m < 5m, speed=0.5m/s > 0.2m/s WITH BRAKING THRUST (u dot v < 0) -> ACCEPTED."""
    result = evaluate_goal(
        goal_type="APPROACH",
        current_dist=3.0,
        fuel_remaining=0.5,
        current_speed=0.5,
        commanded_action=np.array([-1.0, 0.0, 0.0]),  # Opposes vel [0.5, 0, 0]
        rel_vel=np.array([0.5, 0.0, 0.0]),
    )
    assert result["status"] == "accepted"
    assert "Braking" in result["reason"] or "accepted" in result["reason"]
