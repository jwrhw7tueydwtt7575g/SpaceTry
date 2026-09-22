"""Tests for mission safety interlock module."""

import sys
from pathlib import Path
import pytest

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

try:
    from spaceforge_mvp.safety.mission_safety import evaluate_goal
except ImportError:
    from safety.mission_safety import evaluate_goal


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


def test_case_2_approach_below_safe_distance_excessive_speed():
    """Case 2: fuel=0.5, dist=3, speed=0.5, 'APPROACH' -> rejected (dist < 5m and speed > 0.2m/s)."""
    result = evaluate_goal(
        goal_type="APPROACH",
        current_dist=3.0,
        fuel_remaining=0.5,
        current_speed=0.5,
    )
    assert result["status"] == "rejected"
    assert "reason" in result


def test_case_3_fuel_below_floor_triggers_emergency_hold():
    """Case 3: fuel=0.02, dist=10, speed=0.1, 'APPROACH' -> emergency_hold (fuel check wins regardless)."""
    result = evaluate_goal(
        goal_type="APPROACH",
        current_dist=10.0,
        fuel_remaining=0.02,
        current_speed=0.1,
    )
    assert result["status"] == "emergency_hold"
    assert "reason" in result


def test_case_4_inspect_at_close_distance_accepted():
    """Case 4: fuel=0.5, dist=3, speed=0.5, 'INSPECT' -> accepted (distance/speed check is APPROACH-only)."""
    result = evaluate_goal(
        goal_type="INSPECT",
        current_dist=3.0,
        fuel_remaining=0.5,
        current_speed=0.5,
    )
    assert result["status"] == "accepted"
    assert "reason" in result


def test_case_5_approach_close_distance_safe_speed_accepted():
    """New Case (a): fuel=0.5, dist=3, speed=0.1, 'APPROACH' -> accepted (speed <= 0.2m/s threshold)."""
    result = evaluate_goal(
        goal_type="APPROACH",
        current_dist=3.0,
        fuel_remaining=0.5,
        current_speed=0.1,
    )
    assert result["status"] == "accepted"
    assert "reason" in result


def test_case_6_approach_close_distance_excessive_speed_rejected():
    """New Case (b): fuel=0.5, dist=3, speed=0.5, 'APPROACH' -> rejected (speed > 0.2m/s threshold)."""
    result = evaluate_goal(
        goal_type="APPROACH",
        current_dist=3.0,
        fuel_remaining=0.5,
        current_speed=0.5,
    )
    assert result["status"] == "rejected"
    assert "reason" in result
