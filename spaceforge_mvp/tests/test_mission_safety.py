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
    """Case 1: fuel=0.5, dist=10, 'APPROACH' -> accepted."""
    result = evaluate_goal(goal_type="APPROACH", current_dist=10.0, fuel_remaining=0.5)
    assert result["status"] == "accepted"
    assert "reason" in result


def test_case_2_approach_below_safe_distance():
    """Case 2: fuel=0.5, dist=3, 'APPROACH' -> rejected."""
    result = evaluate_goal(goal_type="APPROACH", current_dist=3.0, fuel_remaining=0.5)
    assert result["status"] == "rejected"
    assert "reason" in result


def test_case_3_fuel_below_floor_triggers_emergency_hold():
    """Case 3: fuel=0.02, dist=10, 'APPROACH' -> emergency_hold (fuel check wins even though distance is fine)."""
    result = evaluate_goal(goal_type="APPROACH", current_dist=10.0, fuel_remaining=0.02)
    assert result["status"] == "emergency_hold"
    assert "reason" in result


def test_case_4_inspect_at_close_distance_accepted():
    """Case 4: fuel=0.5, dist=3, 'INSPECT' -> accepted (distance check is APPROACH-only)."""
    result = evaluate_goal(goal_type="INSPECT", current_dist=3.0, fuel_remaining=0.5)
    assert result["status"] == "accepted"
    assert "reason" in result
