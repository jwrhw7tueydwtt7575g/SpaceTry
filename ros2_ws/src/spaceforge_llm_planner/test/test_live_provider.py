import importlib.util
import pytest

def test_live_provider_requires_explicit_environment():
    if not importlib.util.find_spec("rclpy"):
        pytest.skip("ROS 2 unavailable; planner ROS integration is not claimed")
    pytest.skip("Network/provider integration is never part of the default suite")
