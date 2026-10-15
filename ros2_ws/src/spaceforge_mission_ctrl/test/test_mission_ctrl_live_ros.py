import importlib.util
import pytest

def test_live_ros_mission_ctrl_requires_ros():
    if not importlib.util.find_spec("rclpy"):
        pytest.skip("ROS 2 is unavailable; live mission_ctrl node is not claimed")
    pytest.skip("Requires a sourced ROS 2 workspace")
