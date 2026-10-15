import importlib.util
import pytest

def test_live_gazebo_dependencies_are_explicit():
    if not importlib.util.find_spec("rclpy"):
        pytest.skip("ROS 2 is not installed; live Gazebo trainer test is not claimed")
    pytest.skip("Requires a sourced ROS 2 workspace and running Gazebo scenario")
