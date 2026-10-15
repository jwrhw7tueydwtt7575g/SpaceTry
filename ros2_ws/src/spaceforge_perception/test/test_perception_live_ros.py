import importlib.util
import pytest

def test_live_ros_perception_requires_dependencies():
    if not importlib.util.find_spec("rclpy"):
        pytest.skip("ROS 2 is unavailable; live perception node is not claimed")
    pytest.skip("Requires sourced ROS 2 and camera topics")
