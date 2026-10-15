import pytest
from spaceforge_rl_env.contract import SensorState
from spaceforge_rl_env.env import SpacecraftRLEnv

class StaleTransport:
    def reset_simulator(self, seed=None):
        return SensorState(0, (1.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    def publish_wrench(self, force_n, torque_nm, stamp_ns):
        pass
    def wait_for_sensor(self, after_stamp_ns, timeout_s):
        return SensorState(after_stamp_ns, (1.0, 0.0, 0.0), (0.0, 0.0, 0.0))

def test_stale_sensor_is_safe_truncation():
    env = SpacecraftRLEnv(transport=StaleTransport(), sensor_timeout_s=0.001)
    env.reset()
    _, reward, terminated, truncated, info = env.step([0, 0, 0, 0, 0, 0])
    assert not terminated and truncated
    assert reward == -1.0
    assert info["stale_sensor"]

def test_ros_gazebo_integration_requires_host_tools():
    pytest.importorskip("rclpy")
    pytest.skip("Requires sourced ROS 2 Humble and running Gazebo Harmonic world")
