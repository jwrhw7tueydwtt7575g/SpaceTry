import os
import shutil
import subprocess
import time
import pytest


def test_sim_bridge_integration_clock_and_command_flow():
    if not shutil.which("ros2") or not shutil.which("gz"):
        pytest.skip("ROS 2 Humble and Gazebo Harmonic are unavailable")
    if "spaceforge_gazebo" not in os.environ.get("AMENT_PREFIX_PATH", ""):
        pytest.skip("spaceforge_gazebo is not installed in the sourced ROS workspace")
    process = subprocess.Popen(["ros2", "launch", "spaceforge_gazebo", "leo_orbit.launch.py"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        time.sleep(3.0)
        clock = subprocess.run(["ros2", "topic", "echo", "/spaceforge/clock", "--once"], capture_output=True, text=True, timeout=5)
        assert clock.returncode == 0
        assert "clock:" in clock.stdout
        command = subprocess.run(["ros2", "topic", "pub", "--once", "/spaceforge/chaser/cmd_wrench", "geometry_msgs/msg/WrenchStamped", "{wrench: {force: {x: 0.1}}}"], capture_output=True, text=True, timeout=5)
        assert command.returncode == 0
    finally:
        process.terminate()
        process.wait(timeout=5)
