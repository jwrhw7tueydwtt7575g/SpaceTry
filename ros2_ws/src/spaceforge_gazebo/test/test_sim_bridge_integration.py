"""Live ROS 2 + Gazebo integration: clock, odometry and command flow end to end.

Runs only where ROS 2 Humble, Gazebo Harmonic and this workspace are sourced
(registered with CTest, so colcon test runs it on the ROS CI job).
"""
import os
import shutil
import signal
import subprocess
import time

import pytest


def _echo(topic, field_filter=None, timeout=10.0):
    out = subprocess.run(["ros2", "topic", "echo", topic, "--once"], capture_output=True, text=True, timeout=timeout)
    assert out.returncode == 0, out.stderr
    return out.stdout


def _odom_vx(text):
    # nav_msgs/Odometry YAML: twist.twist.linear.x is the first "x:" after "linear:" in twist.
    twist = text.split("twist:", 1)[1]
    linear = twist.split("linear:", 1)[1]
    return float(linear.split("x:", 1)[1].split()[0])


def test_sim_bridge_integration_clock_odometry_and_command_flow():
    if not shutil.which("ros2") or not shutil.which("gz"):
        pytest.skip("ROS 2 Humble and Gazebo Harmonic are unavailable")
    if "spaceforge_gazebo" not in os.environ.get("AMENT_PREFIX_PATH", ""):
        pytest.skip("spaceforge_gazebo is not installed in the sourced ROS workspace")
    process = subprocess.Popen(["ros2", "launch", "spaceforge_gazebo", "leo_orbit.launch.py", "headless:=true"],
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True)
    try:
        deadline = time.time() + 30.0
        clock = ""
        while time.time() < deadline:
            try:
                clock = _echo("/spaceforge/clock", timeout=5)
                break
            except (subprocess.TimeoutExpired, AssertionError):
                time.sleep(1.0)
        assert "clock:" in clock, "no simulation clock bridged from Gazebo"

        before = _odom_vx(_echo("/spaceforge/chaser/odometry"))
        command = subprocess.run(
            ["ros2", "topic", "pub", "--times", "5", "--rate", "5", "/spaceforge/chaser/cmd_wrench",
             "geometry_msgs/msg/WrenchStamped", "{wrench: {force: {x: -0.1}}}"],
            capture_output=True, text=True, timeout=20)
        assert command.returncode == 0, command.stderr
        time.sleep(2.0)
        after = _odom_vx(_echo("/spaceforge/chaser/odometry"))
        # -0.1 N on the 1 kg chaser for over a second must drive vx clearly negative.
        assert after < before - 0.05, (before, after)
    finally:
        os.killpg(process.pid, signal.SIGINT)
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
