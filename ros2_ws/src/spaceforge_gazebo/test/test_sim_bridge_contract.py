from pathlib import Path

ROOT = Path(__file__).parents[1]

def test_bridge_contract_is_mechanical_and_complete():
    source = (ROOT / "src" / "sim_bridge.cpp").read_text()
    docs = (ROOT / "SIM_BRIDGE.md").read_text()
    for token in ("rosgraph_msgs/msg/Clock", "nav_msgs/msg/Odometry", "sensor_msgs/msg/Imu", "sensor_msgs/msg/Image", "sensor_msgs/msg/LaserScan", "geometry_msgs/msg/WrenchStamped"):
        assert token in docs
    for topic in ("/clock", "/chaser/odometry", "/chaser/imu", "/chaser/camera/image_raw", "/chaser/lidar/scan", "/chaser/cmd_wrench"):
        assert topic in source or topic.lstrip("/") in docs
    assert "policy" not in source.lower()
    assert "safety" not in source.lower()
    assert "llm" not in source.lower()
    assert "header.stamp = stamp" in source
    assert "command/accel" in source
