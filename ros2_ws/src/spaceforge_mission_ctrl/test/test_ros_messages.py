from spaceforge_mission_ctrl import ActuatorCommand
from spaceforge_mission_ctrl.ros_messages import to_ros_wrench

def test_ros_message_conversion_requires_ros_or_is_explicit():
    try:
        msg = to_ros_wrench(ActuatorCommand((1,0,0),(0,0,0)), 2_000_000_001)
    except RuntimeError as exc:
        assert "geometry_msgs" in str(exc)
    else:
        assert msg.header.stamp.sec == 2 and msg.header.stamp.nanosec == 1
