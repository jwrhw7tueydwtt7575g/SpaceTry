"""ROS message adapters; core mission_ctrl remains ROS-independent."""
import json
from .schema import ActuatorCommand
from .telemetry import OverrideEvent

def event_payload(event: OverrideEvent): return event.to_dict()
def command_payload(command: ActuatorCommand): return {"force_n": list(command.force_n), "torque_nm": list(command.torque_nm)}

def to_ros_wrench(command, stamp_ns=0, frame_id="target_lvlh"):
    try:
        from geometry_msgs.msg import WrenchStamped
    except ImportError as exc: raise RuntimeError("geometry_msgs is required for ROS message conversion") from exc
    msg = WrenchStamped(); msg.header.stamp.sec = stamp_ns // 1_000_000_000; msg.header.stamp.nanosec = stamp_ns % 1_000_000_000; msg.header.frame_id = frame_id
    # rclpy message fields are strictly typed: ints must be converted to float.
    msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z = (float(v) for v in command.force_n)
    msg.wrench.torque.x, msg.wrench.torque.y, msg.wrench.torque.z = (float(v) for v in command.torque_nm)
    return msg

def to_ros_event(event):
    try:
        from std_msgs.msg import String
    except ImportError as exc: raise RuntimeError("std_msgs is required for ROS event conversion") from exc
    msg = String(); msg.data = json.dumps(event_payload(event), sort_keys=True); return msg
