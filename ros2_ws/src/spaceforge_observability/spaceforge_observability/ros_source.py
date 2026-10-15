"""Optional ROS 2 source: ROS topics become schema envelopes for the bridge.

Lazy ROS imports: usable only on a sourced ROS 2 host. Offline replay and all
default tests never touch this module's ROS path.
"""
from .bridge import TelemetryBridge
from .schema import make_envelope


def odometry_from_msg(msg) -> dict:
    stamp_ns = int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec)
    return make_envelope("odometry", {
        "position_m": (float(msg.pose.pose.position.x), float(msg.pose.pose.position.y),
                       float(msg.pose.pose.position.z)),
        "velocity_mps": (float(msg.twist.twist.linear.x), float(msg.twist.twist.linear.y),
                         float(msg.twist.twist.linear.z))},
        stamp_ns, msg.header.frame_id or "target_lvlh")


def wrench_envelope(stream: str, msg, override=None) -> dict:
    if stream not in ("commanded", "applied_command"):
        raise ValueError(f"not a wrench stream: {stream}")
    stamp_ns = int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec)
    payload = {"force_n": (float(msg.wrench.force.x), float(msg.wrench.force.y), float(msg.wrench.force.z)),
               "torque_nm": (float(msg.wrench.torque.x), float(msg.wrench.torque.y), float(msg.wrench.torque.z))}
    if stream == "applied_command":
        payload["override"] = override
    return make_envelope(stream, payload, stamp_ns, msg.header.frame_id or "target_lvlh")


def main():
    try:
        import rclpy  # noqa: F401 -- availability guard: bare name documents the ROS requirement
        from rclpy.node import Node
        from geometry_msgs.msg import PoseStamped, WrenchStamped
        from nav_msgs.msg import Odometry
        from std_msgs.msg import String
    except ImportError as exc:
        raise RuntimeError("ROS 2 geometry_msgs, nav_msgs, and std_msgs are required") from exc

    class ObservabilitySource(Node):
        def __init__(self, bridge: TelemetryBridge):
            super().__init__("observability_source")
            self.bridge = bridge
            self.create_subscription(Odometry, "odometry", self.on_odom, 10)
            self.create_subscription(WrenchStamped, "commanded_wrench", self.on_commanded, 10)
            self.create_subscription(WrenchStamped, "applied_wrench", self.on_applied, 10)
            self.create_subscription(PoseStamped, "target_pose", self.on_pose, 10)
            self.create_subscription(String, "override_events", self.on_event, 10)
            self.create_subscription(String, "status", self.on_status, 10)

        def _ingest(self, envelope):
            ok, reason = self.bridge.ingest(envelope)
            if not ok:
                self.get_logger().warn(f"observability: dropped invalid envelope: {reason}")

        def on_odom(self, msg):
            self._ingest(odometry_from_msg(msg))

        def on_commanded(self, msg):
            self._ingest(wrench_envelope("commanded", msg))

        def on_applied(self, msg):
            self._ingest(wrench_envelope("applied_command", msg))

        def on_pose(self, msg):
            stamp_ns = int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec)
            self._ingest(make_envelope("target_pose", {
                "translation_m": (float(msg.pose.position.x), float(msg.pose.position.y),
                                  float(msg.pose.position.z)),
                "quaternion_xyzw": (float(msg.pose.orientation.x), float(msg.pose.orientation.y),
                                    float(msg.pose.orientation.z), float(msg.pose.orientation.w)),
                "confidence": 1.0}, stamp_ns, msg.header.frame_id or "camera_optical_frame"))

        def on_event(self, msg):
            import json as _json
            try:
                data = _json.loads(msg.data)
            except (ValueError, TypeError):
                return
            self._ingest(make_envelope("safety_event", {
                "event_id": str(data.get("event_id", "")), "goal_type": str(data.get("goal_type", "")),
                "reason_code": str(data.get("reason_code", "")), "reason": str(data.get("reason", "")),
                "commanded_force_n": tuple(data.get("commanded_force_n", (0, 0, 0))),
                "applied_force_n": tuple(data.get("applied_force_n", (0, 0, 0)))}, 0))

        def on_status(self, msg):
            self._ingest(make_envelope("llm_trace", {
                "trace_id": "ros-status", "status": str(msg.data)[:200],
                "provider": "ros_topic"}, 0))

    raise SystemExit("use the web_bridge entry point: spaceforge-observability --help")


if __name__ == "__main__":
    main()
