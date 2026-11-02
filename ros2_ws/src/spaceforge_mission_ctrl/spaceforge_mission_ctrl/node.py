"""Optional ROS 2 node wrapping MissionController only.

Importing this module never requires ROS: goal parsing lives at module level
so the crash-proof boundary is unit-tested offline.
"""
import json
from .controller import MissionController
from .ros_messages import to_ros_event, to_ros_wrench
from .schema import ActuatorCommand, ConstraintConfig, GoalSpec, Telemetry


def parse_goal_json(text: str) -> GoalSpec:
    """Parse a goal off the wire; raises ValueError on malformed input.

    The node never lets a malformed goal replace the last valid one, so no
    producer (LLM, learned policy or anything else) can crash or bypass the
    boundary with a bad message.
    """
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, TypeError) as exc:
        raise ValueError(f"goal is not valid JSON: {exc}") from exc
    return GoalSpec.from_dict(data)

def main():
    try:
        import rclpy
        from rclpy.node import Node
        from geometry_msgs.msg import WrenchStamped
        from nav_msgs.msg import Odometry
        from std_msgs.msg import String
    except ImportError as exc: raise RuntimeError("ROS 2 geometry_msgs, nav_msgs, and std_msgs are required") from exc

    class MissionCtrlNode(Node):
        def __init__(self):
            super().__init__("mission_ctrl")
            self.controller = MissionController(ConstraintConfig())
            self.telemetry = None
            self.goal = None
            self.command = ActuatorCommand((0,0,0), (0,0,0))
            self.applied_pub = self.create_publisher(WrenchStamped, "applied_wrench", 10)
            self.event_pub = self.create_publisher(String, "override_events", 10)
            self.create_subscription(String, "goal", self.on_goal, 10)
            self.create_subscription(Odometry, "odometry", self.on_odom, 10)
            self.create_subscription(WrenchStamped, "commanded_wrench", self.on_command, 10)

        def on_goal(self, msg):
            try:
                self.goal = parse_goal_json(msg.data)
            except ValueError:
                # Malformed goal from any producer: keep the last valid goal,
                # never crash, never bypass. A deliberately bad goal still gets
                # filtered at the next odometry tick if it ever becomes active.
                self.get_logger().warn("mission_ctrl: rejected malformed goal message")

        def on_command(self, msg):
            self.command = ActuatorCommand((msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z), (msg.wrench.torque.x, msg.wrench.torque.y, msg.wrench.torque.z))

        def on_odom(self, msg):
            if self.goal is None: return
            stamp = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
            self.telemetry = Telemetry((msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z), (msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.linear.z), 1.0, stamp, msg.header.frame_id or "target_lvlh")
            result, event = self.controller.step(self.goal, self.telemetry, self.command, self.get_clock().now().nanoseconds)
            self.applied_pub.publish(to_ros_wrench(result.applied, stamp, self.telemetry.frame_id))
            if event is not None: self.event_pub.publish(to_ros_event(event))

    rclpy.init(); node = MissionCtrlNode(); rclpy.spin(node); node.destroy_node(); rclpy.shutdown()

if __name__ == "__main__": main()
