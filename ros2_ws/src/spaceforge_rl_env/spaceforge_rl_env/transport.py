"""Transport adapters for the ROS-topic RL environment."""
from __future__ import annotations
from typing import Optional
import time
import numpy as np
from .contract import SensorState

class MockTransport:
    """Deterministic adapter backed by the existing pure-Python SpacecraftEnv.

    Communication latency is modeled as a whole-control-step delay line:
    delay_steps = ceil(latency_ms / 1000 / backend_dt). Zero latency (default)
    applies each published wrench on the next sensor wait, exactly as before.
    At dt=1 s any nonzero latency costs one step (conservative quantization);
    use backend dt <= 0.01 s for realistic 1-10 ms studies. Thrust noise is an
    additive Gaussian on the applied force; all draws come from a per-episode
    RNG seeded by reset_simulator, so identical seeds replay identically.
    """
    def __init__(self, backend, mass_kg: float = 1.0, latency_ms: float = 0.0,
                 thrust_noise_std: float = 0.0):
        self.backend = backend
        self.mass_kg = float(mass_kg)
        if float(latency_ms) < 0.0:
            raise ValueError("latency_ms must be non-negative")
        if float(thrust_noise_std) < 0.0:
            raise ValueError("thrust_noise_std must be non-negative")
        self.latency_ms = float(latency_ms)
        self.thrust_noise_std = float(thrust_noise_std)
        self._pending_force = np.zeros(3, dtype=np.float64)
        self._pending_torque = np.zeros(3, dtype=np.float64)
        self._sample: Optional[SensorState] = None
        self.published_commands = []
        self.reset_calls = 0
        # Latest per-step info dict returned by the backend (safety_override,
        # reward_breakdown, ...). Surfaced so evaluators can count interlock
        # fires; never used for control.
        self.last_backend_info: dict = {}
        self._delay_line: list = []
        self._rng = np.random.default_rng(0)

    @property
    def delay_steps(self) -> int:
        """Whole control steps a command waits before execution."""
        import math
        dt = float(self.backend.physics_config.dt)
        if self.latency_ms <= 0.0:
            return 0
        return max(1, int(math.ceil(self.latency_ms / 1000.0 / dt)))

    def _state(self) -> SensorState:
        return SensorState(
            stamp_ns=int(self.backend.step_count * self.backend.physics_config.dt * 1e9),
            relative_position_m=tuple(float(v) for v in self.backend.rel_pos),
            relative_velocity_mps=tuple(float(v) for v in self.backend.rel_vel),
            fuel_fraction=float(self.backend.fuel),
        )

    def reset_simulator(self, seed: Optional[int] = None) -> SensorState:
        self.reset_calls += 1
        self.backend.reset(seed=seed)
        self.last_backend_info = {}
        self._delay_line = []
        self._rng = np.random.default_rng(0 if seed is None else int(seed))
        self._sample = self._state()
        return self._sample

    def publish_wrench(self, force_n, torque_nm, stamp_ns: int) -> None:
        self._pending_force = np.asarray(force_n, dtype=np.float64)
        self._pending_torque = np.asarray(torque_nm, dtype=np.float64)
        self.published_commands.append((int(stamp_ns), self._pending_force.copy(), self._pending_torque.copy()))

    def wait_for_sensor(self, after_stamp_ns: int, timeout_s: float) -> Optional[SensorState]:
        del timeout_s
        self._delay_line.append(np.asarray(self._pending_force, dtype=np.float64))
        if len(self._delay_line) > self.delay_steps:
            force = self._delay_line.pop(0)
        else:
            force = np.zeros(3, dtype=np.float64)
        if self.thrust_noise_std > 0.0:
            force = force + self._rng.normal(0.0, self.thrust_noise_std, size=3)
        action = force / self.mass_kg
        _, _, terminated, truncated, info = self.backend.step(action.astype(np.float32))
        self.last_backend_info = dict(info)
        self._sample = self._state()
        self._sample = SensorState(**{**self._sample.__dict__, "frame_id": "target_lvlh", "child_frame_id": "chaser_base"})
        return self._sample if self._sample.stamp_ns > after_stamp_ns or terminated or truncated else None

class RosTransportAdapter:
    """Lazy ROS 2/Gazebo transport implementation; imports ROS only when used."""
    def __init__(self, node, topics, mass_kg: float = 1.0):
        try:
            from geometry_msgs.msg import WrenchStamped
            from nav_msgs.msg import Odometry
        except ImportError as exc:
            raise RuntimeError("ROS 2 message packages are required for RosTransportAdapter") from exc
        self.node = node
        self.mass_kg = float(mass_kg)
        self.latest: Optional[SensorState] = None
        self.publisher = node.create_publisher(WrenchStamped, topics["command"], 10)
        node.create_subscription(Odometry, topics["odometry"], self._on_odom, 10)

    def _on_odom(self, msg):
        stamp_ns = int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec)
        self.latest = SensorState(stamp_ns=stamp_ns, relative_position_m=(msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z), relative_velocity_mps=(msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.linear.z), vehicle_orientation_xyzw=(msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w), frame_id=msg.header.frame_id, child_frame_id=msg.child_frame_id)

    def reset_simulator(self, seed=None):
        del seed
        self.latest = None
        raise RuntimeError("Gazebo reset service wiring is deployment-specific; use the bridge reset contract")

    def publish_wrench(self, force_n, torque_nm, stamp_ns):
        from geometry_msgs.msg import WrenchStamped
        msg = WrenchStamped()
        msg.header.stamp.sec = int(stamp_ns // 1_000_000_000); msg.header.stamp.nanosec = int(stamp_ns % 1_000_000_000)
        msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z = [float(v) for v in force_n]
        msg.wrench.torque.x, msg.wrench.torque.y, msg.wrench.torque.z = [float(v) for v in torque_nm]
        self.publisher.publish(msg)

    def wait_for_sensor(self, after_stamp_ns, timeout_s):
        import rclpy
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            rclpy.spin_once(self.node, timeout_sec=min(0.01, max(0.0, deadline - time.monotonic())))
            if self.latest is not None and self.latest.stamp_ns > after_stamp_ns:
                return self.latest
        return None
