"""Gymnasium RL environment over a synchronized ROS/Gazebo transport."""
from __future__ import annotations
from typing import Optional
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from .contract import (ACTION_DIM, OBSERVATION_DIM, ANGULAR_RATE_SCALE_RADPS, POSITION_SCALE_M, VELOCITY_SCALE_MPS, ActionLimits, SensorState)
from .transport import MockTransport

class SpacecraftRLEnv(gym.Env):
    """Transport-backed environment; pure Python backend is the default fast oracle."""
    metadata = {"render_modes": []}
    def __init__(self, backend=None, transport=None, action_limits: Optional[ActionLimits] = None, max_steps: int = 300, sensor_timeout_s: float = 0.25, docking_distance_m: float = 0.5, docking_speed_mps: float = 0.1):
        super().__init__()
        if transport is None:
            if backend is None:
                from spaceforge_mvp.env.spacecraft_env import SpacecraftEnv
                backend = SpacecraftEnv()
            transport = MockTransport(backend, mass_kg=(action_limits.mass_kg if action_limits else 1.0))
        self.transport = transport
        self.limits = action_limits or ActionLimits()
        self.max_steps = int(max_steps)
        self.sensor_timeout_s = float(sensor_timeout_s)
        self.docking_distance_m = float(docking_distance_m)
        self.docking_speed_mps = float(docking_speed_mps)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(OBSERVATION_DIM,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(ACTION_DIM,), dtype=np.float32)
        self._sample: Optional[SensorState] = None
        self._last_stamp_ns = -1
        self._step_count = 0
        self._previous_distance = 0.0

    @staticmethod
    def observation_contract() -> tuple[str, ...]:
        return ("relative_position_m_normalized[0:3]", "relative_velocity_mps_normalized[3:6]", "target_orientation_xyzw[6:10]", "vehicle_orientation_xyzw[10:14]", "vehicle_angular_velocity_radps_normalized[14:17]", "fuel_fraction[17]")

    def _observe(self, sample: SensorState) -> np.ndarray:
        values = np.concatenate((
            np.asarray(sample.relative_position_m, dtype=np.float64) / POSITION_SCALE_M,
            np.asarray(sample.relative_velocity_mps, dtype=np.float64) / VELOCITY_SCALE_MPS,
            np.asarray(sample.target_orientation_xyzw, dtype=np.float64),
            np.asarray(sample.vehicle_orientation_xyzw, dtype=np.float64),
            np.asarray(sample.vehicle_angular_velocity_radps, dtype=np.float64) / ANGULAR_RATE_SCALE_RADPS,
            np.asarray([sample.fuel_fraction], dtype=np.float64),
        ))
        return values.astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._sample = self.transport.reset_simulator(seed=seed)
        self._last_stamp_ns = self._sample.stamp_ns
        self._step_count = 0
        self._previous_distance = float(np.linalg.norm(self._sample.relative_position_m))
        return self._observe(self._sample), {"sim_time_ns": self._last_stamp_ns, "frame_id": self._sample.frame_id, "child_frame_id": self._sample.child_frame_id, "reset": True}

    def step(self, action):
        if self._sample is None:
            raise RuntimeError("reset() must be called before step()")
        raw = np.asarray(action, dtype=np.float64).reshape(-1)
        if raw.shape != (ACTION_DIM,):
            raise ValueError(f"action must have shape ({ACTION_DIM},), got {raw.shape}")
        clipped = np.clip(raw, -1.0, 1.0)
        force = clipped[:3] * np.asarray(self.limits.force_n, dtype=np.float64)
        torque = clipped[3:] * np.asarray(self.limits.torque_nm, dtype=np.float64)
        self.transport.publish_wrench(force, torque, self._last_stamp_ns)
        sample = self.transport.wait_for_sensor(self._last_stamp_ns, self.sensor_timeout_s)
        self._step_count += 1
        if sample is None or sample.stamp_ns <= self._last_stamp_ns:
            info = {"termination_reason": "stale_or_missing_sensor", "stale_sensor": True, "sim_time_ns": self._last_stamp_ns, "action_clipped": bool(np.any(raw != clipped))}
            return self._observe(self._sample), -1.0, False, True, info
        self._sample = sample
        self._last_stamp_ns = sample.stamp_ns
        distance = float(np.linalg.norm(sample.relative_position_m))
        speed = float(np.linalg.norm(sample.relative_velocity_mps))
        success = distance < self.docking_distance_m and speed < self.docking_speed_mps
        progress = self._previous_distance - distance
        self._previous_distance = distance
        reward = float(progress)
        terminated = bool(success)
        truncated = bool(not terminated and self._step_count >= self.max_steps)
        info = {"sim_time_ns": sample.stamp_ns, "stale_sensor": False, "distance_m": distance, "speed_mps": speed, "is_success": success, "action_clipped": bool(np.any(raw != clipped)), "force_n": force, "torque_nm": torque}
        if truncated: info["termination_reason"] = "episode_timeout"
        if terminated: info["termination_reason"] = "docking_success"
        return self._observe(sample), reward, terminated, truncated, info

    def close(self):
        close = getattr(self.transport, "close", None)
        if close is not None: close()
