"""Frozen ROS/Python RL state and action contract."""
from dataclasses import dataclass
from typing import Tuple
import numpy as np

OBSERVATION_FIELDS: Tuple[str, ...] = (
    "relative_position_m_normalized", "relative_velocity_mps_normalized",
    "target_orientation_xyzw", "vehicle_orientation_xyzw",
    "vehicle_angular_velocity_radps_normalized", "fuel_fraction",
)
OBSERVATION_DIM = 18
ACTION_DIM = 6
POSITION_SCALE_M = 20.0
VELOCITY_SCALE_MPS = 2.0
ANGULAR_RATE_SCALE_RADPS = 1.0
FORCE_LIMIT_N = np.asarray((1.0, 1.0, 1.0), dtype=np.float64)
TORQUE_LIMIT_NM = np.asarray((1.0, 1.0, 1.0), dtype=np.float64)

@dataclass(frozen=True)
class SensorState:
    """One synchronized Gazebo sample, all values in SI units."""
    stamp_ns: int
    relative_position_m: Tuple[float, float, float]
    relative_velocity_mps: Tuple[float, float, float]
    target_orientation_xyzw: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    vehicle_orientation_xyzw: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    vehicle_angular_velocity_radps: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    fuel_fraction: float = 1.0
    frame_id: str = "target_lvlh"
    child_frame_id: str = "chaser_base"

    def __post_init__(self):
        if self.stamp_ns < 0 or len(self.relative_position_m) != 3 or len(self.relative_velocity_mps) != 3:
            raise ValueError("SensorState requires a non-negative timestamp and 3-vectors")
        if len(self.target_orientation_xyzw) != 4 or len(self.vehicle_orientation_xyzw) != 4:
            raise ValueError("Orientations must be xyzw 4-vectors")
        if not 0.0 <= self.fuel_fraction <= 1.0:
            raise ValueError("fuel_fraction must be in [0, 1]")

@dataclass(frozen=True)
class ActionLimits:
    force_n: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    torque_nm: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    mass_kg: float = 1.0

    def __post_init__(self):
        if self.mass_kg <= 0.0 or any(v <= 0.0 for v in self.force_n + self.torque_nm):
            raise ValueError("Action limits and mass must be positive")
