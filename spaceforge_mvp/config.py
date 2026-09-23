"""Centralized configuration parameters for SpaceForge MVP.

The immutable physics/safety/environment configs contain model parameters.  A
``MissionConfig`` contains per-run values and is deliberately passed into the
environment instead of being hidden in module globals.
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class PhysicsConfig:
    """Clohessy-Wiltshire physics parameters."""
    n: float = 0.00113  # Mean motion in rad/s (~400 km LEO orbit)
    dt: float = 1.0     # Simulation integration step size in seconds


@dataclass(frozen=True)
class SafetyConfig:
    """Mission safety interlock constraints."""
    safe_approach_dist: float = 5.0     # Safety threshold distance in meters
    fuel_floor: float = 0.05            # Minimum fuel fraction emergency threshold
    max_safe_closing_speed: float = 0.2 # Maximum safe speed inside 5m bubble (m/s)
    max_docking_speed: float = 0.1      # Maximum allowable relative speed at docking (<0.5m)


@dataclass(frozen=True)
class EnvConfig:
    """Environment dimensions and reward parameters."""
    max_steps: int = 300
    pos_norm_scale: float = 20.0
    vel_norm_scale: float = 2.0
    initial_pos_bound: float = 20.0
    initial_vel_bound: float = 0.005
    docking_success_dist: float = 0.5
    fuel_rate_scale: float = 0.001

    # Reward shaping weights
    progress_reward_weight: float = 5.0
    velocity_penalty_weight: float = 0.5
    action_penalty_weight: float = 0.01
    docking_bonus: float = 100.0


@dataclass(frozen=True)
class MissionConfig:
    """Inputs that define one simulation or evaluation run.

    Positions and velocities are expressed in the same Hill-frame coordinate
    system.  The dynamics use the derived target-relative state.
    """

    start_pos: Tuple[float, float, float]
    target_pos: Tuple[float, float, float]
    start_vel: Tuple[float, float, float]
    target_vel: Tuple[float, float, float]
    initial_fuel: float
    duration: float
    dt: float
    mean_motion: float
    safety_mode: str = "v1"
    goal_type: str = "APPROACH"

    @property
    def relative_pos(self) -> Tuple[float, float, float]:
        return tuple(a - b for a, b in zip(self.start_pos, self.target_pos))

    @property
    def relative_vel(self) -> Tuple[float, float, float]:
        return tuple(a - b for a, b in zip(self.start_vel, self.target_vel))

    @property
    def max_steps(self) -> int:
        return max(1, int(round(self.duration / self.dt)))


# Instantiated global config objects
PHYSICS_CONFIG = PhysicsConfig()
SAFETY_CONFIG = SafetyConfig()
ENV_CONFIG = EnvConfig()
