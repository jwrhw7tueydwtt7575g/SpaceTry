"""Centralized configuration parameters for SpaceForge MVP.

The immutable physics/safety/environment configs contain model parameters.  A
``MissionConfig`` contains per-run values and is deliberately passed into the
environment instead of being hidden in module globals.
"""

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Mapping, Optional, Tuple


@dataclass(frozen=True)
class PhysicsConfig:
    """Clohessy-Wiltshire physics parameters."""
    n: float = 0.00113  # Mean motion in rad/s (~400 km LEO orbit)
    dt: float = 1.0     # Simulation integration step size in seconds

    def __post_init__(self) -> None:
        if self.n <= 0.0 or self.dt <= 0.0:
            raise ValueError("PhysicsConfig.n and PhysicsConfig.dt must be positive")


@dataclass(frozen=True)
class SafetyConfig:
    """Mission safety interlock constraints."""
    safe_approach_dist: float = 5.0     # Safety threshold distance in meters
    fuel_floor: float = 0.05            # Minimum fuel fraction emergency threshold
    max_safe_closing_speed: float = 0.2 # Maximum safe speed inside 5m bubble (m/s)
    max_docking_speed: float = 0.1      # Maximum allowable relative speed at docking (<0.5m)
    # Proximity rule scope. Default preserves the legacy APPROACH-only behaviour
    # so existing callers/tests are unaffected. Set enforce_proximity_for_all_goals
    # to True (recommended for flight) so geometry alone triggers the speed gate
    # and a goal label can never disable protection.
    proximity_goals: Tuple[str, ...] = ("APPROACH",)
    enforce_proximity_for_all_goals: bool = False

    def __post_init__(self) -> None:
        if self.safe_approach_dist <= 0.0 or self.max_safe_closing_speed <= 0.0:
            raise ValueError("Safety distances and speeds must be positive")
        if not 0.0 <= self.fuel_floor <= 1.0:
            raise ValueError("SafetyConfig.fuel_floor must be in [0, 1]")
        object.__setattr__(self, "proximity_goals", tuple(str(v) for v in self.proximity_goals))
        if not isinstance(self.enforce_proximity_for_all_goals, bool):
            raise ValueError("SafetyConfig.enforce_proximity_for_all_goals must be bool")


@dataclass(frozen=True)
class EnvConfig:
    """Environment dimensions and reward parameters."""
    max_steps: int = 300
    pos_norm_scale: float = 20.0
    vel_norm_scale: float = 2.0
    initial_pos_bound: float = 20.0
    initial_vel_bound: float = 0.005
    docking_success_dist: float = 0.5
    fuel_rate_scale: float = 0.001  # Deprecated compatibility alias; use ActuatorConfig.

    # Reward shaping weights
    progress_reward_weight: float = 5.0
    velocity_penalty_weight: float = 0.5
    action_penalty_weight: float = 0.01
    divergence_penalty_weight: float = 1.0
    fuel_penalty_weight: float = 0.1
    collision_risk_penalty_weight: float = 5.0
    docking_bonus: float = 100.0
    work_volume_bound: float = 100.0

    def __post_init__(self) -> None:
        if self.max_steps < 1:
            raise ValueError("EnvConfig.max_steps must be at least one")
        if min(self.pos_norm_scale, self.vel_norm_scale, self.initial_pos_bound, self.initial_vel_bound, self.docking_success_dist, self.work_volume_bound) <= 0.0:
            raise ValueError("Environment scales and bounds must be positive")
        if min(self.progress_reward_weight, self.velocity_penalty_weight, self.action_penalty_weight, self.divergence_penalty_weight, self.fuel_penalty_weight, self.collision_risk_penalty_weight, self.docking_bonus) < 0.0:
            raise ValueError("Reward weights and docking_bonus must be non-negative")


@dataclass(frozen=True)
class ResetConfig:
    """Bounded stochastic reset distribution for training and evaluation."""
    distribution: str = "uniform"
    position_bound: float = 20.0
    velocity_bound: float = 0.005
    curriculum_stage: int = 0
    curriculum_position_bounds: Tuple[float, ...] = (5.0, 10.0, 20.0)
    curriculum_velocity_bounds: Tuple[float, ...] = (0.05, 0.1, 0.2)

    def __post_init__(self) -> None:
        object.__setattr__(self, "curriculum_position_bounds", tuple(float(v) for v in self.curriculum_position_bounds))
        object.__setattr__(self, "curriculum_velocity_bounds", tuple(float(v) for v in self.curriculum_velocity_bounds))
        if self.distribution not in {"uniform", "radial", "curriculum"}:
            raise ValueError("reset distribution must be uniform, radial, or curriculum")
        if self.position_bound <= 0.0 or self.velocity_bound <= 0.0:
            raise ValueError("reset position and velocity bounds must be positive")
        if self.curriculum_stage < 0 or not self.curriculum_position_bounds or not self.curriculum_velocity_bounds:
            raise ValueError("invalid curriculum reset configuration")
        if any(value <= 0.0 for value in self.curriculum_position_bounds + self.curriculum_velocity_bounds):
            raise ValueError("curriculum bounds must be positive")


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

    def __post_init__(self) -> None:
        for field_name in ("start_pos", "target_pos", "start_vel", "target_vel"):
            try:
                value = tuple(float(item) for item in getattr(self, field_name))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{field_name} must contain three finite numbers") from exc
            if len(value) != 3 or not all(item == item and abs(item) != float("inf") for item in value):
                raise ValueError(f"{field_name} must contain three finite numbers")
            object.__setattr__(self, field_name, value)
        if not 0.0 <= self.initial_fuel <= 1.0:
            raise ValueError("initial_fuel must be in [0, 1]")
        if self.duration <= 0.0 or self.dt <= 0.0 or self.mean_motion <= 0.0:
            raise ValueError("duration, dt, and mean_motion must be positive")
        if self.safety_mode not in {"off", "v1", "v2"}:
            raise ValueError("safety_mode must be one of: off, v1, v2")
        if not self.goal_type.strip():
            raise ValueError("goal_type must be non-empty")

    @property
    def relative_pos(self) -> Tuple[float, float, float]:
        return tuple(a - b for a, b in zip(self.start_pos, self.target_pos))

    @property
    def relative_vel(self) -> Tuple[float, float, float]:
        return tuple(a - b for a, b in zip(self.start_vel, self.target_vel))

    @property
    def max_steps(self) -> int:
        steps = self.duration / self.dt
        if abs(steps - round(steps)) > 1e-9:
            raise ValueError("duration must be an integer multiple of dt")
        return int(round(steps))


@dataclass(frozen=True)
class ActuatorConfig:
    """Action conversion and fuel-consumption parameters."""
    max_cmd_accel: float = 1.0
    fuel_rate_scale: float = 0.001

    def __post_init__(self) -> None:
        if self.max_cmd_accel <= 0.0 or self.fuel_rate_scale < 0.0:
            raise ValueError("Actuator limits must be positive and fuel rate non-negative")


@dataclass(frozen=True)
class ControllerConfig:
    """Controller selection and reproducibility options."""
    name: str = "glideslope"
    checkpoint: Optional[str] = None
    deterministic: bool = True

    def __post_init__(self) -> None:
        if self.name not in {"rl", "glideslope", "hot"}:
            raise ValueError("controller must be one of: rl, glideslope, hot")
        if self.name == "rl" and not self.checkpoint:
            raise ValueError("An RL controller requires a checkpoint path")


@dataclass(frozen=True)
class DomainConfig:
    """Domain randomization for sim-to-real robustness studies (Phase 6).

    Every default is the identity: mass fixed at nominal, no noise, fuel fixed.
    With defaults the environment behaves bit-for-bit as without randomization.
    All draws use the episode RNG, so identical seeds replay identical domains.

    Mass/inertia: per-episode mass ~ U[mass_low_kg, mass_high_kg]; commanded
      acceleration scales by mass_nominal_kg / mass (heavier = lazier).
      Per-axis actuator effectiveness ~ U[1-tol, 1+tol] models inertia
      asymmetry and thruster imbalance.
    Noise: additive Gaussian thrust disturbance (post-interlock, pre-clip) and
      camera/IMU observation noise. Observations are noisy; the true state used
      for docking judgment and rewards is never noisy.
    Fuel: initial fuel ~ U[fuel_low, fuel_high] on non-mission resets.
    """

    mass_nominal_kg: float = 1.0
    mass_low_kg: float = 1.0
    mass_high_kg: float = 1.0
    effectiveness_tol: float = 0.0
    thrust_noise_std: float = 0.0
    obs_pos_noise_std_m: float = 0.0
    obs_vel_noise_std_mps: float = 0.0
    fuel_low: float = 1.0
    fuel_high: float = 1.0

    def __post_init__(self) -> None:
        if not 0.0 < self.mass_nominal_kg:
            raise ValueError("mass_nominal_kg must be positive")
        if not 0.0 < self.mass_low_kg <= self.mass_high_kg:
            raise ValueError("require 0 < mass_low_kg <= mass_high_kg")
        if not 0.0 <= self.effectiveness_tol < 1.0:
            raise ValueError("effectiveness_tol must be in [0, 1)")
        for name in ("thrust_noise_std", "obs_pos_noise_std_m", "obs_vel_noise_std_mps"):
            if float(getattr(self, name)) < 0.0:
                raise ValueError(f"{name} must be non-negative")
        if not 0.0 <= self.fuel_low <= self.fuel_high <= 1.0:
            raise ValueError("require 0 <= fuel_low <= fuel_high <= 1")

    @property
    def is_identity(self) -> bool:
        """True when every draw is degenerate and observations are exact."""
        return (
            self.mass_low_kg == self.mass_high_kg == self.mass_nominal_kg
            and self.effectiveness_tol == 0.0
            and self.thrust_noise_std == 0.0
            and self.obs_pos_noise_std_m == 0.0
            and self.obs_vel_noise_std_mps == 0.0
            and self.fuel_low == self.fuel_high == 1.0
        )


@dataclass(frozen=True)
class FaultConfig:
    """Deterministic fault injection for robustness scenarios (Phase 6).

    Applied post-interlock in fixed order: stuck axis -> bias -> dropout ->
    thrust noise (domain) -> clip. Sensor freeze replays the previous
    observation; fuel leak debits extra fuel per step. All draws use the
    episode RNG. Default (all None/zero) injects nothing.
    """

    stuck_axis: Optional[int] = None
    stuck_value: float = 0.0
    bias: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    dropout_prob: float = 0.0
    sensor_freeze_prob: float = 0.0
    fuel_leak_rate: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "bias", tuple(float(v) for v in self.bias))
        if len(self.bias) != 3 or any(v != v or abs(v) == float("inf") for v in self.bias):
            raise ValueError("bias must be three finite numbers")
        if self.stuck_axis is not None and self.stuck_axis not in (0, 1, 2):
            raise ValueError("stuck_axis must be 0, 1, 2, or None")
        if not abs(self.stuck_value) <= 1.0:
            raise ValueError("stuck_value must be within [-1, 1]")
        for name in ("dropout_prob", "sensor_freeze_prob"):
            if not 0.0 <= float(getattr(self, name)) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if not 0.0 <= float(self.fuel_leak_rate):
            raise ValueError("fuel_leak_rate must be non-negative")

    @property
    def is_identity(self) -> bool:
        return (
            self.stuck_axis is None
            and self.bias == (0.0, 0.0, 0.0)
            and self.dropout_prob == 0.0
            and self.sensor_freeze_prob == 0.0
            and self.fuel_leak_rate == 0.0
        )


@dataclass(frozen=True)
class RunConfig:
    """Canonical immutable configuration for every runtime path."""
    mission: MissionConfig
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    actuator: ActuatorConfig = field(default_factory=ActuatorConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    environment: EnvConfig = field(default_factory=EnvConfig)
    controller: ControllerConfig = field(default_factory=ControllerConfig)
    reset: ResetConfig = field(default_factory=ResetConfig)
    domain: Optional[DomainConfig] = None
    faults: Optional[FaultConfig] = None
    safety_mode: Optional[str] = None
    seed: int = 42

    def __post_init__(self) -> None:
        mode = self.safety_mode or self.mission.safety_mode
        if mode not in {"off", "v1", "v2"}:
            raise ValueError("safety_mode must be one of: off, v1, v2")
        object.__setattr__(self, "safety_mode", mode)
        if abs(self.mission.dt - self.physics.dt) > 1e-12:
            raise ValueError("MissionConfig.dt must match PhysicsConfig.dt")
        if abs(self.mission.mean_motion - self.physics.n) > 1e-12:
            raise ValueError("MissionConfig.mean_motion must match PhysicsConfig.n")
        if self.mission.max_steps < 1:
            raise ValueError("Mission duration must contain at least one timestep")

    @property
    def max_steps(self) -> int:
        return self.mission.max_steps

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["schema_version"] = 1
        payload["mission"]["relative_pos"] = list(self.mission.relative_pos)
        payload["mission"]["relative_vel"] = list(self.mission.relative_vel)
        payload["relative_pos"] = list(self.mission.relative_pos)
        payload["relative_vel"] = list(self.mission.relative_vel)
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RunConfig":
        data = dict(payload)
        data.pop("schema_version", None)
        data.pop("relative_pos", None)
        data.pop("relative_vel", None)
        mission = dict(data.pop("mission"))
        mission.pop("relative_pos", None)
        mission.pop("relative_vel", None)
        domain_data = data.pop("domain", None)
        faults_data = data.pop("faults", None)
        return cls(
            mission=MissionConfig(**mission),
            physics=PhysicsConfig(**data.pop("physics", {})),
            actuator=ActuatorConfig(**data.pop("actuator", {})),
            safety=SafetyConfig(**data.pop("safety", {})),
            environment=EnvConfig(**data.pop("environment", {})),
            controller=ControllerConfig(**data.pop("controller", {})),
            reset=ResetConfig(**data.pop("reset", {})),
            domain=DomainConfig(**domain_data) if domain_data else None,
            faults=FaultConfig(**faults_data) if faults_data else None,
            safety_mode=data.pop("safety_mode", None),
            seed=int(data.pop("seed", 42)),
        )

    def write_json(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True))
        return path

    @classmethod
    def read_json(cls, path: Path) -> "RunConfig":
        return cls.from_dict(json.loads(path.read_text()))


# Instantiated global config objects
PHYSICS_CONFIG = PhysicsConfig()
ACTUATOR_CONFIG = ActuatorConfig()
SAFETY_CONFIG = SafetyConfig()
ENV_CONFIG = EnvConfig()
