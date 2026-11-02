"""Typed mission goals, constraints, telemetry, and actuator commands."""
from dataclasses import dataclass
from enum import Enum
from typing import Tuple

class GoalType(str, Enum):
    APPROACH = "APPROACH"
    INSPECT = "INSPECT"
    SAFE_HOLD = "SAFE_HOLD"
    RETREAT = "RETREAT"
    ABORT = "ABORT"

class ReasonCode(str, Enum):
    ACCEPTED = "ACCEPTED"
    INVALID_GOAL_TYPE = "INVALID_GOAL_TYPE"
    INVALID_VECTOR = "INVALID_VECTOR"
    INVALID_TIMEOUT = "INVALID_TIMEOUT"
    STALE_TELEMETRY = "STALE_TELEMETRY"
    FUEL_BELOW_FLOOR = "FUEL_BELOW_FLOOR"
    KEEP_OUT_VIOLATION = "KEEP_OUT_VIOLATION"
    CLOSING_SPEED_EXCEEDED = "CLOSING_SPEED_EXCEEDED"
    ACTUATOR_LIMIT_EXCEEDED = "ACTUATOR_LIMIT_EXCEEDED"
    SAFE_HOLD_ENFORCED = "SAFE_HOLD_ENFORCED"
    ABORT_ENFORCED = "ABORT_ENFORCED"
    COMMAND_CLIPPED = "COMMAND_CLIPPED"

@dataclass(frozen=True)
class ConstraintConfig:
    fuel_floor: float = 0.05
    keep_out_radius_m: float = 5.0
    max_closing_speed_mps: float = 0.2
    max_force_n: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    max_torque_nm: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    max_goal_timeout_s: float = 300.0
    stale_telemetry_timeout_s: float = 0.25
    # Vehicle model used to size braking and predict one control period ahead:
    # a braking force is never larger than what cancels the velocity within
    # one period, so the boundary cannot over-brake a vehicle into reverse.
    vehicle_mass_kg: float = 1.0
    control_period_s: float = 1.0

    def __post_init__(self):
        if not 0 <= self.fuel_floor <= 1 or min(self.keep_out_radius_m, self.max_closing_speed_mps, self.max_goal_timeout_s, self.stale_telemetry_timeout_s) <= 0: raise ValueError("invalid safety constraints")
        if any(v <= 0 for v in self.max_force_n + self.max_torque_nm): raise ValueError("actuator limits must be positive")
        if self.vehicle_mass_kg <= 0 or self.control_period_s <= 0: raise ValueError("vehicle mass and control period must be positive")

@dataclass(frozen=True)
class GoalSpec:
    goal_type: GoalType
    target_position_m: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    desired_distance_m: float = 5.0
    desired_closing_speed_mps: float = 0.0
    timeout_s: float = 60.0
    max_force_n: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    max_torque_nm: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    goal_id: str = ""

    def to_dict(self):
        from dataclasses import asdict
        data = asdict(self)
        data["goal_type"] = self.goal_type.value
        return data

    @classmethod
    def from_dict(cls, data) -> "GoalSpec":
        """Strict JSON-boundary constructor; raises ValueError on any malformed goal."""
        if not isinstance(data, dict):
            raise ValueError("goal must be a JSON object")
        try:
            goal_type = GoalType(str(data.get("goal_type", "")))
        except ValueError as exc:
            raise ValueError(f"unsupported goal_type: {data.get('goal_type')!r}") from exc

        def _vec(key, size):
            values = data.get(key, (0.0,) * size)
            try:
                items = tuple(float(v) for v in values)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{key} must contain {size} finite numbers") from exc
            if len(items) != size or any(v != v or abs(v) == float("inf") for v in items):
                raise ValueError(f"{key} must contain {size} finite numbers")
            return items

        def _number(key, default):
            try:
                value = float(data.get(key, default))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{key} must be a number") from exc
            if value != value or abs(value) == float("inf"):
                raise ValueError(f"{key} must be finite")
            return value

        return cls(
            goal_type=goal_type,
            target_position_m=_vec("target_position_m", 3),
            desired_distance_m=_number("desired_distance_m", 5.0),
            desired_closing_speed_mps=_number("desired_closing_speed_mps", 0.0),
            timeout_s=_number("timeout_s", 60.0),
            max_force_n=_vec("max_force_n", 3) if "max_force_n" in data else (1.0, 1.0, 1.0),
            max_torque_nm=_vec("max_torque_nm", 3) if "max_torque_nm" in data else (1.0, 1.0, 1.0),
            goal_id=str(data.get("goal_id", "")),
        )

@dataclass(frozen=True)
class Telemetry:
    position_m: Tuple[float, float, float]
    velocity_mps: Tuple[float, float, float]
    fuel_fraction: float
    stamp_ns: int
    frame_id: str = "target_lvlh"

@dataclass(frozen=True)
class ActuatorCommand:
    force_n: Tuple[float, float, float]
    torque_nm: Tuple[float, float, float]

@dataclass(frozen=True)
class ValidationResult:
    accepted: bool
    code: ReasonCode
    reason: str

@dataclass(frozen=True)
class FilterResult:
    applied: ActuatorCommand
    override: bool
    code: ReasonCode
    reason: str
    event_id: str
