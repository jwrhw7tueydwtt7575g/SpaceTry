"""Override event telemetry for auditability."""
from dataclasses import asdict, dataclass
import json
from .schema import ActuatorCommand, FilterResult, GoalSpec, Telemetry

@dataclass(frozen=True)
class OverrideEvent:
    event_id: str
    stamp_ns: int
    goal_type: str
    reason_code: str
    reason: str
    commanded_force_n: tuple
    applied_force_n: tuple
    commanded_torque_nm: tuple
    applied_torque_nm: tuple

    def to_dict(self): return asdict(self)
    def to_json(self): return json.dumps(self.to_dict(), sort_keys=True)

class EventTelemetry:
    def __init__(self): self.events = []; self._counter = 0
    def record(self, goal: GoalSpec, telemetry: Telemetry, commanded: ActuatorCommand, result: FilterResult):
        if not result.override: return None
        self._counter += 1; event_id = result.event_id or f"OVR-{self._counter:06d}"
        event = OverrideEvent(event_id, telemetry.stamp_ns, goal.goal_type.value, result.code.value, result.reason, tuple(commanded.force_n), tuple(result.applied.force_n), tuple(commanded.torque_nm), tuple(result.applied.torque_nm))
        self.events.append(event); return event
