"""Safety-critical deterministic mission controller boundary."""
from .filtering import filter_action
from .schema import ActuatorCommand, ConstraintConfig, GoalSpec, Telemetry
from .telemetry import EventTelemetry
from .validation import validate_goal

class MissionController:
    def __init__(self, constraints: ConstraintConfig = ConstraintConfig(), telemetry: EventTelemetry | None = None):
        self.constraints = constraints
        self.telemetry = telemetry or EventTelemetry()

    def validate_goal(self, goal, telemetry, now_ns):
        return validate_goal(goal, self.constraints, telemetry, now_ns)

    def step(self, goal: GoalSpec, telemetry: Telemetry, commanded: ActuatorCommand, now_ns: int):
        next_id = f"OVR-{len(self.telemetry.events) + 1:06d}"
        result = filter_action(goal, telemetry, commanded, self.constraints, now_ns, next_id)
        event = self.telemetry.record(goal, telemetry, commanded, result)
        return result, event
