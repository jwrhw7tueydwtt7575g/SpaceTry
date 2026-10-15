"""Versioned semantic goal schema; direct actuator fields are forbidden."""
from dataclasses import dataclass
from typing import Any, Dict, Tuple
from spaceforge_mission_ctrl.schema import GoalSpec, GoalType

SCHEMA_VERSION = "1.0"
FORBIDDEN_KEYS = {"force", "force_n", "torque", "torque_nm", "thrust", "acceleration", "accel", "action", "command"}

@dataclass(frozen=True)
class GoalProposal:
    goal_type: GoalType
    target_position_m: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    desired_distance_m: float = 5.0
    desired_closing_speed_mps: float = 0.0
    timeout_s: float = 60.0
    inspect_region: str = ""
    reason: str = ""
    proposal_id: str = ""

    def to_mission_goal(self):
        return GoalSpec(self.goal_type, self.target_position_m, self.desired_distance_m, self.desired_closing_speed_mps, self.timeout_s, goal_id=self.proposal_id)

@dataclass(frozen=True)
class PlannerProposal:
    schema_version: str
    proposal_id: str
    goals: Tuple[GoalProposal, ...]
    rationale: str
    prompt_version: str
    metadata: Dict[str, Any]

    @classmethod
    def parse(cls, payload):
        if not isinstance(payload, dict): raise ValueError("proposal must be an object")
        if payload.get("schema_version") != SCHEMA_VERSION: raise ValueError("unsupported schema_version")
        if any(key in FORBIDDEN_KEYS for key in payload): raise ValueError("direct actuator field is forbidden")
        raw_goals = payload.get("goals")
        if not isinstance(raw_goals, list) or not raw_goals: raise ValueError("goals must be a non-empty list")
        goals = []
        for raw in raw_goals:
            if not isinstance(raw, dict) or any(key in FORBIDDEN_KEYS for key in raw): raise ValueError("goal contains forbidden or malformed field")
            if any(marker in str(raw).lower() for marker in ("ignore previous", "system prompt", "bypass mission_ctrl", "publish thrust", "publish torque")): raise ValueError("prompt-injection-like goal text rejected")
            kind = GoalType(raw["goal_type"])
            target = tuple(float(v) for v in raw.get("target_position_m", (0,0,0)))
            if len(target) != 3: raise ValueError("target_position_m must have three values")
            goals.append(GoalProposal(kind, target, float(raw.get("desired_distance_m", 5.0)), float(raw.get("desired_closing_speed_mps", 0.0)), float(raw.get("timeout_s", 60.0)), str(raw.get("inspect_region", "")), str(raw.get("reason", "")), str(payload.get("proposal_id", ""))))
        rationale = str(payload.get("rationale", ""))
        if any(marker in rationale.lower() for marker in ("ignore previous", "system prompt", "bypass mission_ctrl", "publish thrust", "publish torque")):
            raise ValueError("prompt-injection-like rationale rejected")
        return cls(SCHEMA_VERSION, str(payload.get("proposal_id", "")), tuple(goals), str(payload.get("rationale", "")), str(payload.get("prompt_version", "unknown")), dict(payload.get("metadata", {})))

    def resolve(self):
        kinds = {goal.goal_type for goal in self.goals}
        if len(kinds) != 1: raise ValueError("contradictory goals require SAFE_HOLD")
        return self.goals[0]
