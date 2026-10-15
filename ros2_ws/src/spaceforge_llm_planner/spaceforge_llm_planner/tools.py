"""Read-only state and semantic goal tools exposed to providers."""
from .schema import GoalProposal
from spaceforge_mission_ctrl.schema import GoalType

TOOL_NAMES = ("get_spacecraft_state", "approach_target", "inspect_surface", "return_to_safe_hold", "abort")

class PlannerTools:
    def __init__(self, state_reader): self.state_reader = state_reader
    def definitions(self):
        return [{"name": name, "description": "semantic tool; never an actuator publisher"} for name in TOOL_NAMES]
    def get_spacecraft_state(self): return self.state_reader()
    def approach_target(self, target_position_m=(0,0,0), desired_distance_m=5.0, desired_closing_speed_mps=0.0, timeout_s=60.0):
        return GoalProposal(GoalType.APPROACH, tuple(target_position_m), desired_distance_m, desired_closing_speed_mps, timeout_s)
    def inspect_surface(self, target_position_m=(0,0,0), region="unspecified", timeout_s=60.0):
        return GoalProposal(GoalType.INSPECT, tuple(target_position_m), 5.0, 0.0, timeout_s, region)
    def return_to_safe_hold(self, timeout_s=60.0): return GoalProposal(GoalType.SAFE_HOLD, timeout_s=timeout_s)
    def abort(self, reason="provider requested abort", timeout_s=60.0): return GoalProposal(GoalType.ABORT, timeout_s=timeout_s, reason=reason)
