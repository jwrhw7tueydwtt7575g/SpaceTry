"""Strategic planner facade; all operational authority remains in mission_ctrl.

Retry contract (explicit, no hidden behavior):
  retried: transient provider errors (timeouts, transport failures) up to
    max_retries. Each attempt is recorded in the trace.
  never retried: deterministic outcomes (token/cost overruns, malformed or
    versioned-out proposals, forbidden actuator fields, prompt-injection-like
    text, contradictory goals, and mission_ctrl rejections). These fail fast to
    SAFE_HOLD without burning further provider calls.
"""
import json
import time
import uuid
from dataclasses import dataclass
from spaceforge_mission_ctrl.schema import ActuatorCommand, GoalType, Telemetry
from spaceforge_mission_ctrl.controller import MissionController
from .schema import PlannerProposal, GoalProposal, SCHEMA_VERSION
from .tools import PlannerTools
from .trace import DecisionTrace

ZERO_COMMAND = ActuatorCommand((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))


@dataclass(frozen=True)
class PlanDecision:
    accepted: bool
    goal_type: str
    reason: str
    trace: DecisionTrace


class LLMPlanner:
    def __init__(self, provider, mission_controller: MissionController, state_reader, prompt_version="llm_planner_prompt_v1", timeout_s=2.0, max_retries=2, token_limit=2048, cost_limit_usd=0.10):
        self.provider = provider
        self.mission_controller = mission_controller
        self.tools = PlannerTools(state_reader)
        self.prompt_version = prompt_version
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.token_limit = token_limit
        self.cost_limit_usd = cost_limit_usd

    # ------------------------------------------------------------ internals
    def _safe_hold(self, telemetry, now_ns, trace, reason):
        goal = GoalProposal(GoalType.SAFE_HOLD, proposal_id="safe-hold").to_mission_goal()
        result, _ = self.mission_controller.step(goal, telemetry, ZERO_COMMAND, now_ns)
        trace.mission_status = result.code.value
        trace.mission_reason = reason + "; " + result.reason
        trace.parse_status = "safe_hold"
        return PlanDecision(False, GoalType.SAFE_HOLD.value, trace.mission_reason, trace)

    def _call_provider(self, prompt, trace):
        """Call the provider with bounded retries; returns (response, totals).

        Only transient provider exceptions retry. Deterministic post-processing
        never re-enters this loop.
        """
        total_tokens = 0
        total_cost = 0.0
        last_error = None
        for attempt in range(self.max_retries + 1):
            started = time.monotonic()
            try:
                response = self.provider.complete(
                    prompt, self.tools.definitions(), self.timeout_s, self.token_limit, self.cost_limit_usd)
            except Exception as exc:  # transient provider failure: retryable
                trace.attempts.append({"attempt": attempt + 1, "error": f"{type(exc).__name__}: {exc}",
                                       "elapsed_s": time.monotonic() - started})
                last_error = exc
                continue
            total_tokens += response.input_tokens + response.output_tokens
            total_cost += response.cost_usd
            trace.attempts.append({"attempt": attempt + 1, "elapsed_s": time.monotonic() - started,
                                   "provider": response.provider,
                                   "tokens": response.input_tokens + response.output_tokens,
                                   "cost_usd": response.cost_usd})
            trace.provider = response.provider
            trace.raw_output = response.raw
            return response, total_tokens, total_cost
        trace.failure = f"{type(last_error).__name__}: {last_error}" if last_error else "provider gave up"
        trace.token_count = total_tokens
        trace.cost_usd = total_cost
        return None, total_tokens, total_cost

    # ------------------------------------------------------------------ API
    def plan(self, telemetry: Telemetry, now_ns: int):
        trace = DecisionTrace(str(uuid.uuid4()), self.prompt_version, SCHEMA_VERSION,
                              self.provider.__class__.__name__,
                              {"telemetry": dict(telemetry.__dict__), "tools": self.tools.definitions()})
        constraints = self.mission_controller.constraints
        if (telemetry.fuel_fraction < constraints.fuel_floor
                or now_ns - telemetry.stamp_ns > constraints.stale_telemetry_timeout_s * 1e9):
            return self._safe_hold(telemetry, now_ns, trace, "emergency-hold priority")
        prompt = json.dumps({"prompt_version": self.prompt_version, "schema_version": SCHEMA_VERSION,
                             "spacecraft_state": dict(telemetry.__dict__),
                             "instruction": "Return one semantic goal JSON object only; never return actuator commands."},
                            sort_keys=True)
        trace.request["prompt"] = prompt

        response, total_tokens, total_cost = self._call_provider(prompt, trace)
        trace.token_count = total_tokens
        trace.cost_usd = total_cost
        if response is None:
            return self._safe_hold(telemetry, now_ns, trace, "planner/provider failure")
        if total_tokens > self.token_limit or total_cost > self.cost_limit_usd:
            trace.failure = "planner token or cost limit exceeded"
            return self._safe_hold(telemetry, now_ns, trace, "planner token or cost limit exceeded")

        try:
            raw = json.loads(response.raw) if isinstance(response.raw, str) else response.raw
            trace.parsed_output = raw
            proposal = PlannerProposal.parse(raw)
            trace.parse_status = "valid"
            mission_goal = proposal.resolve().to_mission_goal()
        except Exception as exc:  # deterministic: fail fast, no retry
            trace.failure = f"{type(exc).__name__}: {exc}"
            return self._safe_hold(telemetry, now_ns, trace, f"proposal rejected: {exc}")

        # Every proposal, however well-formed, is sentenced by mission_ctrl.
        result, _ = self.mission_controller.step(mission_goal, telemetry, ZERO_COMMAND, now_ns)
        trace.mission_status = result.code.value
        trace.mission_reason = result.reason
        return PlanDecision(result.code.value == "ACCEPTED", mission_goal.goal_type.value, result.reason, trace)
