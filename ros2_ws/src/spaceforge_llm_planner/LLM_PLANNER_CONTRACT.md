# llm_planner contract v1

llm_planner is strategic only. It may emit versioned semantic goals and invoke read-only state/tool functions. It has no force, torque, acceleration, actuator, policy, or safety-filter API. Every semantic proposal is converted to mission_ctrl.GoalSpec and passed to MissionController before any operational result is accepted.

## Versioned proposal schema

A proposal is JSON with schema_version="1.0", proposal_id, goals (an ordered list), rationale, prompt_version, and metadata. Each goal has goal_type APPROACH, INSPECT, SAFE_HOLD, RETREAT, or ABORT plus semantic fields only: target_position_m, desired_distance_m, desired_closing_speed_mps, timeout_s, and optional inspect_region/reason. Unknown fields, direct force/torque fields, executable text, and unsupported schema versions are rejected.

## Tools

The only tools are get_spacecraft_state (read-only), approach_target, inspect_surface, return_to_safe_hold, and abort. Tool calls return structured observations or semantic GoalProposal values. Tools never publish actuator commands.

## Provider contract

Provider adapters return structured JSON or a JSON string. Each call has a timeout, bounded retry count, token limit, and cost limit. Only transient provider errors (timeouts, transport failures) retry; deterministic outcomes — token/cost overruns, malformed or versioned-out proposals, forbidden actuator fields, prompt-injection-like text, contradictory goals, and mission_ctrl rejections — fail fast to SAFE_HOLD without burning further provider calls. Prompt metadata includes prompt_version and schema_version. OfflineMockProvider is the default test backend and supports explicit per-response `(payload, input_tokens, output_tokens, cost_usd)` tuples for exercising limit paths without a network. OpenAICompatibleProvider is lazy and network-optional. GeminiProvider is lazy, stdlib-only, and network-optional: `GeminiProvider(api_key, model="gemini-3.8-flash")` performs no I/O at construction, enforces the call timeout, maps usage metadata to token counts, and raises retryable errors on HTTP 5xx/transport failures. Live calls run only with an explicit key (env `SPACEFORGE_GEMINI_API_KEY` in tests); overload responses are reported as saturated, never as passes.

## Priority and traces

Emergency fuel/stale telemetry handling remains mission_ctrl priority. Contradictory operational goals are rejected and converted to SAFE_HOLD. Every request, provider attempt, raw output, parse/validation result, mission_ctrl result, token/cost estimate, retry, and failure is written to a complete DecisionTrace. No network call is required by the default tests.
