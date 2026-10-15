# SpaceForge llm_planner

This package is a strategic semantic-goal planner only. It has no thrust, torque, acceleration, actuator, policy, or safety-filter API. Every proposal is converted to a versioned GoalSpec and sent through mission_ctrl.

The default backend is OfflineMockProvider. OpenAICompatibleProvider is lazy and uses a standard HTTP-compatible endpoint only when explicitly selected. GeminiProvider (stdlib urllib, no extra dependencies, default model gemini-3.8-flash) calls the Generative Language REST API only when explicitly selected; the key travels in the request URL per Google's convention and never enters traces, logs, or exceptions. Read it from `GEMINI_API_KEY` — never commit it. Provider calls have timeout, retry, token, and cost bounds.

Every decision produces a complete DecisionTrace containing prompt/schema versions, provider attempts, raw/parsed output, failures, token/cost accounting, and mission_ctrl status.

Retry contract: transient provider errors retry up to max_retries; deterministic failures (bad schema, forbidden fields, injection-like text, contradictory goals, token/cost overruns, mission_ctrl rejections) fail fast to SAFE_HOLD on the first attempt. Emergency fuel/stale-telemetry pre-empts the provider entirely. Tests (all offline): valid routing, malformed/actuator outputs, contradictions, stale/fuel priority, provider failure with retry counting, injection rejection, ABORT flow, NaN-target mission_ctrl veto, schema-version mismatch, token/cost limits, and trace completeness.
