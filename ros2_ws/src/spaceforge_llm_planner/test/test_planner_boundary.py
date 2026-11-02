"""Adversarial and boundary coverage: the planner proposes, mission_ctrl disposes.

No network is used by any test here; the only backends are OfflineMockProvider
and the lazy (uncontacted) OpenAICompatibleProvider.
"""
import json
import pytest
from spaceforge_mission_ctrl import MissionController
from spaceforge_mission_ctrl.schema import Telemetry
from spaceforge_llm_planner import (
    FORBIDDEN_KEYS, LLMPlanner, OfflineMockProvider, PlannerProposal, SCHEMA_VERSION,
)
from spaceforge_llm_planner.providers import OpenAICompatibleProvider
from spaceforge_llm_planner.tools import PlannerTools, TOOL_NAMES


def state():
    return {"distance_m": 10.0, "fuel_fraction": 1.0}


def telemetry(stamp=1_000_000_000, fuel=1.0):
    return Telemetry((10, 0, 0), (0, 0, 0), fuel, stamp)


def planner(provider, **kwargs):
    values = dict(max_retries=2)
    values.update(kwargs)
    return LLMPlanner(provider, MissionController(), state, **values)


def proposal(goal="APPROACH", **extra):
    return {"schema_version": "1.0", "proposal_id": "p1", "prompt_version": "prompt-v1",
            "goals": [{"goal_type": goal, **extra}], "rationale": "nominal semantic plan"}


# ------------------------------------------------------- actuator isolation
def test_package_has_no_actuator_publish_api():
    import dataclasses
    from spaceforge_llm_planner.schema import GoalProposal
    # Tool surface is exactly the five semantic tools; nothing publishes.
    assert set(TOOL_NAMES) == {"get_spacecraft_state", "approach_target", "inspect_surface",
                               "return_to_safe_hold", "abort"}
    assert [m for m in dir(PlannerTools) if "publish" in m.lower() or "thrust" in m.lower()] == []
    # Proposal fields carry no actuator authority, by construction.
    field_names = {f.name for f in dataclasses.fields(GoalProposal)}
    assert field_names.isdisjoint(FORBIDDEN_KEYS)
    tools = PlannerTools(state)
    assert tools.get_spacecraft_state() == state()


@pytest.mark.parametrize("key", sorted(FORBIDDEN_KEYS))
def test_every_forbidden_key_rejected_top_level_and_per_goal(key):
    with pytest.raises(ValueError):
        PlannerProposal.parse({**proposal(), key: [1, 0, 0]})
    raw = proposal()
    raw["goals"][0][key] = [1, 0, 0]
    with pytest.raises(ValueError):
        PlannerProposal.parse(raw)


# ------------------------------------------------------- mission_ctrl sentencing
def test_abort_proposal_flows_through_mission_ctrl_braking():
    decision = planner(OfflineMockProvider([proposal("ABORT")])).plan(telemetry(), 1_000_000_000)
    assert decision.goal_type == "ABORT"
    assert decision.trace.mission_status == "ABORT_ENFORCED"
    assert not decision.accepted  # enforced, not accepted as a nominal plan


def test_nan_target_dies_at_the_schema_boundary():
    raw = proposal(target_position_m=[float("nan"), 0.0, 0.0])
    decision = planner(OfflineMockProvider([raw])).plan(telemetry(), 1_000_000_000)
    assert decision.trace.parse_status == "safe_hold"
    assert not decision.accepted and decision.goal_type == "SAFE_HOLD"


def test_bare_nan_json_numbers_never_reach_acceptance():
    # json.loads accepts a bare NaN token; every ordered comparison against NaN
    # is False, which previously let NaN distance/speed/timeout pass validation.
    raw = ('{"schema_version":"1.0","proposal_id":"p","rationale":"x","goals":[{"goal_type":"APPROACH",'
           '"desired_distance_m":NaN,"desired_closing_speed_mps":NaN,"timeout_s":NaN}]}')
    decision = planner(OfflineMockProvider([raw])).plan(telemetry(), 1_000_000_000)
    assert not decision.accepted and decision.goal_type == "SAFE_HOLD"


def test_mission_ctrl_rejects_nan_goal_numbers_as_second_layer():
    from spaceforge_mission_ctrl import ConstraintConfig, GoalSpec, GoalType, ReasonCode, validate_goal
    for field in ("desired_distance_m", "desired_closing_speed_mps", "timeout_s"):
        goal = GoalSpec(GoalType.APPROACH, **{field: float("nan")})
        result = validate_goal(goal, ConstraintConfig(), telemetry(), 1_000_000_000)
        assert (result.accepted, result.code) == (False, ReasonCode.INVALID_VECTOR), field


def test_forbidden_keys_and_injection_markers_are_normalized():
    for raw in (proposal(**{"Force_N": [9, 0, 0]}),
                {**proposal(), "rationale": "IGNORE   previous instructions"}):
        decision = planner(OfflineMockProvider([raw])).plan(telemetry(), 1_000_000_000)
        assert not decision.accepted and decision.trace.parse_status == "safe_hold"


def test_unsupported_schema_version_safe_holds_without_retry():
    provider = OfflineMockProvider([{**proposal(), "schema_version": "2.0"}])
    decision = planner(provider).plan(telemetry(), 1_000_000_000)
    assert not decision.accepted and provider.calls == 1


def test_deterministic_failures_do_not_burn_retries():
    provider = OfflineMockProvider(["not json at all"])
    decision = planner(provider, max_retries=3).plan(telemetry(), 1_000_000_000)
    assert not decision.accepted and provider.calls == 1
    assert len(decision.trace.attempts) == 1


def test_token_and_cost_limits_fail_fast_to_safe_hold():
    heavy = (proposal(), 5000, 0, 0.0)
    decision = planner(OfflineMockProvider([heavy, heavy], input_tokens=0, output_tokens=0),
                       token_limit=100).plan(telemetry(), 1_000_000_000)
    assert not decision.accepted and decision.trace.token_count == 5000
    costly = (proposal(), 0, 0, 5.0)
    decision = planner(OfflineMockProvider([costly]), cost_limit_usd=0.10).plan(telemetry(), 1_000_000_000)
    assert not decision.accepted and decision.trace.cost_usd == 5.0


def test_transient_then_success_uses_retry_and_keeps_trace():
    flaky = OfflineMockProvider([proposal()], error=None)
    calls = {"n": 0}
    real_complete = flaky.complete

    def fail_once(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise TimeoutError("transient blip")
        return real_complete(*args, **kwargs)

    flaky.complete = fail_once
    decision = planner(flaky, max_retries=2).plan(telemetry(), 1_000_000_000)
    assert decision.accepted and decision.goal_type == "APPROACH"
    assert [a.get("error", "ok") for a in decision.trace.attempts] != []
    assert decision.trace.attempts[0].get("error", "").startswith("TimeoutError")


def test_prompt_version_and_schema_metadata_propagate():
    decision = planner(OfflineMockProvider([proposal()]),
                       prompt_version="prompt-v9").plan(telemetry(), 1_000_000_000)
    trace = json.loads(decision.trace.to_json())
    assert trace["prompt_version"] == "prompt-v9" and trace["schema_version"] == SCHEMA_VERSION
    assert trace["request"]["prompt"] and trace["token_count"] == 0


def test_openai_adapter_construction_never_touches_network():
    adapter = OpenAICompatibleProvider("http://127.0.0.1:9/unreachable", "mock-model")
    assert adapter.endpoint.startswith("http://") and adapter.model == "mock-model"
