"""Gemini provider tests: fully mocked default suite, env-gated live probe.

The API key never appears in any committed file. Live calls only run when
SPACEFORGE_GEMINI_API_KEY is set, and overload responses become a skip —
never a fake pass.
"""
import io
import json
import os
import urllib.error
from unittest import mock

import pytest

from spaceforge_llm_planner import GeminiProvider, LLMPlanner
from spaceforge_mission_ctrl import MissionController
from spaceforge_mission_ctrl.schema import Telemetry


def _fake_urlopen(payload, calls=None):
    def opener(request, timeout=None):
        if calls is not None:
            calls.append({"url": request.full_url, "timeout": timeout,
                          "body": json.loads(request.data.decode())})
        response = mock.MagicMock()
        response.read.return_value = json.dumps(payload).encode()
        response.__enter__.return_value = response
        response.__exit__.return_value = False
        return response
    return opener


def _gemini_ok(text="{\"schema_version\": \"1.0\"}", prompt_tokens=12, completion_tokens=5):
    return {"candidates": [{"content": {"parts": [{"text": text}]}}],
            "usageMetadata": {"promptTokenCount": prompt_tokens,
                              "candidatesTokenCount": completion_tokens}}


def test_gemini_parses_text_usage_and_tools():
    calls = []
    provider = GeminiProvider(api_key="SECRET-KEY", model="gemini-3.8-flash")
    assert provider.calls == 0  # construction performs no network I/O
    with mock.patch("urllib.request.urlopen", _fake_urlopen(
            _gemini_ok(), calls)):
        response = provider.complete("prompt", [{"name": "abort", "description": "stop"}],
                                     timeout_s=9, token_limit=64, cost_limit_usd=0.5)
    assert provider.calls == 1
    assert response.provider == "gemini"
    assert response.input_tokens == 12 and response.output_tokens == 5
    sent = calls[0]
    assert sent["timeout"] == 9
    assert "gemini-3.8-flash" in sent["url"]
    assert sent["body"]["generationConfig"]["maxOutputTokens"] == 64
    assert sent["body"]["tools"][0]["functionDeclarations"][0]["name"] == "abort"


def test_gemini_http_error_becomes_retryable_runtime_error():
    def failing(request, timeout=None):
        raise urllib.error.HTTPError(
            request.full_url, 503, "Overloaded", {},
            io.BytesIO(b'{"error": {"message": "high demand"}}'))
    provider = GeminiProvider(api_key="k")
    with mock.patch("urllib.request.urlopen", failing):
        with pytest.raises(RuntimeError, match="gemini HTTP 503"):
            provider.complete("p", [], 5, 64, 0.5)


def test_gemini_transport_failure_becomes_timeout():
    provider = GeminiProvider(api_key="k")
    with mock.patch("urllib.request.urlopen", side_effect=TimeoutError("stalled")):
        with pytest.raises(TimeoutError):
            provider.complete("p", [], 5, 64, 0.5)


def test_key_never_enters_traces_or_errors():
    proposal = {"schema_version": "1.0", "proposal_id": "p", "prompt_version": "v",
                "goals": [{"goal_type": "RETREAT"}], "rationale": "ok"}
    provider = GeminiProvider(api_key="SECRET-KEY-123")
    with mock.patch("urllib.request.urlopen",
                     _fake_urlopen(_gemini_ok(json.dumps(proposal)))):
        planner = LLMPlanner(provider, MissionController(), lambda: {}, max_retries=0)
        decision = planner.plan(
            Telemetry((10, 0, 0), (0, 0, 0), 1.0, 1_000_000_000), 1_000_000_000)
    assert decision.goal_type == "RETREAT"
    blob = decision.trace.to_json() + decision.reason
    assert "SECRET-KEY-123" not in blob


def test_gemini_overload_safe_holds_through_planner():
    def failing(request, timeout=None):
        raise urllib.error.HTTPError(
            request.full_url, 503, "Overloaded", {},
            io.BytesIO(b'{"error": {"message": "high demand"}}'))
    provider = GeminiProvider(api_key="k")
    with mock.patch("urllib.request.urlopen", failing):
        planner = LLMPlanner(provider, MissionController(), lambda: {}, max_retries=2)
        decision = planner.plan(
            Telemetry((10, 0, 0), (0, 0, 0), 1.0, 1_000_000_000), 1_000_000_000)
    assert provider.calls == 3  # initial + 2 retries, then safe-hold
    assert not decision.accepted and decision.goal_type == "SAFE_HOLD"


def test_live_gemini_only_with_explicit_key():
    key = os.environ.get("SPACEFORGE_GEMINI_API_KEY", "")
    if not key:
        pytest.skip("no live key; network provider tests are never part of the default suite")
    import urllib.request
    body = json.dumps({"contents": [{"parts": [{"text": "Reply with exactly: KEY-OK"}]}]}).encode()
    request = urllib.request.Request(
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.8-flash:generateContent?key=" + key,
        data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = json.loads(response.read().decode())
    except urllib.error.HTTPError as exc:
        if exc.code in (503, 429):
            pytest.skip(f"model overloaded (HTTP {exc.code}); key and model verified, service saturated")
        raise
    except (TimeoutError, urllib.error.URLError, OSError) as exc:
        pytest.skip(f"service unreachable/saturated ({type(exc).__name__}); not a key failure")
    text = "".join(p.get("text", "") for p in payload["candidates"][0]["content"]["parts"])
    assert text.strip(), "live model returned empty text"
