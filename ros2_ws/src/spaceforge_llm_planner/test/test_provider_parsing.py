"""Offline tests for provider response parsing and cost accounting (no network)."""
import io
import json
import unittest.mock as mock

from spaceforge_llm_planner.providers import GeminiProvider, OpenAICompatibleProvider


class _Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _serve(payload):
    return mock.patch("urllib.request.urlopen", return_value=_Response(json.dumps(payload).encode()))


def test_openai_tool_call_reply_is_parsed_from_arguments():
    proposal = {"schema_version": "1.0", "goals": [{"goal_type": "SAFE_HOLD"}]}
    reply = {"choices": [{"message": {"role": "assistant", "content": None, "tool_calls": [
        {"type": "function", "function": {"name": "propose_goal", "arguments": json.dumps(proposal)}}]}}],
        "usage": {"prompt_tokens": 1000, "completion_tokens": 500}}
    with _serve(reply):
        out = OpenAICompatibleProvider("http://localhost", "m", input_usd_per_1k=0.002,
                                       output_usd_per_1k=0.008).complete("p", [], 1, 100, 1)
    assert out.raw == proposal
    assert abs(out.cost_usd - (1000 * 0.002 + 500 * 0.008) / 1000) < 1e-12


def test_openai_plain_content_reply_still_works():
    reply = {"choices": [{"message": {"content": "{\"x\": 1}"}}], "usage": {}}
    with _serve(reply):
        out = OpenAICompatibleProvider("http://localhost", "m").complete("p", [], 1, 100, 1)
    assert out.raw == "{\"x\": 1}" and out.cost_usd == 0.0


def test_gemini_cost_uses_configured_prices():
    reply = {"candidates": [{"content": {"parts": [{"text": "{}"}]}}],
             "usageMetadata": {"promptTokenCount": 2000, "candidatesTokenCount": 1000}}
    with _serve(reply):
        out = GeminiProvider(api_key="k", input_usd_per_1k=0.001, output_usd_per_1k=0.004).complete("p", [], 1, 100, 1)
    assert abs(out.cost_usd - 0.006) < 1e-12
