"""Provider-neutral offline, OpenAI-compatible, and Gemini adapters."""
from dataclasses import dataclass
import json
import urllib.error
import urllib.request

@dataclass(frozen=True)
class ProviderResponse:
    raw: object
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    provider: str = "unknown"

class OfflineMockProvider:
    """Deterministic offline backend; never performs network I/O.

    Queued items are either raw payloads (dict/str, billed with the
    constructor's token/cost defaults) or explicit
    ``(payload, input_tokens, output_tokens, cost_usd)`` tuples for exercising
    token/cost-limit paths without a network.
    """

    def __init__(self, responses=None, error=None, input_tokens: int = 0,
                 output_tokens: int = 0, cost_usd: float = 0.0):
        self.responses = list(responses or [])
        self.error = error
        self.calls = 0
        self.default_tokens = (int(input_tokens), int(output_tokens))
        self.default_cost = float(cost_usd)

    def complete(self, prompt, tools, timeout_s, token_limit, cost_limit_usd):
        del prompt, tools, timeout_s, token_limit, cost_limit_usd
        self.calls += 1
        if self.error:
            raise self.error
        if not self.responses:
            raise RuntimeError("offline mock has no response")
        value = self.responses.pop(0)
        if isinstance(value, tuple):
            payload, input_tokens, output_tokens, cost_usd = value
        else:
            payload = value
            input_tokens, output_tokens = self.default_tokens
            cost_usd = self.default_cost
        return ProviderResponse(payload, int(input_tokens), int(output_tokens), float(cost_usd), provider="offline_mock")

class OpenAICompatibleProvider:
    """Lazy standard HTTP adapter; construction does not perform network I/O."""
    def __init__(self, endpoint, model, api_key=""):
        self.endpoint = endpoint; self.model = model; self.api_key = api_key

    def complete(self, prompt, tools, timeout_s, token_limit, cost_limit_usd):
        body = json.dumps({"model": self.model, "messages": [{"role": "user", "content": prompt}], "tools": tools, "max_tokens": token_limit}).encode()
        request = urllib.request.Request(self.endpoint, data=body, headers={"Content-Type": "application/json", **({"Authorization": "Bearer " + self.api_key} if self.api_key else {})})
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            payload = json.loads(response.read().decode())
        choice = payload["choices"][0]["message"].get("content", payload["choices"][0]["message"] .get("tool_calls"))
        usage = payload.get("usage", {})
        return ProviderResponse(choice, int(usage.get("prompt_tokens", 0)), int(usage.get("completion_tokens", 0)), 0.0, "openai_compatible")


class GeminiProvider:
    """Google Gemini REST adapter (stdlib only, no extra dependencies).

    Construction performs no network I/O. The API key travels only in the
    request URL and is never placed in traces, logs, or exceptions — planner
    traces record token counts and the provider name, never credentials.
    Read the key from the environment (GEMINI_API_KEY), never commit it.
    """

    ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models"

    def __init__(self, api_key="", model="gemini-3.8-flash"):
        self.api_key = api_key
        self.model = model
        self.calls = 0

    def _url(self):
        # Key stays in the URL (Google's convention); never logged or traced.
        return f"{self.ENDPOINT}/{self.model}:generateContent?key={self.api_key}"

    def complete(self, prompt, tools, timeout_s, token_limit, cost_limit_usd):
        declarations = [{"name": t.get("name", ""), "description": t.get("description", "")}
                        for t in (tools or []) if isinstance(t, dict)]
        body = json.dumps({
            "contents": [{"parts": [{"text": str(prompt)}]}],
            "tools": [{"functionDeclarations": declarations}] if declarations else [],
            "generationConfig": {"maxOutputTokens": int(token_limit)},
        }).encode()
        request = urllib.request.Request(
            self._url(), data=body, headers={"Content-Type": "application/json"})
        self.calls += 1
        try:
            with urllib.request.urlopen(request, timeout=timeout_s) as response:
                payload = json.loads(response.read().decode())
        except urllib.error.HTTPError as exc:
            try:
                detail = json.loads(exc.read().decode()).get("error", {}).get("message", "")
            except Exception:
                detail = ""
            raise RuntimeError(f"gemini HTTP {exc.code}: {detail[:200]}") from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            raise TimeoutError(f"gemini transport failure: {exc}") from exc
        candidates = payload.get("candidates", [])
        parts = ((candidates[0].get("content", {}) if candidates else {}).get("parts", []))
        text = "".join(str(part.get("text", "")) for part in parts if isinstance(part, dict))
        usage = payload.get("usageMetadata", {})
        return ProviderResponse(
            text,
            int(usage.get("promptTokenCount", 0)),
            int(usage.get("candidatesTokenCount", 0)),
            0.0,  # server-side pricing varies; planner enforces count/cost bounds instead
            "gemini",
        )
