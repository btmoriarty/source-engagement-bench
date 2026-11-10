"""
sim/harness/llm_client.py

LLM client for the sim harness.

Wraps the Anthropic API per the manifest's model block (provider, endpoint
profile, pinned name, sampling params). Both the real client and MockLLMClient
are callable as client(system_prompt, user_prompt) -> LLMResult.

OpenAICompatibleLLMClient runs the sim against any OpenAI-compatible endpoint
(Ollama, vLLM, llama.cpp) with no paid vendor API and no new dependency.

CONTRACT CHANGE: the runner stub declared llm_client(system, user) -> str.
This widens the return to LLMResult so the runner can capture token counts and
the API-returned model version. The runner reads result.text where a string was
previously expected.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel

from sim.harness.parser import LLMConfig


# ZDR (zero-data-retention) endpoint is required for IRB-bound work. The two
# profiles map to different base URLs at call time; pricing tables key off name.
_ENDPOINT_BASE_URLS = {
    "standard": "https://api.anthropic.com",
    "zdr": "https://api.anthropic.com",  # ZDR is gated by org config + header, not a separate host
}


class LLMResult(BaseModel):
    """
    Result of a single LLM call.

    text : the response text
    input_tokens / output_tokens : token counts (None when unknown, e.g. mock)
    model_version : exact model version string the API returned (None for mock)
    """
    text: str
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    model_version: Optional[str] = None


class AnthropicLLMClient:
    """
    Anthropic API wrapper honoring the manifest model block.

    Network calls are intentionally not exercised in tests; the SDK import and
    client construction are deferred to call time so the module imports without
    the anthropic package installed.
    """

    def __init__(self, model_config: LLMConfig) -> None:
        if model_config.endpoint_profile not in _ENDPOINT_BASE_URLS:
            raise ValueError(
                f"Unknown endpoint_profile: {model_config.endpoint_profile!r}; "
                "expected 'standard' or 'zdr'"
            )
        self.model_config = model_config
        self.endpoint_profile = model_config.endpoint_profile
        self.base_url = _ENDPOINT_BASE_URLS[model_config.endpoint_profile]
        self._client: object | None = None

    def _ensure_client(self) -> object:
        if self._client is None:
            # Imported lazily: keeps the module importable without the SDK and
            # keeps the network out of the test path.
            from anthropic import Anthropic  # type: ignore

            self._client = Anthropic(base_url=self.base_url)
        return self._client

    def __call__(self, system_prompt: str, user_prompt: str) -> LLMResult:
        client = self._ensure_client()
        cfg = self.model_config
        # The Anthropic API rejects setting both temperature and top_p in one call,
        # so we send temperature only. The manifest keeps top_p for the config record;
        # top_p 1.0 (the manifest default) is a no-op, so dropping it preserves behaviour.
        response = client.messages.create(  # type: ignore[attr-defined]
            model=cfg.name,
            max_tokens=cfg.max_tokens,
            temperature=cfg.temperature,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        text = "".join(
            block.text for block in response.content if getattr(block, "type", None) == "text"
        )
        usage = getattr(response, "usage", None)
        return LLMResult(
            text=text,
            input_tokens=getattr(usage, "input_tokens", None) if usage else None,
            output_tokens=getattr(usage, "output_tokens", None) if usage else None,
            model_version=getattr(response, "model", None),
        )


class OpenAICompatibleLLMClient:
    """
    Client for any OpenAI-compatible chat endpoint (Ollama, vLLM, llama.cpp,
    LM Studio). No paid vendor API: point base_url at a local server and the sim
    runs at zero marginal cost. Uses only the standard library (urllib), so the
    harness needs no new dependency.

    The sim has no human subjects (synthetic documents, LLM agents), so it is not
    ZDR/IRB-bound and is free to run against a local open-weight model. Using an
    open, locally-run model also strengthens the methods paper's reproducibility.

    base_url example (Ollama): http://localhost:11434/v1
    The manifest's model.name is sent as the model id (e.g. "qwen2.5:7b-instruct").
    Token counts come from the endpoint's usage block when present.
    """

    def __init__(self, base_url: str, model_config: LLMConfig, timeout_s: int = 300) -> None:
        self.base_url = base_url.rstrip("/")
        self.model_config = model_config
        self.timeout_s = timeout_s
        self.max_tokens_override = None  # per-call override (reasoning-budget experiment)

    def __call__(self, system_prompt: str, user_prompt: str) -> LLMResult:
        import json
        import urllib.request

        cfg = self.model_config
        payload = {
            "model": cfg.name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": cfg.temperature,
            "top_p": cfg.top_p,
            "max_tokens": (self.max_tokens_override or cfg.max_tokens),
            "stream": False,
        }
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        text = body["choices"][0]["message"]["content"]
        usage = body.get("usage") or {}
        return LLMResult(
            text=text,
            input_tokens=usage.get("prompt_tokens"),
            output_tokens=usage.get("completion_tokens"),
            model_version=body.get("model"),
        )


class MockLLMClient:
    """
    Test double with the same interface as AnthropicLLMClient. Returns a fixed,
    realistically shaped response and never touches the network. Token counts and
    model_version stay None, matching what a mock run honestly knows.
    """

    DEFAULT_RESPONSE = (
        "Decision: revise. The suggestion is plausible, but the source coverage "
        "looks thin; I would check the second citation before accepting."
    )

    def __init__(self, response_text: Optional[str] = None) -> None:
        self.response_text = response_text or self.DEFAULT_RESPONSE
        self.calls: list[tuple[str, str]] = []

    def __call__(self, system_prompt: str, user_prompt: str) -> LLMResult:
        self.calls.append((system_prompt, user_prompt))
        return LLMResult(text=self.response_text)
