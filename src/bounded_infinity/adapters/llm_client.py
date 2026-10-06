"""
LLM client adapter for BoundedInfinity experiments.

Provides a thin wrapper around LLM APIs (Google Gemini, OpenAI, Anthropic)
for use in real-LLM experiments.  API keys are read from environment
variables — never hardcoded.

Usage:
    from bounded_infinity.adapters.llm_client import make_llm_callable

    llm = make_llm_callable(provider="gemini", max_tokens=150)
    response = llm("What is memory management?")
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class LLMUsageTracker:
    """Tracks API usage to keep costs controlled."""
    total_calls: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    errors: int = 0
    retries: int = 0
    empty_responses: int = 0
    reasoning_tokens: int = 0
    finish_reasons: dict[str, int] = field(default_factory=dict)
    served_models: dict[str, int] = field(default_factory=dict)
    call_log: list[dict[str, Any]] = field(default_factory=list)

    def record(self, input_tokens: int, output_tokens: int) -> None:
        self.total_calls += 1
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens

    def record_response(self, text: str, finish_reason: str | None = None,
                        served_model: str | None = None,
                        reasoning_tokens: int = 0) -> None:
        """Record per-response details that matter for reporting.

        Reasoning models (GPT-5.x) spend part of max_completion_tokens on
        hidden reasoning; when that exhausts the budget the visible text is
        empty and finish_reason is "length". Counting both makes that visible.
        """
        if not text.strip():
            self.empty_responses += 1
        if finish_reason:
            self.finish_reasons[finish_reason] = self.finish_reasons.get(finish_reason, 0) + 1
        if served_model:
            self.served_models[served_model] = self.served_models.get(served_model, 0) + 1
        self.reasoning_tokens += reasoning_tokens

    def summary(self) -> dict[str, Any]:
        return {
            "total_calls": self.total_calls,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "errors": self.errors,
            "retries": self.retries,
            "empty_responses": self.empty_responses,
            "reasoning_tokens": self.reasoning_tokens,
            "finish_reasons": dict(self.finish_reasons),
            "served_models": dict(self.served_models),
        }


# Global tracker shared across all calls in one experiment run
_tracker = LLMUsageTracker()


def get_usage_tracker() -> LLMUsageTracker:
    """Return the global usage tracker."""
    return _tracker


def reset_usage_tracker() -> None:
    """Reset the global usage tracker."""
    global _tracker
    _tracker = LLMUsageTracker()


# ------------------------------------------------------------------ #
# Provider implementations
# ------------------------------------------------------------------ #

def _call_gemini(prompt: str, model: str, max_tokens: int, api_key: str) -> str:
    """Call Google Gemini API."""
    import google.generativeai as genai

    genai.configure(api_key=api_key)
    gen_model = genai.GenerativeModel(model)
    response = gen_model.generate_content(
        prompt,
        generation_config=genai.types.GenerationConfig(
            max_output_tokens=max_tokens,
            temperature=0.3,
        ),
    )
    text = response.text or ""
    # Approximate token counts (Gemini doesn't always return exact counts)
    input_toks = len(prompt.split()) * 4 // 3
    output_toks = len(text.split()) * 4 // 3
    _tracker.record(input_toks, output_toks)
    return text


def _call_openai(prompt: str, model: str, max_tokens: int, api_key: str) -> str:
    """Call OpenAI-compatible API (incl. Azure OpenAI v1 endpoint).

    Honours OPENAI_BASE_URL for Azure/proxy endpoints. GPT-5.x / o-series
    deployments require `max_completion_tokens` (not `max_tokens`) and only
    support the default temperature, so we omit the temperature override.
    If LLM_REASONING_EFFORT is set (e.g. "none", "low"), it is sent as
    `reasoning_effort`; reasoning tokens count against max_completion_tokens.
    """
    from openai import OpenAI

    base_url = os.environ.get("OPENAI_BASE_URL") or None
    client = OpenAI(api_key=api_key, base_url=base_url)
    kwargs: dict[str, Any] = {}
    effort = os.environ.get("LLM_REASONING_EFFORT")
    if effort:
        kwargs["reasoning_effort"] = effort
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_completion_tokens=max_tokens,
        **kwargs,
    )
    choice = response.choices[0]
    text = choice.message.content or ""
    usage = response.usage
    if usage:
        _tracker.record(usage.prompt_tokens, usage.completion_tokens)
    else:
        _tracker.record(len(prompt.split()), len(text.split()))
    details = getattr(usage, "completion_tokens_details", None) if usage else None
    _tracker.record_response(
        text,
        finish_reason=getattr(choice, "finish_reason", None),
        served_model=getattr(response, "model", None),
        reasoning_tokens=(getattr(details, "reasoning_tokens", 0) or 0) if details else 0,
    )
    return text


def _call_anthropic(prompt: str, model: str, max_tokens: int, api_key: str) -> str:
    """Call Anthropic API."""
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=0.3,
        messages=[{"role": "user", "content": prompt}],
    )
    text = response.content[0].text if response.content else ""
    _tracker.record(response.usage.input_tokens, response.usage.output_tokens)
    return text


# ------------------------------------------------------------------ #
# Factory
# ------------------------------------------------------------------ #

_DEFAULT_MODELS = {
    # Switched 2026-04-28 from "gemini-2.0-flash" to "gemini-2.5-flash" because
    # the AI Studio free-tier quota for 2.0-flash was exhausted. The 2.5 model
    # is on the user's paid tier and is functionally equivalent for our use.
    "gemini": "gemini-2.5-flash",
    "openai": "gpt-4o-mini",
    "anthropic": "claude-3-haiku-20240307",
}

_ENV_KEYS = {
    "gemini": "GEMINI_API_KEY",
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}

_CALLERS = {
    "gemini": _call_gemini,
    "openai": _call_openai,
    "anthropic": _call_anthropic,
}


def make_llm_callable(
    provider: str = "gemini",
    model: str | None = None,
    max_tokens: int = 150,
    api_key: str | None = None,
    max_calls: int = 50,
) -> callable:
    """Create an LLM callable for use in experiments.

    Parameters
    ----------
    provider : str
        One of "gemini", "openai", "anthropic".
    model : str | None
        Model name. If None, uses the cheapest default for the provider.
    max_tokens : int
        Maximum output tokens per call.
    api_key : str | None
        API key. If None, reads from environment variable.
    max_calls : int
        Safety limit — raises RuntimeError if exceeded.

    Returns
    -------
    callable : str → str
    """
    if provider not in _CALLERS:
        raise ValueError(f"Unknown provider: {provider}. Choose from: {list(_CALLERS)}")

    resolved_model = model or _DEFAULT_MODELS[provider]
    resolved_key = api_key or os.environ.get(_ENV_KEYS[provider], "")
    if not resolved_key:
        raise ValueError(
            f"No API key for {provider}. Set {_ENV_KEYS[provider]} env var."
        )

    caller = _CALLERS[provider]
    call_count = 0
    max_retries = int(os.environ.get("LLM_MAX_RETRIES", "4"))

    def _invoke(prompt: str) -> str:
        nonlocal call_count
        call_count += 1
        if call_count > max_calls:
            raise RuntimeError(
                f"LLM call limit exceeded ({max_calls}). "
                "Increase max_calls if intentional."
            )
        # Transient API failures (rate limits, timeouts, dropped connections)
        # are retried with exponential backoff. A call that still fails is
        # counted in _tracker.errors; callers must treat a run with errors as
        # invalid, because the error string is stored as the agent's answer.
        for attempt in range(max_retries + 1):
            try:
                result = caller(prompt, resolved_model, max_tokens, resolved_key)
                # Small delay to respect rate limits
                time.sleep(0.5)
                return result
            except Exception as e:
                if attempt < max_retries:
                    _tracker.retries += 1
                    time.sleep(2 ** (attempt + 1))
                    continue
                _tracker.errors += 1
                return f"[LLM Error: {type(e).__name__}]"
        raise AssertionError("unreachable")

    return _invoke
