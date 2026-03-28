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
    call_log: list[dict[str, Any]] = field(default_factory=list)

    def record(self, input_tokens: int, output_tokens: int) -> None:
        self.total_calls += 1
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens

    def summary(self) -> dict[str, Any]:
        return {
            "total_calls": self.total_calls,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "errors": self.errors,
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
    """Call OpenAI API."""
    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.3,
    )
    text = response.choices[0].message.content or ""
    usage = response.usage
    if usage:
        _tracker.record(usage.prompt_tokens, usage.completion_tokens)
    else:
        _tracker.record(len(prompt.split()), len(text.split()))
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
    "gemini": "gemini-2.0-flash",
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

    def _invoke(prompt: str) -> str:
        nonlocal call_count
        call_count += 1
        if call_count > max_calls:
            raise RuntimeError(
                f"LLM call limit exceeded ({max_calls}). "
                "Increase max_calls if intentional."
            )
        try:
            result = caller(prompt, resolved_model, max_tokens, resolved_key)
            # Small delay to respect rate limits
            time.sleep(0.5)
            return result
        except Exception as e:
            _tracker.errors += 1
            return f"[LLM Error: {type(e).__name__}]"

    return _invoke
