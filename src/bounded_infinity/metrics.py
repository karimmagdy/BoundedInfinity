"""
Information metrics for measuring state content and eviction quality.

Defines the information metric I(s) used to quantify how much "content"
an agent state carries, and to measure the preservation ratio ρ(φ) of the
hierarchical summarization function φ.

For the formal framework:
  - I(s) quantifies the information content of a state s ∈ S
  - The preservation ratio ρ = I(φ(s_p, [s_c₁, …])) / (I(s_p) + ΣI(s_cᵢ))
  - Theorem 3 requires ρ ≥ 1 − ε for configurable ε

Two concrete metrics are provided:
  1. structural_info — counts keys and payload size (for arbitrary dict states)
  2. token_info — counts tokens (for LLM agent states with text content)
"""

from __future__ import annotations

import sys
from typing import Any, Callable


# ------------------------------------------------------------------ #
# Information metric type
# ------------------------------------------------------------------ #

InformationMetric = Callable[[dict[str, Any]], float]


# ------------------------------------------------------------------ #
# Structural information metric
# ------------------------------------------------------------------ #

def _deep_size(obj: Any, _seen: set[int] | None = None) -> int:
    """Approximate deep byte size of an object (no double-counting)."""
    if _seen is None:
        _seen = set()
    obj_id = id(obj)
    if obj_id in _seen:
        return 0
    _seen.add(obj_id)

    size = sys.getsizeof(obj)
    if isinstance(obj, dict):
        for k, v in obj.items():
            size += _deep_size(k, _seen) + _deep_size(v, _seen)
    elif isinstance(obj, (list, tuple, set, frozenset)):
        for item in obj:
            size += _deep_size(item, _seen)
    return size


def structural_info(state: dict[str, Any]) -> float:
    """Structural information metric: key count + deep payload size.

    I(s) = |keys(s)| + deep_sizeof(s) / 1024

    Combines structural complexity (number of keys, including nested)
    with payload size.  The /1024 normalization keeps the two components
    on a comparable scale for typical agent states.

    Ignores keys starting with '_' (internal metadata) so that
    eviction bookkeeping (_summaries, _evicted_children_count, etc.)
    does not inflate the parent's measured information.
    """
    def _count_keys(d: dict[str, Any]) -> int:
        count = 0
        for k, v in d.items():
            if isinstance(k, str) and k.startswith("_"):
                continue
            count += 1
            if isinstance(v, dict):
                count += _count_keys(v)
            elif isinstance(v, list):
                for item in v:
                    if isinstance(item, dict):
                        count += _count_keys(item)
        return count

    def _payload_size(d: dict[str, Any]) -> int:
        total = 0
        for k, v in d.items():
            if isinstance(k, str) and k.startswith("_"):
                continue
            total += sys.getsizeof(v)
            if isinstance(v, str):
                total += len(v)
            elif isinstance(v, (list, tuple)):
                for item in v:
                    total += sys.getsizeof(item)
                    if isinstance(item, str):
                        total += len(item)
                    elif isinstance(item, dict):
                        total += _payload_size(item)
            elif isinstance(v, dict):
                total += _payload_size(v)
        return total

    keys = _count_keys(state)
    payload = _payload_size(state)
    return float(keys) + payload / 1024.0


# ------------------------------------------------------------------ #
# Token-based information metric (for LLM agent states)
# ------------------------------------------------------------------ #

def _count_text_chars(obj: Any) -> int:
    """Recursively count total characters in all string values."""
    if isinstance(obj, str):
        return len(obj)
    if isinstance(obj, dict):
        return sum(_count_text_chars(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return sum(_count_text_chars(item) for item in obj)
    return 0


def token_info(state: dict[str, Any]) -> float:
    """Token-based information metric: approximate token count.

    I(s) ≈ total_characters / 4  (standard approximation: ~4 chars/token)

    For LLM agent states where the primary payload is text (prompts,
    responses, reasoning traces).  Ignores '_'-prefixed keys.
    """
    filtered = {k: v for k, v in state.items()
                if not (isinstance(k, str) and k.startswith("_"))}
    chars = _count_text_chars(filtered)
    return chars / 4.0


# ------------------------------------------------------------------ #
# Preservation ratio computation
# ------------------------------------------------------------------ #

def preservation_ratio(
    parent_before: dict[str, Any],
    child_states: list[dict[str, Any]],
    parent_after: dict[str, Any],
    metric: InformationMetric = structural_info,
) -> float:
    """Compute the information preservation ratio ρ of a summarization.

    ρ = I(parent_after) / (I(parent_before) + Σ I(child_i))

    A ratio of 1.0 means perfect preservation; lower means information
    was lost.  The formal bound requires ρ ≥ 1 − ε.

    Returns 1.0 if the denominator is zero (no information to preserve).
    """
    info_before = metric(parent_before)
    info_children = sum(metric(c) for c in child_states)
    total_input = info_before + info_children

    if total_input == 0.0:
        return 1.0

    info_after = metric(parent_after)
    return info_after / total_input
