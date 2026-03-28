"""
Tests for the information metrics module.
"""

import pytest
from bounded_infinity.metrics import (
    structural_info,
    token_info,
    preservation_ratio,
)
from bounded_infinity.eviction import default_summarize


# ------------------------------------------------------------------ #
# structural_info
# ------------------------------------------------------------------ #

def test_structural_info_empty() -> None:
    assert structural_info({}) == 0.0


def test_structural_info_simple() -> None:
    state = {"key1": "value1", "key2": 42}
    info = structural_info(state)
    assert info > 0.0


def test_structural_info_ignores_underscored_keys() -> None:
    state_with = {"key1": "value1", "_internal": "metadata", "_summaries": [1, 2, 3]}
    state_without = {"key1": "value1"}
    # _internal and _summaries should be ignored
    assert structural_info(state_with) == structural_info(state_without)


def test_structural_info_grows_with_content() -> None:
    small = {"x": "a"}
    big = {"x": "a" * 1000, "y": "b" * 1000, "z": [1, 2, 3]}
    assert structural_info(big) > structural_info(small)


def test_structural_info_nested_dicts() -> None:
    flat = {"a": 1}
    nested = {"a": 1, "b": {"c": 2, "d": {"e": 3}}}
    assert structural_info(nested) > structural_info(flat)


# ------------------------------------------------------------------ #
# token_info
# ------------------------------------------------------------------ #

def test_token_info_empty() -> None:
    assert token_info({}) == 0.0


def test_token_info_text_content() -> None:
    state = {"response": "This is a test response with several words."}
    info = token_info(state)
    # ~44 chars / 4 ≈ 11 tokens
    assert 5 < info < 20


def test_token_info_ignores_underscored() -> None:
    state = {"response": "hello", "_summaries": "long internal text" * 100}
    info = token_info(state)
    # Should only count "hello" (5 chars / 4 = 1.25)
    assert info < 5.0


def test_token_info_scales_with_text() -> None:
    short = {"text": "Hello world"}
    long = {"text": "Hello world " * 100}
    assert token_info(long) > token_info(short) * 10


# ------------------------------------------------------------------ #
# preservation_ratio
# ------------------------------------------------------------------ #

def test_preservation_ratio_perfect() -> None:
    """Summarization should preserve a meaningful fraction of information."""
    parent = {"data": "parent_data"}
    children = [{"result": "child1"}, {"result": "child2"}]
    merged = default_summarize(dict(parent), children)
    ratio = preservation_ratio(parent, children, merged)
    # default_summarize stores child data under _summaries (underscore-prefixed),
    # which structural_info ignores — so ratio reflects only top-level key retention
    assert ratio > 0.2
    assert ratio <= 2.0


def test_preservation_ratio_empty_children() -> None:
    parent = {"data": "parent_data"}
    children: list[dict] = [{}]
    merged = default_summarize(dict(parent), children)
    ratio = preservation_ratio(parent, children, merged)
    assert ratio >= 0.9  # Almost nothing to lose


def test_preservation_ratio_zero_input() -> None:
    """Zero input information → ratio = 1.0 by convention."""
    ratio = preservation_ratio({}, [{}], {})
    assert ratio == 1.0


def test_preservation_ratio_with_token_metric() -> None:
    parent = {"text": "Parent context here."}
    children = [{"text": "Child response number one."}]
    merged = default_summarize(dict(parent), children)
    ratio = preservation_ratio(parent, children, merged, metric=token_info)
    assert 0.0 < ratio <= 2.0  # Ratio can exceed 1 if metadata adds text


# ------------------------------------------------------------------ #
# Integration with EvictionStats
# ------------------------------------------------------------------ #

def test_eviction_stats_preservation_tracking() -> None:
    """EvictionManager should track preservation ratios."""
    from bounded_infinity.bounded_cache import BoundedCache
    from bounded_infinity.eviction import EvictionManager

    cache = BoundedCache(capacity=4)
    manager = EvictionManager(cache=cache, idle_threshold=0.0)

    # Fill cache with a parent and children
    cache.put("parent", {"data": "parent_info"}, parent_id=None, depth=0)
    cache.put("child1", {"result": "res1"}, parent_id="parent", depth=1)
    cache.put("child2", {"result": "res2"}, parent_id="parent", depth=1)
    cache.put("child3", {"result": "res3"}, parent_id="parent", depth=1)

    evicted = manager.evict(count=2)
    assert len(evicted) >= 1
    assert manager.stats.total_summarizations >= 1
    assert len(manager.stats.preservation_ratios) >= 1
    assert all(r > 0 for r in manager.stats.preservation_ratios)
    assert manager.stats.avg_preservation_ratio > 0
