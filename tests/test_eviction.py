"""
Tests for EvictionManager.
"""

import time

import pytest

from bounded_infinity.bounded_cache import BoundedCache
from bounded_infinity.eviction import EvictionManager, default_summarize


def _make_full_cache(capacity: int = 4) -> BoundedCache:
    """Helper: create a full cache with agents at various depths."""
    cache = BoundedCache(capacity=capacity)
    for i in range(capacity):
        cache.put(
            f"agent-{i}",
            {"value": i},
            parent_id=f"agent-{i-1}" if i > 0 else None,
            depth=i,
        )
    return cache


def test_needs_eviction() -> None:
    cache = _make_full_cache(4)
    em = EvictionManager(cache)
    assert em.needs_eviction()


def test_no_eviction_needed() -> None:
    cache = BoundedCache(capacity=4)
    cache.put("a1", {"x": 1})
    em = EvictionManager(cache)
    assert not em.needs_eviction()


def test_evict_single() -> None:
    cache = _make_full_cache(4)
    em = EvictionManager(cache, eviction_batch_size=1, idle_threshold=0.0)
    evicted = em.evict(count=1)
    assert len(evicted) == 1
    assert cache.size == 3


def test_evict_deepest_first() -> None:
    """The deepest agent (highest depth) should be evicted first."""
    cache = _make_full_cache(4)
    em = EvictionManager(cache, eviction_batch_size=1, idle_threshold=0.0)
    evicted = em.evict(count=1)
    # agent-3 is at depth 3, should be evicted first
    assert "agent-3" in evicted


def test_ensure_capacity() -> None:
    cache = _make_full_cache(4)
    em = EvictionManager(cache, idle_threshold=0.0)
    evicted = em.ensure_capacity(needed=2)
    assert cache.free >= 2
    assert len(evicted) >= 2


def test_hierarchical_summarization() -> None:
    """When a child is evicted, its state should be summarized into the parent."""
    cache = BoundedCache(capacity=3)
    cache.put("parent", {"goal": "research"}, depth=0)
    cache.put("child1", {"finding": "result A"}, parent_id="parent", depth=1)
    cache.put("child2", {"finding": "result B"}, parent_id="parent", depth=1)

    em = EvictionManager(cache, eviction_batch_size=1, idle_threshold=0.0)
    evicted = em.evict(count=1)
    assert len(evicted) == 1

    # Parent should now contain a summary of the evicted child
    parent_entry = cache.get("parent")
    assert parent_entry is not None
    assert "_summaries" in parent_entry.state
    assert parent_entry.state["_evicted_children_count"] >= 1


def test_stats_tracking() -> None:
    cache = _make_full_cache(4)
    em = EvictionManager(cache, idle_threshold=0.0)
    em.evict(count=2)
    assert em.stats.total_evictions == 2
    assert len(em.stats.eviction_history) == 1


# ------------------------------------------------------------------ #
# default_summarize function
# ------------------------------------------------------------------ #

def test_default_summarize_basic() -> None:
    parent = {"goal": "research"}
    children = [{"finding": "A"}, {"finding": "B"}]
    merged = default_summarize(parent, children)
    assert merged["goal"] == "research"
    assert len(merged["_summaries"]) == 2
    assert merged["_evicted_children_count"] == 2


def test_default_summarize_bounded() -> None:
    """Summaries should be bounded by _max_summaries."""
    parent = {"_max_summaries": 3}
    children = [{"v": i} for i in range(10)]
    merged = default_summarize(parent, children)
    assert len(merged["_summaries"]) == 3  # only last 3 kept


def test_default_summarize_accumulates() -> None:
    """Multiple rounds of summarization should accumulate."""
    state = {"goal": "X"}
    state = default_summarize(state, [{"r": 1}])
    assert state["_evicted_children_count"] == 1
    state = default_summarize(state, [{"r": 2}, {"r": 3}])
    assert state["_evicted_children_count"] == 3
    assert len(state["_summaries"]) == 3


# ------------------------------------------------------------------ #
# Hilbert locality clustering
# ------------------------------------------------------------------ #

def test_hilbert_cluster_basic() -> None:
    """_hilbert_cluster should select the tightest cluster."""
    from bounded_infinity.bounded_cache import CacheEntry
    entries = [
        CacheEntry(agent_id="a", state={}, hilbert_index=10, depth=1),
        CacheEntry(agent_id="b", state={}, hilbert_index=12, depth=1),
        CacheEntry(agent_id="c", state={}, hilbert_index=100, depth=1),
        CacheEntry(agent_id="d", state={}, hilbert_index=200, depth=1),
    ]
    result = EvictionManager._hilbert_cluster(entries, 2)
    assert len(result) == 2
    # a and b are the tightest cluster (span=2)
    ids = {e.agent_id for e in result}
    assert ids == {"a", "b"}


def test_hilbert_cluster_identity() -> None:
    """When count >= pool size, returns entire pool."""
    from bounded_infinity.bounded_cache import CacheEntry
    entries = [
        CacheEntry(agent_id="a", state={}, hilbert_index=10, depth=1),
        CacheEntry(agent_id="b", state={}, hilbert_index=20, depth=1),
    ]
    result = EvictionManager._hilbert_cluster(entries, 5)
    assert len(result) == 2


def test_preservation_ratio_tracking() -> None:
    """EvictionStats should track preservation ratios."""
    cache = BoundedCache(capacity=4)
    cache.put("parent", {"data": "info"}, parent_id=None, depth=0)
    cache.put("c1", {"result": "a"}, parent_id="parent", depth=1)
    cache.put("c2", {"result": "b"}, parent_id="parent", depth=1)
    cache.put("c3", {"result": "c"}, parent_id="parent", depth=1)

    em = EvictionManager(cache, idle_threshold=0.0)
    em.evict(count=2)

    assert len(em.stats.preservation_ratios) >= 1
    assert em.stats.avg_preservation_ratio > 0
