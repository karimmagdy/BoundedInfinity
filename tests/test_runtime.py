"""
Tests for BoundedInfinityRuntime — integration tests of the full system.
"""

import pytest
from typing import Any, Callable

from bounded_infinity.runtime import BoundedInfinityRuntime


def _counting_executor(agent_id: str, task: dict[str, Any],
                       state: dict[str, Any],
                       spawn: Callable[[dict[str, Any]], str]) -> dict[str, Any]:
    """Simple executor that records execution and optionally spawns children."""
    result = dict(state)
    result["executed"] = True
    result["agent_id"] = agent_id

    # If task says to spawn children, do so
    n_children = task.get("spawn_children", 0)
    child_ids = []
    for i in range(n_children):
        cid = spawn({"child_index": i, "spawn_children": 0})
        child_ids.append(cid)
    result["children_spawned"] = child_ids
    return result


# ------------------------------------------------------------------ #
# Basic lifecycle
# ------------------------------------------------------------------ #

def test_spawn_root() -> None:
    rt = BoundedInfinityRuntime(cache_size=16)
    root_id = rt.spawn(task={"goal": "test"})
    assert root_id is not None
    assert rt.cache.size == 1


def test_execute_root() -> None:
    rt = BoundedInfinityRuntime(cache_size=16, executor=_counting_executor)
    root_id = rt.spawn(task={"goal": "test"})
    state = rt.execute(root_id)
    assert state["executed"] is True


def test_query() -> None:
    rt = BoundedInfinityRuntime(cache_size=16, executor=_counting_executor)
    root_id = rt.spawn(task={"goal": "test"})
    rt.execute(root_id)
    state = rt.query(root_id)
    assert state is not None
    assert state["executed"] is True


def test_terminate() -> None:
    rt = BoundedInfinityRuntime(cache_size=16)
    root_id = rt.spawn(task={"goal": "test"})
    rt.terminate(root_id)
    assert not rt.cache.contains(root_id)


def test_execute_unknown_raises() -> None:
    rt = BoundedInfinityRuntime(cache_size=16)
    with pytest.raises(ValueError):
        rt.execute("no-such-agent")


# ------------------------------------------------------------------ #
# Spawning sub-agents
# ------------------------------------------------------------------ #

def test_spawn_children_during_execution() -> None:
    rt = BoundedInfinityRuntime(cache_size=16, executor=_counting_executor)
    root_id = rt.spawn(task={"spawn_children": 3})
    state = rt.execute(root_id)
    assert len(state["children_spawned"]) == 3
    assert rt.cache.size == 4  # root + 3 children


# ------------------------------------------------------------------ #
# Automatic eviction under pressure
# ------------------------------------------------------------------ #

def test_automatic_eviction() -> None:
    """Cache stays bounded even when spawning more agents than capacity."""
    rt = BoundedInfinityRuntime(
        cache_size=8,
        executor=_counting_executor,
        eviction_batch_size=2,
        idle_threshold=0.0,  # All agents are immediately eviction candidates
    )

    # Spawn more agents than the cache can hold
    root_id = rt.spawn(task={"spawn_children": 0})
    for i in range(20):
        rt.spawn(parent_id=root_id, task={"idx": i})

    # Cache should never exceed capacity
    assert rt.cache.size <= 8
    assert rt.stats.total_spawns == 21


def test_memory_bounded_deep_tree() -> None:
    """Build a deep tree that would exceed cache; memory stays bounded."""
    rt = BoundedInfinityRuntime(
        cache_size=4,
        idle_threshold=0.0,
    )

    current_id = rt.spawn(task={"depth": 0})
    for d in range(1, 50):
        current_id = rt.spawn(parent_id=current_id, task={"depth": d})
        assert rt.cache.size <= 4, f"Cache exceeded capacity at depth {d}"


# ------------------------------------------------------------------ #
# Hierarchical summarization via eviction
# ------------------------------------------------------------------ #

def test_evicted_state_query_returns_reconstructed() -> None:
    """Querying an evicted agent should return a reconstructed state."""
    rt = BoundedInfinityRuntime(
        cache_size=3,
        idle_threshold=0.0,
    )
    root = rt.spawn(task={"goal": "root"})
    c1 = rt.spawn(parent_id=root, task={"goal": "child1"})
    c2 = rt.spawn(parent_id=root, task={"goal": "child2"})
    # This spawn will trigger eviction
    c3 = rt.spawn(parent_id=root, task={"goal": "child3"})

    # At least one child should have been evicted
    # Query an evicted agent — should get reconstructed state
    for cid in [c1, c2, c3]:
        state = rt.query(cid)
        assert state is not None  # Should always get something back


def test_reconstruction_chain_walk() -> None:
    """Reconstructed state should contain ancestor summaries via chain walk."""
    rt = BoundedInfinityRuntime(
        cache_size=3,
        idle_threshold=0.0,
    )
    root = rt.spawn(task={"goal": "root"})
    c1 = rt.spawn(parent_id=root, task={"goal": "child1"})
    c2 = rt.spawn(parent_id=root, task={"goal": "child2"})
    # c3 triggers eviction of c1 (deepest/oldest child)
    c3 = rt.spawn(parent_id=root, task={"goal": "child3"})

    # Find an evicted child
    evicted_id = None
    for cid in [c1, c2, c3]:
        if not rt.cache.contains(cid):
            evicted_id = cid
            break

    if evicted_id is not None:
        state = rt.query(evicted_id)
        assert state is not None
        assert state.get("_reconstructed") is True
        assert "_chain_depth" in state
        assert "_ancestor_summaries" in state
        assert state["_chain_depth"] >= 1


def test_executed_after_eviction_is_not_flagged_reconstructed() -> None:
    """An agent evicted before it ran, then executed, is a normal cached entry.

    execute() runs such an agent from a stub marked "_reconstructed"; that
    marker must not survive into the stored state, or a later query of the
    (resident) agent looks like a reconstruction with no summaries.
    """
    rt = BoundedInfinityRuntime(cache_size=3, executor=_counting_executor,
                                idle_threshold=0.0)
    root = rt.spawn(task={"goal": "root"})
    children = [rt.spawn(parent_id=root, task={"goal": f"c{i}"}) for i in range(4)]
    evicted = next(c for c in children if not rt.cache.contains(c))
    rt.execute(evicted)
    assert rt.cache.contains(evicted)
    state = rt.query(evicted)
    assert state["executed"] is True
    assert not state.get("_reconstructed")


def test_hilbert_index_computation() -> None:
    """Runtime should compute Hilbert indices for cached entries."""
    rt = BoundedInfinityRuntime(cache_size=16)
    root = rt.spawn(task={"goal": "test"})
    entry = rt.cache.get(root)
    assert entry is not None
    assert isinstance(entry.hilbert_index, int)
    assert entry.hilbert_index >= 0


def test_state_dimensions_compressor() -> None:
    """With state_dimensions > 0, StateCompressor should be active."""
    rt = BoundedInfinityRuntime(cache_size=16, state_dimensions=3)
    assert rt._compressor is not None
    root = rt.spawn(task={"x": 1.0, "y": 2.0, "z": 3.0})
    entry = rt.cache.get(root)
    assert entry is not None
    assert isinstance(entry.hilbert_index, int)


# ------------------------------------------------------------------ #
# Memory metrics
# ------------------------------------------------------------------ #

def test_memory_usage_metrics() -> None:
    rt = BoundedInfinityRuntime(cache_size=16, executor=_counting_executor)
    root = rt.spawn(task={"spawn_children": 3})
    rt.execute(root)
    metrics = rt.memory_usage
    assert metrics["cache_capacity"] == 16
    assert metrics["cache_size"] == 4
    assert metrics["total_agents_registered"] >= 4
    assert metrics["stats"]["spawns"] == 4
    assert metrics["stats"]["executions"] == 1


# ------------------------------------------------------------------ #
# Stress: large number of agents in bounded memory
# ------------------------------------------------------------------ #

def test_1000_agents_bounded_memory() -> None:
    """Spawn 1000 agents with a cache of 32 — memory must stay bounded."""
    rt = BoundedInfinityRuntime(
        cache_size=32,
        idle_threshold=0.0,
    )
    root = rt.spawn(task={"root": True})
    for i in range(999):
        rt.spawn(parent_id=root, task={"i": i})

    assert rt.cache.size <= 32
    assert rt.stats.total_spawns == 1000
    assert rt.memory_usage["cache_size"] <= 32
