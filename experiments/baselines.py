"""
Baseline implementations for comparison against BoundedInfinity.

All baselines implement the SwarmBackend protocol:
  spawn(parent_id, task) → agent_id
  execute(agent_id)      → state
  query(agent_id)        → state | None
  terminate(agent_id)    → None

Six baselines:
  1. UnboundedBackend       — stores every agent state (will OOM at scale)
  2. LRUBackend             — fixed-size LRU, evicted states are *lost*
  3. FIFOBackend            — fixed-size FIFO, evicted states are *lost*
  4. RandomBackend          — fixed-size random eviction, evicted states are *lost*
  5. LRUSummaryBackend      — fixed-size LRU *with* hierarchical summarization
  6. TieredMemoryBackend    — hot/cold two-tier memory (MemGPT-style)

Baselines 1-4 do NOT summarize evicted states into parents.
Baselines 5-6 DO summarize, isolating BIC's Cantor/Hilbert contribution.
"""

from __future__ import annotations

import random
import time
import uuid
from collections import OrderedDict
from typing import Any, Callable


# ------------------------------------------------------------------ #
# Noop executor (shared)
# ------------------------------------------------------------------ #

def _noop_executor(agent_id: str, task: dict[str, Any],
                   state: dict[str, Any],
                   spawn: Callable[[dict[str, Any]], str]) -> dict[str, Any]:
    return {**state, "task_received": task, "status": "executed"}


# ------------------------------------------------------------------ #
# 1. Unbounded Backend
# ------------------------------------------------------------------ #

class UnboundedBackend:
    """Stores every agent state in a plain dict.  No eviction, no limit.

    Will consume O(N) memory for N agents.  This is the "what if we
    just kept everything?" baseline.
    """

    def __init__(self, executor: Any = None) -> None:
        self.executor = executor or _noop_executor
        self._states: dict[str, dict[str, Any]] = {}
        self._parents: dict[str, str | None] = {}
        self._tasks: dict[str, dict[str, Any]] = {}
        self.total_spawns = 0

    def spawn(self, parent_id: str | None = None,
              task: dict[str, Any] | None = None) -> str:
        agent_id = f"ub-{uuid.uuid4().hex[:12]}"
        self._states[agent_id] = {"task": task or {}}
        self._parents[agent_id] = parent_id
        self._tasks[agent_id] = task or {}
        self.total_spawns += 1
        return agent_id

    def execute(self, agent_id: str) -> dict[str, Any]:
        state = self._states.get(agent_id, {"task": self._tasks.get(agent_id, {})})

        def _spawn_child(sub_task: dict[str, Any]) -> str:
            return self.spawn(parent_id=agent_id, task=sub_task)

        updated = self.executor(agent_id, self._tasks.get(agent_id, {}),
                                state, _spawn_child)
        self._states[agent_id] = updated
        return updated

    def query(self, agent_id: str) -> dict[str, Any] | None:
        return self._states.get(agent_id)

    def terminate(self, agent_id: str) -> None:
        self._states.pop(agent_id, None)

    @property
    def memory_size(self) -> int:
        """Number of stored states."""
        return len(self._states)


# ------------------------------------------------------------------ #
# 2. LRU Backend (fixed capacity, no summarization)
# ------------------------------------------------------------------ #

class LRUBackend:
    """Fixed-size LRU cache.  Evicted states are permanently lost.

    This tests whether simply bounding memory (without summarization) is
    sufficient — the answer quality should degrade significantly.
    """

    def __init__(self, capacity: int = 64, executor: Any = None) -> None:
        self.capacity = capacity
        self.executor = executor or _noop_executor
        self._cache: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._tasks: dict[str, dict[str, Any]] = {}
        self.total_spawns = 0
        self.total_evictions = 0

    def _evict_if_needed(self) -> None:
        while len(self._cache) >= self.capacity:
            evicted_id, _ = self._cache.popitem(last=False)
            self.total_evictions += 1

    def spawn(self, parent_id: str | None = None,
              task: dict[str, Any] | None = None) -> str:
        agent_id = f"lru-{uuid.uuid4().hex[:12]}"
        self._tasks[agent_id] = task or {}
        self._evict_if_needed()
        self._cache[agent_id] = {"task": task or {}}
        self.total_spawns += 1
        return agent_id

    def execute(self, agent_id: str) -> dict[str, Any]:
        state = self._cache.get(agent_id, {"task": self._tasks.get(agent_id, {})})
        if agent_id in self._cache:
            self._cache.move_to_end(agent_id, last=True)

        def _spawn_child(sub_task: dict[str, Any]) -> str:
            return self.spawn(parent_id=agent_id, task=sub_task)

        updated = self.executor(agent_id, self._tasks.get(agent_id, {}),
                                state, _spawn_child)
        # Re-insert (may have been evicted by child spawns)
        self._evict_if_needed()
        self._cache[agent_id] = updated
        return updated

    def query(self, agent_id: str) -> dict[str, Any] | None:
        if agent_id in self._cache:
            self._cache.move_to_end(agent_id, last=True)
            return self._cache[agent_id]
        return None  # Evicted = gone forever

    def terminate(self, agent_id: str) -> None:
        self._cache.pop(agent_id, None)

    @property
    def memory_size(self) -> int:
        return len(self._cache)


# ------------------------------------------------------------------ #
# 3. FIFO Backend (fixed capacity, no summarization)
# ------------------------------------------------------------------ #

class FIFOBackend:
    """Fixed-size FIFO cache.  Oldest agents are evicted first.

    Unlike LRU, access does not refresh position — strictly first-in,
    first-out.
    """

    def __init__(self, capacity: int = 64, executor: Any = None) -> None:
        self.capacity = capacity
        self.executor = executor or _noop_executor
        self._cache: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._tasks: dict[str, dict[str, Any]] = {}
        self.total_spawns = 0
        self.total_evictions = 0

    def _evict_if_needed(self) -> None:
        while len(self._cache) >= self.capacity:
            self._cache.popitem(last=False)
            self.total_evictions += 1

    def spawn(self, parent_id: str | None = None,
              task: dict[str, Any] | None = None) -> str:
        agent_id = f"fifo-{uuid.uuid4().hex[:12]}"
        self._tasks[agent_id] = task or {}
        self._evict_if_needed()
        self._cache[agent_id] = {"task": task or {}}
        self.total_spawns += 1
        return agent_id

    def execute(self, agent_id: str) -> dict[str, Any]:
        state = self._cache.get(agent_id, {"task": self._tasks.get(agent_id, {})})

        def _spawn_child(sub_task: dict[str, Any]) -> str:
            return self.spawn(parent_id=agent_id, task=sub_task)

        updated = self.executor(agent_id, self._tasks.get(agent_id, {}),
                                state, _spawn_child)
        if agent_id in self._cache:
            self._cache[agent_id] = updated
        return updated

    def query(self, agent_id: str) -> dict[str, Any] | None:
        return self._cache.get(agent_id)

    def terminate(self, agent_id: str) -> None:
        self._cache.pop(agent_id, None)

    @property
    def memory_size(self) -> int:
        return len(self._cache)


# ------------------------------------------------------------------ #
# 4. Random Eviction Backend (fixed capacity, no summarization)
# ------------------------------------------------------------------ #

class RandomBackend:
    """Fixed-size cache with random eviction.  No summarization.

    A common strawman in cache literature.
    """

    def __init__(self, capacity: int = 64, executor: Any = None,
                 seed: int = 42) -> None:
        self.capacity = capacity
        self.executor = executor or _noop_executor
        self._cache: dict[str, dict[str, Any]] = {}
        self._tasks: dict[str, dict[str, Any]] = {}
        self._rng = random.Random(seed)
        self.total_spawns = 0
        self.total_evictions = 0

    def _evict_if_needed(self) -> None:
        while len(self._cache) >= self.capacity:
            victim = self._rng.choice(list(self._cache.keys()))
            del self._cache[victim]
            self.total_evictions += 1

    def spawn(self, parent_id: str | None = None,
              task: dict[str, Any] | None = None) -> str:
        agent_id = f"rnd-{uuid.uuid4().hex[:12]}"
        self._tasks[agent_id] = task or {}
        self._evict_if_needed()
        self._cache[agent_id] = {"task": task or {}}
        self.total_spawns += 1
        return agent_id

    def execute(self, agent_id: str) -> dict[str, Any]:
        state = self._cache.get(agent_id, {"task": self._tasks.get(agent_id, {})})

        def _spawn_child(sub_task: dict[str, Any]) -> str:
            return self.spawn(parent_id=agent_id, task=sub_task)

        updated = self.executor(agent_id, self._tasks.get(agent_id, {}),
                                state, _spawn_child)
        if agent_id in self._cache:
            self._cache[agent_id] = updated
        return updated

    def query(self, agent_id: str) -> dict[str, Any] | None:
        return self._cache.get(agent_id)

    def terminate(self, agent_id: str) -> None:
        self._cache.pop(agent_id, None)

    @property
    def memory_size(self) -> int:
        return len(self._cache)


# ------------------------------------------------------------------ #
# 5. LRU + Summarization Backend (fair comparison)
# ------------------------------------------------------------------ #

def _default_summarize_standalone(parent_state: dict[str, Any],
                                  child_states: list[dict[str, Any]]) -> dict[str, Any]:
    """Standalone summarization identical to BIC's default_summarize."""
    merged = dict(parent_state)
    existing_summaries: list[dict[str, Any]] = list(merged.get("_summaries", []))
    for child in child_states:
        summary: dict[str, Any] = {}
        for k, v in child.items():
            if isinstance(k, str) and k.startswith("_"):
                continue
            summary[k] = v
        existing_summaries.append(summary)
    max_summaries = merged.get("_max_summaries", 16)
    if len(existing_summaries) > max_summaries:
        existing_summaries = existing_summaries[-max_summaries:]
    merged["_summaries"] = existing_summaries
    merged["_evicted_children_count"] = merged.get("_evicted_children_count", 0) + len(child_states)
    return merged


class LRUSummaryBackend:
    """Fixed-size LRU with hierarchical summarization on eviction.

    Unlike plain LRU, evicted states are summarized into their parent's
    state before removal.  This isolates BIC's Cantor/Hilbert novelty:
    if LRU+Summary matches BIC, then the architecture adds no value
    beyond summarization itself.
    """

    def __init__(self, capacity: int = 64, executor: Any = None) -> None:
        self.capacity = capacity
        self.executor = executor or _noop_executor
        self._cache: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._parents: dict[str, str | None] = {}
        self._tasks: dict[str, dict[str, Any]] = {}
        self.total_spawns = 0
        self.total_evictions = 0

    def _evict_if_needed(self) -> None:
        while len(self._cache) >= self.capacity:
            evicted_id, evicted_state = self._cache.popitem(last=False)
            self.total_evictions += 1
            # Summarize into parent if parent is cached
            parent_id = self._parents.get(evicted_id)
            if parent_id and parent_id in self._cache:
                self._cache[parent_id] = _default_summarize_standalone(
                    self._cache[parent_id], [evicted_state]
                )

    def spawn(self, parent_id: str | None = None,
              task: dict[str, Any] | None = None) -> str:
        agent_id = f"lrus-{uuid.uuid4().hex[:12]}"
        self._tasks[agent_id] = task or {}
        self._parents[agent_id] = parent_id
        self._evict_if_needed()
        self._cache[agent_id] = {"task": task or {}}
        self.total_spawns += 1
        return agent_id

    def execute(self, agent_id: str) -> dict[str, Any]:
        state = self._cache.get(agent_id, {"task": self._tasks.get(agent_id, {})})
        if agent_id in self._cache:
            self._cache.move_to_end(agent_id, last=True)

        def _spawn_child(sub_task: dict[str, Any]) -> str:
            return self.spawn(parent_id=agent_id, task=sub_task)

        updated = self.executor(agent_id, self._tasks.get(agent_id, {}),
                                state, _spawn_child)
        self._evict_if_needed()
        self._cache[agent_id] = updated
        return updated

    def query(self, agent_id: str) -> dict[str, Any] | None:
        if agent_id in self._cache:
            self._cache.move_to_end(agent_id, last=True)
            return self._cache[agent_id]
        # Evicted but summarized — walk parents for summaries
        parent_id = self._parents.get(agent_id)
        while parent_id is not None:
            if parent_id in self._cache:
                parent_state = self._cache[parent_id]
                return {
                    "_reconstructed": True,
                    "_ancestor_summaries": [{
                        "ancestor_id": parent_id,
                        "summaries": parent_state.get("_summaries", []),
                    }],
                    "task": self._tasks.get(agent_id, {}),
                    "agent_id": agent_id,
                }
            parent_id = self._parents.get(parent_id)
        return None

    def terminate(self, agent_id: str) -> None:
        self._cache.pop(agent_id, None)

    @property
    def memory_size(self) -> int:
        return len(self._cache)


# ------------------------------------------------------------------ #
# 6. Tiered Memory Backend (MemGPT-style hot/cold)
# ------------------------------------------------------------------ #

class TieredMemoryBackend:
    """Two-tier memory inspired by MemGPT's main/archival architecture.

    - Hot tier: fixed-size dict (capacity=M), full states
    - Cold tier: unlimited dict, stores summarized/compressed states

    On eviction from hot: summarize state and move to cold tier.
    On query: check hot first, then cold. Cold returns a summarized version.

    This simulates MemGPT's core memory management pattern without
    requiring actual LLM calls for memory management decisions.
    """

    def __init__(self, capacity: int = 64, executor: Any = None) -> None:
        self.capacity = capacity
        self.executor = executor or _noop_executor
        self._hot: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._cold: dict[str, dict[str, Any]] = {}
        self._parents: dict[str, str | None] = {}
        self._tasks: dict[str, dict[str, Any]] = {}
        self.total_spawns = 0
        self.total_evictions = 0

    def _evict_if_needed(self) -> None:
        while len(self._hot) >= self.capacity:
            evicted_id, evicted_state = self._hot.popitem(last=False)
            self.total_evictions += 1
            # Summarize: keep only top-level non-internal keys + truncate strings
            summary: dict[str, Any] = {"_cold_summary": True}
            for k, v in evicted_state.items():
                if isinstance(k, str) and k.startswith("_"):
                    continue
                if isinstance(v, str) and len(v) > 200:
                    summary[k] = v[:200] + "..."
                else:
                    summary[k] = v
            self._cold[evicted_id] = summary
            # Also summarize into parent if parent is in hot tier
            parent_id = self._parents.get(evicted_id)
            if parent_id and parent_id in self._hot:
                self._hot[parent_id] = _default_summarize_standalone(
                    self._hot[parent_id], [evicted_state]
                )

    def spawn(self, parent_id: str | None = None,
              task: dict[str, Any] | None = None) -> str:
        agent_id = f"tier-{uuid.uuid4().hex[:12]}"
        self._tasks[agent_id] = task or {}
        self._parents[agent_id] = parent_id
        self._evict_if_needed()
        self._hot[agent_id] = {"task": task or {}}
        self.total_spawns += 1
        return agent_id

    def execute(self, agent_id: str) -> dict[str, Any]:
        # Promote from cold if needed
        if agent_id not in self._hot and agent_id in self._cold:
            self._evict_if_needed()
            self._hot[agent_id] = self._cold.pop(agent_id)

        state = self._hot.get(agent_id, {"task": self._tasks.get(agent_id, {})})
        if agent_id in self._hot:
            self._hot.move_to_end(agent_id, last=True)

        def _spawn_child(sub_task: dict[str, Any]) -> str:
            return self.spawn(parent_id=agent_id, task=sub_task)

        updated = self.executor(agent_id, self._tasks.get(agent_id, {}),
                                state, _spawn_child)
        self._evict_if_needed()
        self._hot[agent_id] = updated
        return updated

    def query(self, agent_id: str) -> dict[str, Any] | None:
        # Check hot tier first
        if agent_id in self._hot:
            self._hot.move_to_end(agent_id, last=True)
            return self._hot[agent_id]
        # Check cold tier
        if agent_id in self._cold:
            cold_state = self._cold[agent_id]
            return {**cold_state, "_reconstructed": True}
        # Check parent summaries
        parent_id = self._parents.get(agent_id)
        while parent_id is not None:
            if parent_id in self._hot:
                parent_state = self._hot[parent_id]
                return {
                    "_reconstructed": True,
                    "_ancestor_summaries": [{
                        "ancestor_id": parent_id,
                        "summaries": parent_state.get("_summaries", []),
                    }],
                    "task": self._tasks.get(agent_id, {}),
                    "agent_id": agent_id,
                }
            if parent_id in self._cold:
                return {
                    "_reconstructed": True,
                    "_cold_summary": True,
                    "task": self._tasks.get(agent_id, {}),
                }
            parent_id = self._parents.get(parent_id)
        return None

    def terminate(self, agent_id: str) -> None:
        self._hot.pop(agent_id, None)
        self._cold.pop(agent_id, None)

    @property
    def memory_size(self) -> int:
        return len(self._hot)


# ------------------------------------------------------------------ #
# BIC adapter (wraps BoundedInfinityRuntime into SwarmBackend)
# ------------------------------------------------------------------ #

class BICBackend:
    """Wraps BoundedInfinityRuntime to expose the SwarmBackend interface
    with identical method signatures as the baselines.
    """

    def __init__(self, cache_size: int = 64, executor: Any = None,
                 **kwargs: Any) -> None:
        from bounded_infinity.runtime import BoundedInfinityRuntime
        self._rt = BoundedInfinityRuntime(
            cache_size=cache_size,
            executor=executor,
            idle_threshold=0.0,
            eviction_batch_size=max(1, cache_size // 8),
            **kwargs,
        )

    def spawn(self, parent_id: str | None = None,
              task: dict[str, Any] | None = None) -> str:
        return self._rt.spawn(parent_id=parent_id, task=task)

    def execute(self, agent_id: str) -> dict[str, Any]:
        return self._rt.execute(agent_id)

    def query(self, agent_id: str) -> dict[str, Any] | None:
        return self._rt.query(agent_id)

    def terminate(self, agent_id: str) -> None:
        self._rt.terminate(agent_id)

    @property
    def runtime(self):
        return self._rt

    @property
    def memory_size(self) -> int:
        return self._rt.cache.size

    @property
    def eviction_stats(self):
        return self._rt.eviction_manager.stats


# ------------------------------------------------------------------ #
# 7. LRU+Summary with Ancestor-Walk on eviction (reviewer Q1 / Tier B B1)
# ------------------------------------------------------------------ #
#
# Strictly stronger LRU+Summary baseline: when the immediate parent of an
# evicted child is *also* evicted, the existing LRUSummaryBackend drops
# the child's state on the floor.  This variant walks up the ancestor
# chain to find the nearest *cached* ancestor and folds the evicted
# child's state into THAT ancestor's slot.  This emulates BIC's
# eviction-time chain repair without using BIC's Cantor-structured
# slot addressing.
#
# Compared to BIC, this baseline still lacks: (a) Cantor addressing,
# (b) depth-prioritised eviction policy, (c) ancestor pinning.
# Compared to LRUSummaryBackend, it adds (only): registry-assisted
# ancestor folding on eviction.

class LRUSummaryAWBackend:
    """LRU+Summary with eviction-time ancestor walk.

    Identical to LRUSummaryBackend except that, when the immediate
    parent of an evicted agent is itself evicted, the evicted state is
    folded into the *nearest cached ancestor* (via a registry walk)
    instead of being dropped.  This addresses reviewer concern Q1
    (baseline fairness for chain repair).
    """

    def __init__(self, capacity: int = 64, executor: Any = None) -> None:
        self.capacity = capacity
        self.executor = executor or _noop_executor
        self._cache: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._parents: dict[str, str | None] = {}
        self._tasks: dict[str, dict[str, Any]] = {}
        self.total_spawns = 0
        self.total_evictions = 0
        self.total_chain_repairs = 0  # diagnostic: how often did we have to walk past the immediate parent?

    def _nearest_cached_ancestor(self, agent_id: str) -> str | None:
        """Walk parent pointers until we find one that is currently cached."""
        cur = self._parents.get(agent_id)
        while cur is not None:
            if cur in self._cache:
                return cur
            cur = self._parents.get(cur)
        return None

    def _evict_if_needed(self) -> None:
        while len(self._cache) >= self.capacity:
            evicted_id, evicted_state = self._cache.popitem(last=False)
            self.total_evictions += 1
            anchor = self._nearest_cached_ancestor(evicted_id)
            # Diagnostic: did we have to walk past the immediate parent?
            immediate_parent = self._parents.get(evicted_id)
            if anchor is not None and anchor != immediate_parent:
                self.total_chain_repairs += 1
            if anchor is not None:
                self._cache[anchor] = _default_summarize_standalone(
                    self._cache[anchor], [evicted_state]
                )

    def spawn(self, parent_id: str | None = None,
              task: dict[str, Any] | None = None) -> str:
        agent_id = f"lrusaw-{uuid.uuid4().hex[:12]}"
        self._tasks[agent_id] = task or {}
        self._parents[agent_id] = parent_id
        self._evict_if_needed()
        self._cache[agent_id] = {"task": task or {}}
        self.total_spawns += 1
        return agent_id

    def execute(self, agent_id: str) -> dict[str, Any]:
        state = self._cache.get(agent_id, {"task": self._tasks.get(agent_id, {})})
        if agent_id in self._cache:
            self._cache.move_to_end(agent_id, last=True)

        def _spawn_child(sub_task: dict[str, Any]) -> str:
            return self.spawn(parent_id=agent_id, task=sub_task)

        updated = self.executor(agent_id, self._tasks.get(agent_id, {}),
                                state, _spawn_child)
        self._evict_if_needed()
        self._cache[agent_id] = updated
        return updated

    def query(self, agent_id: str) -> dict[str, Any] | None:
        if agent_id in self._cache:
            self._cache.move_to_end(agent_id, last=True)
            return self._cache[agent_id]
        # Evicted: walk to nearest cached ancestor and return its summaries
        anchor = self._nearest_cached_ancestor(agent_id)
        if anchor is None:
            return None
        anchor_state = self._cache[anchor]
        return {
            "_reconstructed": True,
            "_ancestor_summaries": [{
                "ancestor_id": anchor,
                "summaries": anchor_state.get("_summaries", []),
            }],
            "task": self._tasks.get(agent_id, {}),
            "agent_id": agent_id,
        }

    def terminate(self, agent_id: str) -> None:
        self._cache.pop(agent_id, None)

    @property
    def memory_size(self) -> int:
        return len(self._cache)


# ------------------------------------------------------------------ #
# 8. LRU+Pin: ancestor-protection baseline (reviewer Q5 / Tier B B2)
# ------------------------------------------------------------------ #
#
# Standard LRU eviction except that shallow agents (root + agents at
# depth <= pin_depth) are *pinned* and never evicted.  This emulates
# the "active-path pinning" the reviewer asked for: in a depth-d
# decomposition tree, leaves still get evicted under cache pressure
# but their ancestors are protected, so the ancestor chain is intact
# at query time.
#
# This is closer in spirit to BIC's depth-prioritised eviction
# (BIC = "evict deepest first") than plain LRU.  Comparing BIC vs
# LRU+Pin isolates how much advantage BIC derives from
# *deterministic Cantor addressing* vs from *depth-priority eviction*.

class LRUPinBackend:
    """LRU eviction with shallow-ancestor pinning.

    Agents at depth <= pin_depth (default: 1, i.e. root + immediate
    children) are pinned and never evicted.  All other agents follow
    standard LRU eviction.  Evicted states are NOT summarised into
    their parent (this is plain pin-only LRU; see LRUSummaryAWBackend
    for the variant that combines pinning with summarisation).
    """

    def __init__(self, capacity: int = 64, executor: Any = None,
                 pin_depth: int = 1) -> None:
        self.capacity = capacity
        self.executor = executor or _noop_executor
        self.pin_depth = pin_depth
        self._cache: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._parents: dict[str, str | None] = {}
        self._depths: dict[str, int] = {}
        self._tasks: dict[str, dict[str, Any]] = {}
        self.total_spawns = 0
        self.total_evictions = 0
        self.total_pin_skips = 0  # diagnostic: how often did we skip a pinned candidate during eviction?

    def _is_pinned(self, agent_id: str) -> bool:
        return self._depths.get(agent_id, 0) <= self.pin_depth

    def _evict_if_needed(self) -> None:
        while len(self._cache) >= self.capacity:
            # Find the LRU non-pinned victim
            victim = None
            for candidate in list(self._cache.keys()):
                if not self._is_pinned(candidate):
                    victim = candidate
                    break
                self.total_pin_skips += 1
            if victim is None:
                # All cached agents are pinned: cache is over-pinned.
                # Fall back to evicting the LRU pinned agent (a hard
                # safety valve so we never exceed capacity).
                victim = next(iter(self._cache))
            del self._cache[victim]
            self.total_evictions += 1

    def spawn(self, parent_id: str | None = None,
              task: dict[str, Any] | None = None) -> str:
        agent_id = f"lrupin-{uuid.uuid4().hex[:12]}"
        self._tasks[agent_id] = task or {}
        self._parents[agent_id] = parent_id
        parent_depth = self._depths.get(parent_id, -1) if parent_id else -1
        self._depths[agent_id] = parent_depth + 1
        self._evict_if_needed()
        self._cache[agent_id] = {"task": task or {}}
        self.total_spawns += 1
        return agent_id

    def execute(self, agent_id: str) -> dict[str, Any]:
        state = self._cache.get(agent_id, {"task": self._tasks.get(agent_id, {})})
        if agent_id in self._cache:
            self._cache.move_to_end(agent_id, last=True)

        def _spawn_child(sub_task: dict[str, Any]) -> str:
            return self.spawn(parent_id=agent_id, task=sub_task)

        updated = self.executor(agent_id, self._tasks.get(agent_id, {}),
                                state, _spawn_child)
        self._evict_if_needed()
        self._cache[agent_id] = updated
        return updated

    def query(self, agent_id: str) -> dict[str, Any] | None:
        if agent_id in self._cache:
            self._cache.move_to_end(agent_id, last=True)
            return self._cache[agent_id]
        return None  # No summarisation: evicted = gone

    def terminate(self, agent_id: str) -> None:
        self._cache.pop(agent_id, None)

    @property
    def memory_size(self) -> int:
        return len(self._cache)


def make_backend(name: str, capacity: int = 64, executor: Any = None,
                 seed: int = 42, **kwargs: Any):
    """Factory for creating backends by name."""
    backends = {
        "bic": lambda: BICBackend(cache_size=capacity, executor=executor, **kwargs),
        "unbounded": lambda: UnboundedBackend(executor=executor),
        "lru": lambda: LRUBackend(capacity=capacity, executor=executor),
        "fifo": lambda: FIFOBackend(capacity=capacity, executor=executor),
        "random": lambda: RandomBackend(capacity=capacity, executor=executor, seed=seed),
        "lru-summary": lambda: LRUSummaryBackend(capacity=capacity, executor=executor),
        "lru-summary-aw": lambda: LRUSummaryAWBackend(capacity=capacity, executor=executor),
        "lru-pin": lambda: LRUPinBackend(capacity=capacity, executor=executor, **kwargs),
        "tiered": lambda: TieredMemoryBackend(capacity=capacity, executor=executor),
    }
    if name not in backends:
        raise ValueError(f"Unknown backend: {name}. Choose from: {list(backends)}")
    return backends[name]()
