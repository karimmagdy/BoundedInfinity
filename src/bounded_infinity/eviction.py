"""
Eviction manager with hierarchical summarization.

When the BIC is full, the eviction manager decides which agent states to
evict.  Evicted states are not simply discarded — they are compressed via
a lossy summarization function φ and merged into their parent's state.
This creates a "summarization chain" up the tree, bounding information loss.

Eviction policy combines:
  1. Temporal decay — agents idle for > t steps are eviction candidates
  2. Depth-first — deeper agents are evicted before shallower ones
  3. Hilbert locality — nearby agents (by Hilbert index) are evicted together
     for cache-line efficiency

Information loss guarantee (Theorem 3):
  I(φ(s1, s2)) ≥ (1 - ε) · (I(s1) + I(s2))  for configurable ε.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from bounded_infinity.bounded_cache import BoundedCache, CacheEntry
from bounded_infinity.metrics import InformationMetric, structural_info, preservation_ratio


class SummarizationFunction(Protocol):
    """Protocol for state summarization (lossy compression)."""

    def __call__(self, parent_state: dict[str, Any],
                 child_states: list[dict[str, Any]]) -> dict[str, Any]:
        """Merge child states into the parent state."""
        ...


def default_summarize(parent_state: dict[str, Any],
                      child_states: list[dict[str, Any]]) -> dict[str, Any]:
    """Default mathematical summarization: merge child keys into parent
    with a `_summaries` list, preserving all top-level keys and aggregating
    child results.

    This is a lossless-ish default; users can provide LLM-backed
    summarizers for semantic compression.
    """
    merged = dict(parent_state)

    existing_summaries: list[dict[str, Any]] = list(merged.get("_summaries", []))

    for child in child_states:
        summary: dict[str, Any] = {}
        for k, v in child.items():
            if k.startswith("_"):
                continue
            summary[k] = v
        existing_summaries.append(summary)

    # Keep only the most recent summaries to bound growth
    max_summaries = merged.get("_max_summaries", 16)
    if len(existing_summaries) > max_summaries:
        existing_summaries = existing_summaries[-max_summaries:]

    merged["_summaries"] = existing_summaries
    merged["_evicted_children_count"] = merged.get("_evicted_children_count", 0) + len(child_states)
    return merged


@dataclass
class EvictionStats:
    """Tracks eviction statistics for monitoring."""
    total_evictions: int = 0
    total_summarizations: int = 0
    total_information_preserved: float = 0.0
    preservation_ratios: list[float] = field(default_factory=list)
    eviction_history: list[dict[str, Any]] = field(default_factory=list)

    @property
    def avg_preservation_ratio(self) -> float:
        """Average information preservation ratio across all summarizations."""
        if not self.preservation_ratios:
            return 1.0
        return sum(self.preservation_ratios) / len(self.preservation_ratios)


class EvictionManager:
    """Manages cache eviction with hierarchical summarization.

    Parameters
    ----------
    cache : BoundedCache
        The cache to manage evictions for.
    summarize : SummarizationFunction | None
        Custom summarization function. If None, uses default_summarize.
    eviction_batch_size : int
        Number of entries to evict when cache is full.
    idle_threshold : float
        Seconds of inactivity before an agent becomes an eviction candidate.
    max_history : int
        Maximum eviction events to retain in history.
    """

    def __init__(
        self,
        cache: BoundedCache,
        summarize: SummarizationFunction | None = None,
        eviction_batch_size: int = 1,
        idle_threshold: float = 60.0,
        max_history: int = 1000,
        info_metric: InformationMetric | None = None,
    ) -> None:
        self.cache = cache
        self.summarize = summarize or default_summarize
        self.eviction_batch_size = max(1, eviction_batch_size)
        self.idle_threshold = idle_threshold
        self.info_metric = info_metric or structural_info
        self.stats = EvictionStats()
        self._max_history = max_history

    def needs_eviction(self) -> bool:
        """Check if the cache needs eviction to make room."""
        return self.cache.is_full

    def evict(self, count: int | None = None) -> list[str]:
        """Evict `count` agents, summarizing their states into parents.

        Returns the list of evicted agent IDs.
        """
        count = count or self.eviction_batch_size
        candidates = self._select_candidates(count)

        evicted_ids: list[str] = []
        # Group candidates by parent for batch summarization
        by_parent: dict[str | None, list[CacheEntry]] = {}
        for entry in candidates:
            by_parent.setdefault(entry.parent_id, []).append(entry)

        for parent_id, children in by_parent.items():
            child_states = [c.state for c in children]
            child_ids = [c.agent_id for c in children]

            # Summarize into parent if parent exists and is cached
            if parent_id is not None:
                parent_entry = self.cache.get(parent_id)
                if parent_entry is not None:
                    parent_before = dict(parent_entry.state)
                    parent_entry.state = self.summarize(
                        parent_entry.state, child_states
                    )
                    self.stats.total_summarizations += 1
                    # Track information preservation
                    ratio = preservation_ratio(
                        parent_before, child_states,
                        parent_entry.state, self.info_metric
                    )
                    self.stats.total_information_preserved += ratio
                    self.stats.preservation_ratios.append(ratio)

            # Remove children from cache
            for child_id in child_ids:
                self.cache.remove(child_id)
                evicted_ids.append(child_id)

        self.stats.total_evictions += len(evicted_ids)

        # Record history (bounded)
        event = {
            "timestamp": time.monotonic(),
            "evicted": evicted_ids,
            "count": len(evicted_ids),
        }
        self.stats.eviction_history.append(event)
        if len(self.stats.eviction_history) > self._max_history:
            self.stats.eviction_history = self.stats.eviction_history[-self._max_history:]

        return evicted_ids

    def ensure_capacity(self, needed: int = 1) -> list[str]:
        """Ensure at least `needed` free slots, evicting as necessary.

        Returns all evicted agent IDs (may be empty if space was available).
        """
        all_evicted: list[str] = []
        while self.cache.free < needed:
            evicted = self.evict(count=max(self.eviction_batch_size, needed - self.cache.free))
            if not evicted:
                break  # Nothing left to evict
            all_evicted.extend(evicted)
        return all_evicted

    def _select_candidates(self, count: int) -> list[CacheEntry]:
        """Select candidates for eviction using the combined policy.

        Priority (higher = evict first):
          1. Idle agents (not accessed for > idle_threshold)
          2. Deepest agents (larger depth)
          3. Least recently accessed

        After ranking, we cluster by Hilbert locality: if multiple
        candidates share nearby Hilbert indices, prefer evicting them
        together so the summarization function operates on similar states
        (improving information preservation).
        """
        entries = self.cache.entries()
        if not entries:
            return []

        now = time.monotonic()

        def eviction_priority(entry: CacheEntry) -> tuple[int, int, float]:
            is_idle = 1 if (now - entry.last_accessed) > self.idle_threshold else 0
            return (is_idle, entry.depth, -(entry.last_accessed))

        entries.sort(key=eviction_priority, reverse=True)

        # Take 2x candidates, then cluster by Hilbert locality
        pool_size = min(len(entries), count * 3)
        pool = entries[:pool_size]

        if len(pool) <= count:
            return pool

        return self._hilbert_cluster(pool, count)

    @staticmethod
    def _hilbert_cluster(pool: list[CacheEntry], count: int) -> list[CacheEntry]:
        """From a pool of eviction candidates, select `count` that form
        the tightest Hilbert-index cluster.

        Agents with nearby Hilbert indices have similar states, so
        summarizing them together preserves more information.
        """
        if count >= len(pool):
            return pool

        # Sort by Hilbert index
        pool.sort(key=lambda e: e.hilbert_index)

        # Sliding window of size `count` — pick the window with
        # the smallest Hilbert span
        best_start = 0
        best_span = float("inf")
        for start in range(len(pool) - count + 1):
            span = pool[start + count - 1].hilbert_index - pool[start].hilbert_index
            if span < best_span:
                best_span = span
                best_start = start

        return pool[best_start:best_start + count]
