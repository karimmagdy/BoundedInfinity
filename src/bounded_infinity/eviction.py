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
        registry: Any | None = None,
        eviction_ancestor_walk: bool = True,
        pin_depth: int = 0,
        eviction_policy: str = "depth",
    ) -> None:
        self.cache = cache
        self.summarize = summarize or default_summarize
        self.eviction_batch_size = max(1, eviction_batch_size)
        self.idle_threshold = idle_threshold
        self.info_metric = info_metric or structural_info
        self.stats = EvictionStats()
        self._max_history = max_history
        # When True, eviction folds into the nearest cached ancestor; when
        # False, only the immediate cached parent (legacy behaviour). The
        # flag exists so the contribution can be reported as a clean ablation.
        self.eviction_ancestor_walk = eviction_ancestor_walk
        # Registry gives full parent pointers (incl. for already-evicted
        # ancestors), enabling eviction-time ancestor-walk folding so an
        # evicted state is summarised into the nearest *cached* ancestor
        # rather than dropped when its immediate parent is also gone. This
        # makes the eviction path symmetric with query-time reconstruction.
        self.registry = registry
        self.total_chain_repairs = 0
        # Active-path pinning: agents at depth <= pin_depth are protected from
        # eviction so the shallow "spine" of the tree always stays resident to
        # anchor reconstruction. 0 disables pinning (legacy behaviour). A
        # safety valve in _select_candidates still allows evicting pinned
        # entries if every candidate is pinned (cache would otherwise deadlock).
        self.pin_depth = pin_depth
        self.total_pin_protected = 0
        # "depth" = depth-priority (BIC default); "lru" = evict least-recently
        # used among non-pinned (the combined heuristic's concentration policy).
        self.eviction_policy = eviction_policy

    def _nearest_cached_ancestor(self, agent_id: str,
                                 excluded: set[str] | None = None) -> str | None:
        """Walk parent pointers (via the registry) to the nearest ancestor
        that is currently cached and not itself being evicted in this batch.

        Returns the anchor agent_id, or None if no cached ancestor exists.
        Falls back to the immediate cached parent when no registry is
        available (preserving legacy behaviour).
        """
        excluded = excluded or set()
        if self.registry is None:
            entry = self.cache.get(agent_id)
            pid = entry.parent_id if entry is not None else None
            if pid is not None and pid not in excluded and self.cache.get(pid) is not None:
                return pid
            return None
        rec = self.registry.get(agent_id)
        cur = rec.parent_id if rec is not None else None
        while cur is not None:
            if cur not in excluded and self.cache.get(cur) is not None:
                return cur
            anc = self.registry.get(cur)
            cur = anc.parent_id if anc is not None else None
        return None

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
        # Group candidates by their nearest *cached* ancestor (anchor), so an
        # evicted state whose immediate parent is also evicted is still folded
        # into a live ancestor instead of being dropped. The whole eviction
        # batch is excluded when resolving anchors so we never fold into a
        # node that is itself leaving the cache.
        batch_ids = {entry.agent_id for entry in candidates}
        by_anchor: dict[str | None, list[CacheEntry]] = {}
        for entry in candidates:
            if self.eviction_ancestor_walk:
                anchor_id = self._nearest_cached_ancestor(entry.agent_id, excluded=batch_ids)
            else:
                pid = entry.parent_id
                anchor_id = pid if (pid is not None and pid not in batch_ids
                                    and self.cache.get(pid) is not None) else None
            if anchor_id is not None and anchor_id != entry.parent_id:
                self.total_chain_repairs += 1
            by_anchor.setdefault(anchor_id, []).append(entry)

        for anchor_id, children in by_anchor.items():
            child_states = [c.state for c in children]
            child_ids = [c.agent_id for c in children]

            # Summarize into the nearest cached ancestor if one exists
            if anchor_id is not None:
                anchor_entry = self.cache.get(anchor_id)
                if anchor_entry is not None:
                    anchor_before = dict(anchor_entry.state)
                    anchor_entry.state = self.summarize(
                        anchor_entry.state, child_states
                    )
                    self.stats.total_summarizations += 1
                    # Track information preservation
                    ratio = preservation_ratio(
                        anchor_before, child_states,
                        anchor_entry.state, self.info_metric
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

        # Active-path pinning: keep depth<=pin_depth resident. Only drop the
        # pinned set if there is nothing else to evict (safety valve), so the
        # cache can never deadlock when it fills with pinned spine nodes.
        if self.pin_depth > 0:
            evictable = [e for e in entries if e.depth > self.pin_depth]
            if evictable:
                self.total_pin_protected += len(entries) - len(evictable)
                entries = evictable

        now = time.monotonic()

        # LRU mode: evict the least-recently-accessed non-pinned entries,
        # ignoring depth. This concentrates content into the pinned shallow
        # spine exactly like the strongest combined heuristic baseline.
        if self.eviction_policy == "lru":
            entries.sort(key=lambda e: e.last_accessed)  # oldest first
            return entries[:count]

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
