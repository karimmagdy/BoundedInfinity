"""
BoundedInfinity Runtime — the main orchestrator.

Ties together the Agent Registry, Bounded Cache, Eviction Manager, and
(optionally) the Hilbert State Compressor into a single, easy-to-use API:

  runtime = BoundedInfinityRuntime(cache_size=1024)
  root    = runtime.spawn(parent_id=None, task={"goal": "research X"})
  result  = runtime.execute(root)
  state   = runtime.query(root)
  runtime.terminate(root)

The runtime guarantees:
  - At most `cache_size` agent states in memory at any time (Theorem 1)
  - Automatic eviction with hierarchical summarization when full (Theorem 3/4)
  - Zero fragmentation via periodic compaction (Theorem 2)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from bounded_infinity.agent_registry import AgentRecord, AgentRegistry, AgentStatus
from bounded_infinity.bounded_cache import BoundedCache, CacheFull
from bounded_infinity.eviction import EvictionManager, SummarizationFunction
from bounded_infinity.hilbert_curve import StateCompressor
from bounded_infinity.metrics import InformationMetric, structural_info


class AgentExecutor(Protocol):
    """Protocol for user-supplied agent execution logic."""

    def __call__(self, agent_id: str, task: dict[str, Any],
                 state: dict[str, Any],
                 spawn: Callable[[dict[str, Any]], str]) -> dict[str, Any]:
        """Execute an agent's task.

        Parameters
        ----------
        agent_id : str
            The agent's unique ID.
        task : dict
            The task to execute.
        state : dict
            The agent's current state (may contain summaries of evicted children).
        spawn : callable
            Call spawn(sub_task) to spawn a sub-agent. Returns the sub-agent's ID.

        Returns
        -------
        dict — the updated state after execution.
        """
        ...


def _noop_executor(agent_id: str, task: dict[str, Any],
                   state: dict[str, Any],
                   spawn: Callable[[dict[str, Any]], str]) -> dict[str, Any]:
    """Default no-op executor — just marks the task as seen."""
    return {**state, "task_received": task, "status": "executed"}


@dataclass
class RuntimeStats:
    """Aggregate runtime statistics."""
    total_spawns: int = 0
    total_executions: int = 0
    total_evictions: int = 0
    total_terminations: int = 0
    peak_cache_occupancy: int = 0


class BoundedInfinityRuntime:
    """Main entry point for the BoundedInfinity system.

    Parameters
    ----------
    cache_size : int
        Maximum number of agent states held in memory (M).
    executor : AgentExecutor | None
        User-supplied function that runs agent logic.
    summarize : SummarizationFunction | None
        Custom summarization for eviction. Default merges child keys into parent.
    eviction_batch_size : int
        How many agents to evict at once when cache is full.
    idle_threshold : float
        Seconds of inactivity before an agent is eligible for eviction.
    auto_compact_interval : int
        Compact the cache every N evictions (0 to disable).
    addressing : str
        Slot addressing for the registry: "cantor" (default) or "hash"
        (plain hash of the ancestry path, for the Cantor-vs-hash ablation).
    """

    def __init__(
        self,
        cache_size: int = 1024,
        executor: AgentExecutor | None = None,
        summarize: SummarizationFunction | None = None,
        eviction_batch_size: int = 8,
        idle_threshold: float = 60.0,
        auto_compact_interval: int = 100,
        state_dimensions: int = 0,
        hilbert_resolution: int = 8,
        info_metric: InformationMetric | None = None,
        eviction_ancestor_walk: bool = True,
        pin_depth: int = 0,
        eviction_policy: str = "depth",
        reconstruct_full_chain: bool = False,
        addressing: str = "cantor",
    ) -> None:
        self.cache_size = cache_size
        self.reconstruct_full_chain = reconstruct_full_chain
        self.executor = executor or _noop_executor
        self.auto_compact_interval = auto_compact_interval
        self.info_metric = info_metric or structural_info

        # Hilbert state compressor (0 = disabled, uses state fingerprint)
        self.state_dimensions = state_dimensions
        self._compressor: StateCompressor | None = None
        if state_dimensions > 0:
            self._compressor = StateCompressor(
                dimensions=state_dimensions, resolution=hilbert_resolution
            )

        self.cache = BoundedCache(capacity=cache_size)
        self.registry = AgentRegistry(cache_capacity=cache_size, addressing=addressing)
        self.eviction_manager = EvictionManager(
            cache=self.cache,
            summarize=summarize,
            eviction_batch_size=eviction_batch_size,
            idle_threshold=idle_threshold,
            info_metric=self.info_metric,
            registry=self.registry,
            eviction_ancestor_walk=eviction_ancestor_walk,
            pin_depth=pin_depth,
            eviction_policy=eviction_policy,
        )
        self.stats = RuntimeStats()
        self._evictions_since_compact = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def spawn(self, parent_id: str | None = None,
              task: dict[str, Any] | None = None) -> str:
        """Spawn a new agent (possibly as a sub-agent of parent_id).

        If the cache is full, eviction is triggered automatically — the
        system never crashes due to memory.

        Returns the new agent's ID.
        """
        # Register in the tree
        record = self.registry.register(parent_id=parent_id, task=task)

        # Ensure cache has room
        self.eviction_manager.ensure_capacity(needed=1)

        # Store initial state in cache
        initial_state: dict[str, Any] = {"task": task or {}}
        self.cache.put(
            agent_id=record.agent_id,
            state=initial_state,
            hilbert_index=self._compute_hilbert_index(initial_state),
            parent_id=record.parent_id,
            depth=record.depth,
            preferred_slot=record.preferred_slot,
        )

        self.stats.total_spawns += 1
        self.stats.peak_cache_occupancy = max(
            self.stats.peak_cache_occupancy, self.cache.size
        )

        # Update evicted agents in registry
        self._sync_eviction_status()

        return record.agent_id

    def execute(self, agent_id: str) -> dict[str, Any]:
        """Execute an agent's task using the configured executor.

        The executor may call `spawn` to create sub-agents.

        Returns the agent's updated state.
        """
        record = self.registry.get(agent_id)
        if record is None:
            raise ValueError(f"Agent {agent_id} not found in registry")

        # Load state from cache
        entry = self.cache.get(agent_id)
        if entry is None:
            # Agent was evicted — reconstruct minimal state
            state: dict[str, Any] = {"task": record.task, "_reconstructed": True}
        else:
            state = entry.state

        self.registry.set_status(agent_id, AgentStatus.RUNNING)

        # Provide a spawn callback bound to this agent as parent
        def _spawn_child(sub_task: dict[str, Any]) -> str:
            return self.spawn(parent_id=agent_id, task=sub_task)

        # Execute
        updated_state = self.executor(agent_id, record.task, state, _spawn_child)

        # Write back to cache (ensure room first)
        if not self.cache.contains(agent_id):
            self.eviction_manager.ensure_capacity(needed=1)
        self.cache.put(
            agent_id=agent_id,
            state=updated_state,
            hilbert_index=self._compute_hilbert_index(updated_state),
            parent_id=record.parent_id,
            depth=record.depth,
            preferred_slot=record.preferred_slot,
        )

        self.registry.set_status(agent_id, AgentStatus.COMPLETED)
        self.stats.total_executions += 1
        self.stats.peak_cache_occupancy = max(
            self.stats.peak_cache_occupancy, self.cache.size
        )

        self._maybe_compact()
        return updated_state

    def query(self, agent_id: str) -> dict[str, Any] | None:
        """Query an agent's state.

        If the agent is cached, returns its state directly in O(1).
        If evicted, walks up the ancestor chain collecting summaries
        at each level until a cached ancestor (or the root) is reached,
        then returns a merged reconstruction.  This is the O(log N)
        retrieval path from Theorem 3.
        """
        entry = self.cache.get(agent_id)
        if entry is not None:
            return entry.state

        # Agent not in cache — attempt reconstruction via ancestor chain
        record = self.registry.get(agent_id)
        if record is None:
            return None

        return self._reconstruct_state(record)

    def terminate(self, agent_id: str) -> None:
        """Terminate an agent, freeing its cache slot."""
        self.cache.remove(agent_id)
        self.registry.set_status(agent_id, AgentStatus.TERMINATED)
        self.stats.total_terminations += 1

    def compact(self) -> int:
        """Manually trigger cache compaction. Returns number of entries moved."""
        return self.cache.compact()

    # ------------------------------------------------------------------
    # Metrics & introspection
    # ------------------------------------------------------------------

    @property
    def memory_usage(self) -> dict[str, Any]:
        """Current memory metrics."""
        return {
            "cache_size": self.cache.size,
            "cache_capacity": self.cache.capacity,
            "cache_free": self.cache.free,
            "cache_occupancy": self.cache.occupancy,
            "fragmentation": self.cache.fragmentation_ratio,
            "total_agents_registered": self.registry.total_agents,
            "max_tree_depth": self.registry.max_depth,
            "stats": {
                "spawns": self.stats.total_spawns,
                "executions": self.stats.total_executions,
                "evictions": self.eviction_manager.stats.total_evictions,
                "terminations": self.stats.total_terminations,
                "peak_occupancy": self.stats.peak_cache_occupancy,
            },
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _compute_hilbert_index(self, state: dict[str, Any]) -> int:
        """Compute a Hilbert index for the given state.

        If a StateCompressor is configured (state_dimensions > 0), extracts
        a numeric feature vector from the state and compresses it.  Otherwise
        returns a hash-based fingerprint that still provides some locality.
        """
        if self._compressor is not None:
            vec = self._extract_state_vector(state)
            if vec is not None:
                return self._compressor.compress(vec)
        # Fallback: hash-based fingerprint from state content
        return hash(str(sorted(
            (k, str(v)[:64]) for k, v in state.items()
            if not (isinstance(k, str) and k.startswith("_"))
        ))) % (1 << 30)

    def _extract_state_vector(self, state: dict[str, Any]) -> list[float] | None:
        """Try to extract a numeric vector from the state for Hilbert compression.

        Looks for a '_state_vector' key (explicit) or collects numeric values.
        Returns None if the state has no numeric content.
        """
        # Explicit vector takes priority
        if "_state_vector" in state:
            vec = state["_state_vector"]
            if isinstance(vec, (list, tuple)) and len(vec) == self.state_dimensions:
                return [float(x) for x in vec]

        # Auto-extract: collect numeric values from top-level keys
        nums: list[float] = []
        for k, v in state.items():
            if isinstance(k, str) and k.startswith("_"):
                continue
            if isinstance(v, (int, float)):
                nums.append(float(v))
            if len(nums) >= self.state_dimensions:
                break

        if len(nums) == self.state_dimensions:
            # Normalize to [0, 1] via sigmoid
            import math
            return [1.0 / (1.0 + math.exp(-x)) if abs(x) < 500 else (1.0 if x > 0 else 0.0)
                    for x in nums]
        return None

    def _reconstruct_state(self, record: AgentRecord) -> dict[str, Any]:
        """Reconstruct an evicted agent's state by walking the ancestor chain.

        Walks from the agent's parent up to the root (or nearest cached
        ancestor), collecting _summaries at each level.  Merges them into
        a reconstruction dict with provenance tracking.

        Time complexity: O(d) where d is the agent's depth in the tree.
        Information loss: bounded by (1 − ε)^d per Theorem 3.
        """
        summaries_chain: list[dict[str, Any]] = []
        chain_depth = 0
        current_id = record.parent_id

        # Walk ancestors. Legacy mode stops at the first cached ancestor even
        # if its summary buffer is empty (a "live" resident node), which yields
        # content-empty reconstructions. Full-chain mode walks all the way to
        # the root, collecting every cached ancestor that actually has summary
        # content, so a reconstruction is empty only if no ancestor holds any.
        while current_id is not None:
            chain_depth += 1
            ancestor_entry = self.cache.get(current_id)
            if ancestor_entry is not None:
                summ = ancestor_entry.state.get("_summaries")
                if summ:
                    summaries_chain.append({
                        "ancestor_id": current_id,
                        "ancestor_depth": ancestor_entry.depth,
                        "summaries": summ,
                    })
                if not self.reconstruct_full_chain:
                    break  # Stop — we have a live anchor point

            # Keep walking up via the registry (covers evicted ancestors too).
            ancestor_record = self.registry.get(current_id)
            if ancestor_record is None:
                break
            current_id = ancestor_record.parent_id

        return {
            "_reconstructed": True,
            "_chain_depth": chain_depth,
            "_ancestor_summaries": summaries_chain,
            "task": record.task,
            "agent_id": record.agent_id,
            "depth": record.depth,
        }

    def _sync_eviction_status(self) -> None:
        """Mark evicted agents in the registry."""
        cached_ids = self.cache.agent_ids()
        for record in self.registry.active_agents():
            if record.agent_id not in cached_ids and record.status not in (
                AgentStatus.EVICTED, AgentStatus.TERMINATED, AgentStatus.COMPLETED
            ):
                record.status = AgentStatus.EVICTED

    def _maybe_compact(self) -> None:
        """Auto-compact if the interval has been reached."""
        if self.auto_compact_interval <= 0:
            return
        evictions = self.eviction_manager.stats.total_evictions
        if evictions - self._evictions_since_compact >= self.auto_compact_interval:
            self.cache.compact()
            self._evictions_since_compact = evictions
