"""
Agent Registry — Maps agent IDs to cache slots via Cantor addressing.

Each agent in the spawn tree is identified by its *tree path*: a sequence of
sibling indices from the root.  The registry:

  1. Maintains the parent-child relationships (the spawn tree structure)
  2. Computes the Cantor-based address for each agent: addr(a) = π^(k)(path)
  3. Maps that address to a preferred cache slot: slot = addr mod M
  4. Tracks agent metadata (depth, status, children)

This is the bridge between the mathematical addressing scheme and the
physical BIC cache.

The addressing function is selectable for the Cantor-vs-hash ablation:
``addressing="cantor"`` (default) uses generalized Cantor pairing of the
tree path; ``addressing="hash"`` uses a plain cryptographic hash of the full
ancestry path. Nothing else in the registry changes between the two modes.
"""

from __future__ import annotations

import hashlib
import itertools
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

from bounded_infinity.cantor_pairing import tree_address

ADDRESSING_MODES = ("cantor", "hash")


def ancestry_hash(path: tuple[int, ...] | list[int]) -> int:
    """Plain hash of an agent's ancestry path (sibling indices root→leaf).

    Deterministic across processes (unlike the builtin ``hash``, which is
    salted per interpreter): BLAKE2b over the comma-joined path, truncated to
    64 bits. Not a bijection and not invertible, which is the point of the
    ablation against Cantor addressing.

    >>> ancestry_hash([2, 0, 3]) == ancestry_hash((2, 0, 3))
    True
    """
    digest = hashlib.blake2b(",".join(map(str, path)).encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big")


class AgentStatus(Enum):
    """Lifecycle status of an agent."""
    PENDING = auto()     # Spawned but not yet executed
    RUNNING = auto()     # Currently executing
    COMPLETED = auto()   # Finished execution
    EVICTED = auto()     # State evicted from cache (summarized into parent)
    TERMINATED = auto()  # Explicitly terminated


@dataclass
class AgentRecord:
    """Metadata for a registered agent."""
    agent_id: str
    parent_id: str | None
    tree_path: tuple[int, ...]
    cantor_address: int | None  # None when the registry uses hash addressing
    depth: int
    status: AgentStatus = AgentStatus.PENDING
    children: list[str] = field(default_factory=list)
    task: dict[str, Any] = field(default_factory=dict)
    preferred_slot: int | None = None
    address: int = 0  # the address actually used to derive preferred_slot


class AgentRegistry:
    """Registry of all agents in the swarm, mapping IDs to tree positions
    and cache slots.

    Parameters
    ----------
    cache_capacity : int
        The BIC cache capacity M, used to compute preferred slots.
    addressing : str
        "cantor" (default) or "hash"; see the module docstring.
    """

    def __init__(self, cache_capacity: int, addressing: str = "cantor") -> None:
        if addressing not in ADDRESSING_MODES:
            raise ValueError(f"addressing must be one of {ADDRESSING_MODES}, got {addressing!r}")
        self.cache_capacity = cache_capacity
        self.addressing = addressing
        self._agents: dict[str, AgentRecord] = {}
        self._id_counter = itertools.count()
        self._child_counters: dict[str, int] = {}  # parent_id → next child index

    def register(self, parent_id: str | None = None,
                 task: dict[str, Any] | None = None) -> AgentRecord:
        """Register a new agent in the swarm.

        Parameters
        ----------
        parent_id : str | None
            The parent agent's ID. None for the root agent.
        task : dict | None
            The task assigned to this agent.

        Returns
        -------
        AgentRecord with the agent's metadata and Cantor-derived preferred slot.
        """
        agent_id = f"agent-{next(self._id_counter)}"

        if parent_id is None:
            # Root agent
            tree_path: tuple[int, ...] = ()
            depth = 0
        else:
            parent = self._agents.get(parent_id)
            if parent is None:
                raise ValueError(f"Parent agent {parent_id} not found in registry")
            # Assign sibling index
            sibling_idx = self._child_counters.get(parent_id, 0)
            self._child_counters[parent_id] = sibling_idx + 1
            tree_path = parent.tree_path + (sibling_idx,)
            depth = parent.depth + 1
            parent.children.append(agent_id)

        if self.addressing == "hash":
            # The hash has fixed width, so the full ancestry path is used.
            addr = ancestry_hash(tree_path)
            cantor_addr = None
        else:
            # Limit path depth for Cantor address computation to avoid
            # astronomically large numbers in deep trees (Cantor pairing
            # grows double-exponentially with depth).
            MAX_ADDR_DEPTH = 8
            addr_path = tree_path[-MAX_ADDR_DEPTH:] if len(tree_path) > MAX_ADDR_DEPTH else tree_path
            addr = tree_address(list(addr_path))
            cantor_addr = addr
        preferred_slot = addr % self.cache_capacity

        record = AgentRecord(
            agent_id=agent_id,
            parent_id=parent_id,
            tree_path=tree_path,
            cantor_address=cantor_addr,
            depth=depth,
            status=AgentStatus.PENDING,
            task=task or {},
            preferred_slot=preferred_slot,
            address=addr,
        )
        self._agents[agent_id] = record
        return record

    def get(self, agent_id: str) -> AgentRecord | None:
        """Look up an agent record."""
        return self._agents.get(agent_id)

    def set_status(self, agent_id: str, status: AgentStatus) -> None:
        """Update an agent's lifecycle status."""
        record = self._agents.get(agent_id)
        if record is None:
            raise ValueError(f"Agent {agent_id} not found")
        record.status = status

    def children_of(self, agent_id: str) -> list[AgentRecord]:
        """Get all direct children of an agent."""
        record = self._agents.get(agent_id)
        if record is None:
            return []
        return [self._agents[cid] for cid in record.children if cid in self._agents]

    def subtree(self, agent_id: str) -> list[AgentRecord]:
        """Get all descendants of an agent (BFS)."""
        result: list[AgentRecord] = []
        queue = [agent_id]
        while queue:
            current_id = queue.pop(0)
            record = self._agents.get(current_id)
            if record is None:
                continue
            if current_id != agent_id:
                result.append(record)
            queue.extend(record.children)
        return result

    def depth_range(self, min_depth: int = 0,
                    max_depth: int | None = None) -> list[AgentRecord]:
        """Get all agents within a depth range."""
        return [
            r for r in self._agents.values()
            if r.depth >= min_depth and (max_depth is None or r.depth <= max_depth)
        ]

    @property
    def total_agents(self) -> int:
        """Total number of registered agents."""
        return len(self._agents)

    @property
    def max_depth(self) -> int:
        """Maximum depth in the agent tree."""
        if not self._agents:
            return 0
        return max(r.depth for r in self._agents.values())

    def active_agents(self) -> list[AgentRecord]:
        """Agents that are PENDING or RUNNING."""
        return [
            r for r in self._agents.values()
            if r.status in (AgentStatus.PENDING, AgentStatus.RUNNING)
        ]
