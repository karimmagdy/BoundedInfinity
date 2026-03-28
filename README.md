# BoundedInfinity

**Bounding Unbounded Agentic Swarms** — A mathematically rigorous memory
management architecture for multi-agent systems.

## Problem

When building complex multi-agent systems, agents recursively spawn sub-agents
to solve sub-tasks, creating an unbounded, exponentially growing tree of memory,
API calls, and context demands. Current frameworks handle this by simply hitting
hard limits — crashing or truncating context.

## Solution

BoundedInfinity provides a **Bounded Infinity Cache (BIC)** that uses:

- **Cantor pairing functions** to bijectively map the infinite agent tree into a
  flat address space
- **Hilbert space-filling curves** to compress high-dimensional agent state into
  cache-friendly 1D indices preserving locality
- **Hierarchical summarization eviction** to bound information loss when the
  cache is full

With formal proofs guaranteeing:

1. **Bounded Memory** — O(M) memory for any number of agents
2. **Zero Fragmentation** — no external fragmentation after any sequence of
   allocations and evictions
3. **Retrievability** — O(1) for cached states, O(log N) reconstruction for
   evicted states within bounded error ε
4. **Indefinite Execution** — the system runs forever without memory growth

## Quick Start

```bash
pip install -e ".[dev]"
pytest
```

```python
from bounded_infinity import BoundedInfinityRuntime

runtime = BoundedInfinityRuntime(cache_size=1024)
root = runtime.spawn(parent_id=None, task={"goal": "research topic X"})
result = runtime.execute(root)
```

## Project Structure

```
src/bounded_infinity/
├── cantor_pairing.py    # Generalized Cantor pairing/unpairing
├── hilbert_curve.py     # Hilbert curve state compression
├── bounded_cache.py     # Fixed-size state cache
├── eviction.py          # Hierarchical summarization eviction
├── agent_registry.py    # Agent ID → cache slot addressing
├── runtime.py           # Main BoundedInfinity orchestrator
└── adapters/            # Framework adapters (LangGraph, AutoGen, CrewAI)
tests/                   # Unit + property-based tests
benchmarks/              # Performance benchmarks
docs/                    # Paper and technical report
```

## License

Apache-2.0
