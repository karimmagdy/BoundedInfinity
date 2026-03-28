"""
Benchmarks for BoundedInfinity.

Three benchmark scenarios:
  1. Exponential spawning — measures memory vs. naive unbounded approach
  2. Long-running — continuous spawn/execute/terminate for N steps
  3. Memory profile — tracks memory usage over time for plotting
"""

from __future__ import annotations

import sys
import time
import tracemalloc
from dataclasses import dataclass
from typing import Any, Callable

from bounded_infinity.runtime import BoundedInfinityRuntime


def _bench_executor(agent_id: str, task: dict[str, Any],
                    state: dict[str, Any],
                    spawn: Callable[[dict[str, Any]], str]) -> dict[str, Any]:
    """Benchmark executor: spawns children based on task params."""
    result = dict(state)
    result["executed"] = True
    k = task.get("branching_factor", 0)
    depth = task.get("depth", 0)
    max_depth = task.get("max_depth", 0)
    if depth < max_depth:
        for i in range(k):
            spawn({
                "branching_factor": k,
                "depth": depth + 1,
                "max_depth": max_depth,
            })
    return result


@dataclass
class BenchmarkResult:
    name: str
    elapsed_seconds: float
    total_spawns: int
    peak_cache_occupancy: int
    cache_capacity: int
    peak_memory_bytes: int
    evictions: int
    samples: list[dict[str, Any]]


def benchmark_exponential_spawning(
    cache_size: int = 64,
    branching_factor: int = 3,
    max_depth: int = 5,
) -> BenchmarkResult:
    """Each agent spawns k children up to depth d.  Total agents = (k^(d+1)-1)/(k-1)."""
    tracemalloc.start()
    start = time.perf_counter()

    rt = BoundedInfinityRuntime(
        cache_size=cache_size,
        executor=_bench_executor,
        idle_threshold=0.0,
        eviction_batch_size=max(1, cache_size // 8),
    )

    samples: list[dict[str, Any]] = []
    root = rt.spawn(task={
        "branching_factor": branching_factor,
        "depth": 0,
        "max_depth": max_depth,
    })

    # BFS execution
    queue = [root]
    step = 0
    while queue:
        agent_id = queue.pop(0)
        state = rt.execute(agent_id)
        children = state.get("children_spawned", [])
        # Collect children from registry instead (executor doesn't return them in this setup)
        record = rt.registry.get(agent_id)
        if record:
            queue.extend(record.children)

        step += 1
        if step % 50 == 0:
            current, peak = tracemalloc.get_traced_memory()
            samples.append({
                "step": step,
                "cache_size": rt.cache.size,
                "memory_bytes": current,
                "evictions": rt.eviction_manager.stats.total_evictions,
            })

    elapsed = time.perf_counter() - start
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return BenchmarkResult(
        name="exponential_spawning",
        elapsed_seconds=elapsed,
        total_spawns=rt.stats.total_spawns,
        peak_cache_occupancy=rt.stats.peak_cache_occupancy,
        cache_capacity=cache_size,
        peak_memory_bytes=peak_mem,
        evictions=rt.eviction_manager.stats.total_evictions,
        samples=samples,
    )


def benchmark_long_running(
    cache_size: int = 32,
    total_steps: int = 10_000,
) -> BenchmarkResult:
    """Continuous spawn/execute/terminate cycle for N steps."""
    tracemalloc.start()
    start = time.perf_counter()

    rt = BoundedInfinityRuntime(
        cache_size=cache_size,
        idle_threshold=0.0,
        eviction_batch_size=max(1, cache_size // 4),
    )

    root = rt.spawn(task={"root": True})
    samples: list[dict[str, Any]] = []
    active: list[str] = [root]

    for step in range(total_steps):
        # Spawn a new child of a random active agent
        parent = active[step % len(active)] if active else root
        try:
            new_id = rt.spawn(parent_id=parent, task={"step": step})
            active.append(new_id)
        except Exception:
            pass

        # Terminate oldest active agent periodically
        if len(active) > cache_size // 2 and step % 3 == 0:
            old = active.pop(0)
            rt.terminate(old)

        if step % 500 == 0:
            current, peak = tracemalloc.get_traced_memory()
            samples.append({
                "step": step,
                "cache_size": rt.cache.size,
                "memory_bytes": current,
                "evictions": rt.eviction_manager.stats.total_evictions,
                "active_agents": len(active),
            })

    elapsed = time.perf_counter() - start
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return BenchmarkResult(
        name="long_running",
        elapsed_seconds=elapsed,
        total_spawns=rt.stats.total_spawns,
        peak_cache_occupancy=rt.stats.peak_cache_occupancy,
        cache_capacity=cache_size,
        peak_memory_bytes=peak_mem,
        evictions=rt.eviction_manager.stats.total_evictions,
        samples=samples,
    )


def benchmark_naive_unbounded(total_agents: int = 1000) -> dict[str, Any]:
    """Baseline: naive unbounded list of agent states (no cache)."""
    tracemalloc.start()
    start = time.perf_counter()

    agents: list[dict[str, Any]] = []
    for i in range(total_agents):
        agents.append({
            "agent_id": f"agent-{i}",
            "state": {"task": {"i": i}, "data": list(range(50))},
            "parent": f"agent-{i-1}" if i > 0 else None,
        })

    elapsed = time.perf_counter() - start
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return {
        "name": "naive_unbounded",
        "total_agents": total_agents,
        "elapsed_seconds": elapsed,
        "peak_memory_bytes": peak_mem,
    }


def run_all() -> None:
    """Run all benchmarks and print results."""
    print("=" * 60)
    print("BoundedInfinity Benchmarks")
    print("=" * 60)

    # Exponential spawning
    print("\n--- Exponential Spawning (k=3, d=5, cache=64) ---")
    r1 = benchmark_exponential_spawning(cache_size=64, branching_factor=3, max_depth=5)
    total_expected = (3**6 - 1) // 2  # 364 agents
    print(f"  Total agents spawned: {r1.total_spawns} (expected ~{total_expected})")
    print(f"  Peak cache occupancy: {r1.peak_cache_occupancy} / {r1.cache_capacity}")
    print(f"  Evictions:            {r1.evictions}")
    print(f"  Peak memory:          {r1.peak_memory_bytes / 1024:.1f} KB")
    print(f"  Elapsed:              {r1.elapsed_seconds:.3f}s")

    # Long running
    print("\n--- Long Running (10k steps, cache=32) ---")
    r2 = benchmark_long_running(cache_size=32, total_steps=10_000)
    print(f"  Total agents spawned: {r2.total_spawns}")
    print(f"  Peak cache occupancy: {r2.peak_cache_occupancy} / {r2.cache_capacity}")
    print(f"  Evictions:            {r2.evictions}")
    print(f"  Peak memory:          {r2.peak_memory_bytes / 1024:.1f} KB")
    print(f"  Elapsed:              {r2.elapsed_seconds:.3f}s")

    # Naive baseline
    print("\n--- Naive Unbounded Baseline (1000 agents) ---")
    r3 = benchmark_naive_unbounded(total_agents=1000)
    print(f"  Total agents:  {r3['total_agents']}")
    print(f"  Peak memory:   {r3['peak_memory_bytes'] / 1024:.1f} KB")
    print(f"  Elapsed:       {r3['elapsed_seconds']:.3f}s")

    # Comparison
    print("\n--- Memory Comparison (1000 agents) ---")
    r4 = benchmark_exponential_spawning(cache_size=32, branching_factor=2, max_depth=9)
    print(f"  BIC (cache=32):   {r4.peak_memory_bytes / 1024:.1f} KB  ({r4.total_spawns} agents)")
    print(f"  Naive unbounded:  {r3['peak_memory_bytes'] / 1024:.1f} KB  ({r3['total_agents']} agents)")
    ratio = r3["peak_memory_bytes"] / max(r4.peak_memory_bytes, 1)
    print(f"  Memory saved:     {ratio:.1f}x")

    print("\n" + "=" * 60)
    print("All benchmarks complete.")


if __name__ == "__main__":
    run_all()
