"""
Timing benchmark: wall-clock latency for spawn/query/evict operations.

Uses synthetic agents (no LLM needed) to isolate pure infrastructure
overhead across cache sizes [8, 16, 32, 64, 128].  Reports mean +/- std
wall-clock time for each operation type and backend.

Usage:
    python -m experiments.run_timing_benchmark
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
_SRC_DIR = _PROJECT_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from experiments.baselines import BICBackend, make_backend

RESULTS_DIR = Path(__file__).resolve().parent / "results"

CACHE_SIZES = [8, 16, 32, 64, 128]
BACKENDS = ["bic", "lru", "lru-summary", "fifo", "random", "unbounded"]
NUM_AGENTS = 200  # spawn this many agents per run to get stable timing
NUM_REPEATS = 5   # repeat each configuration for error bars


def _synthetic_executor(agent_id: str, task: dict[str, Any],
                        state: dict[str, Any],
                        spawn) -> dict[str, Any]:
    """Minimal synthetic executor -- no LLM calls."""
    result = dict(state)
    result["response"] = f"Synthetic response for agent {agent_id}"
    result["completed"] = True
    return result


def benchmark_single(backend_name: str, cache_size: int,
                     num_agents: int) -> dict[str, Any]:
    """Benchmark spawn, execute, query, and evict for one configuration.

    Returns a dict with lists of per-operation timings.
    """
    if backend_name == "bic":
        backend = BICBackend(cache_size=cache_size, executor=_synthetic_executor)
    elif backend_name == "unbounded":
        backend = make_backend("unbounded", executor=_synthetic_executor)
    else:
        backend = make_backend(backend_name, capacity=cache_size,
                               executor=_synthetic_executor)

    spawn_times = []
    execute_times = []
    query_times = []
    evict_times = []  # measured as terminate() wall-clock
    agent_ids = []

    # Phase 1: spawn + execute all agents
    for i in range(num_agents):
        parent_id = agent_ids[-1] if agent_ids else None

        t0 = time.perf_counter()
        aid = backend.spawn(parent_id=parent_id,
                            task={"question": f"Task {i}", "depth": i % 4})
        spawn_times.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        backend.execute(aid)
        execute_times.append(time.perf_counter() - t0)

        agent_ids.append(aid)

    # Phase 2: query all agents (some may have been evicted)
    for aid in agent_ids:
        t0 = time.perf_counter()
        backend.query(aid)
        query_times.append(time.perf_counter() - t0)

    # Phase 3: terminate (evict) all agents
    for aid in agent_ids:
        t0 = time.perf_counter()
        backend.terminate(aid)
        evict_times.append(time.perf_counter() - t0)

    return {
        "spawn": spawn_times,
        "execute": execute_times,
        "query": query_times,
        "evict": evict_times,
    }


def main():
    print("=" * 80)
    print("Timing Benchmark: spawn / execute / query / evict")
    print(f"  Agents per run: {NUM_AGENTS}")
    print(f"  Repeats: {NUM_REPEATS}")
    print(f"  Cache sizes: {CACHE_SIZES}")
    print(f"  Backends: {BACKENDS}")
    print("=" * 80)

    all_results = {}

    for cs in CACHE_SIZES:
        for bname in BACKENDS:
            key = f"{bname}_cache{cs}"
            spawn_means, exec_means, query_means, evict_means = [], [], [], []

            for rep in range(NUM_REPEATS):
                timings = benchmark_single(bname, cs, NUM_AGENTS)
                spawn_means.append(np.mean(timings["spawn"]))
                exec_means.append(np.mean(timings["execute"]))
                query_means.append(np.mean(timings["query"]))
                evict_means.append(np.mean(timings["evict"]))

            result = {
                "backend": bname,
                "cache_size": cs,
                "num_agents": NUM_AGENTS,
                "num_repeats": NUM_REPEATS,
                "spawn_mean_us": round(float(np.mean(spawn_means)) * 1e6, 2),
                "spawn_std_us": round(float(np.std(spawn_means)) * 1e6, 2),
                "execute_mean_us": round(float(np.mean(exec_means)) * 1e6, 2),
                "execute_std_us": round(float(np.std(exec_means)) * 1e6, 2),
                "query_mean_us": round(float(np.mean(query_means)) * 1e6, 2),
                "query_std_us": round(float(np.std(query_means)) * 1e6, 2),
                "evict_mean_us": round(float(np.mean(evict_means)) * 1e6, 2),
                "evict_std_us": round(float(np.std(evict_means)) * 1e6, 2),
            }
            all_results[key] = result

            print(f"  {bname:14s} cache={cs:4d}  "
                  f"spawn={result['spawn_mean_us']:8.1f}+/-{result['spawn_std_us']:.1f}us  "
                  f"exec={result['execute_mean_us']:8.1f}+/-{result['execute_std_us']:.1f}us  "
                  f"query={result['query_mean_us']:8.1f}+/-{result['query_std_us']:.1f}us  "
                  f"evict={result['evict_mean_us']:8.1f}+/-{result['evict_std_us']:.1f}us")

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        "benchmark": "timing_benchmark",
        "num_agents": NUM_AGENTS,
        "num_repeats": NUM_REPEATS,
        "cache_sizes": CACHE_SIZES,
        "backends": BACKENDS,
        "results": all_results,
    }
    outpath = RESULTS_DIR / "timing_benchmark.json"
    with open(outpath, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n-> Saved to {outpath}")

    # Summary table
    print("\n" + "=" * 100)
    print("TIMING SUMMARY (mean +/- std, microseconds per operation)")
    print("=" * 100)
    print(f"{'Backend':14s} {'Cache':>6s}  "
          f"{'Spawn (us)':>18s}  {'Execute (us)':>18s}  "
          f"{'Query (us)':>18s}  {'Evict (us)':>18s}")
    print("-" * 100)

    for cs in CACHE_SIZES:
        for bname in BACKENDS:
            key = f"{bname}_cache{cs}"
            if key not in all_results:
                continue
            r = all_results[key]
            print(f"{bname:14s} {cs:6d}  "
                  f"{r['spawn_mean_us']:8.1f}+/-{r['spawn_std_us']:6.1f}  "
                  f"{r['execute_mean_us']:8.1f}+/-{r['execute_std_us']:6.1f}  "
                  f"{r['query_mean_us']:8.1f}+/-{r['query_std_us']:6.1f}  "
                  f"{r['evict_mean_us']:8.1f}+/-{r['evict_std_us']:6.1f}")
        if cs != CACHE_SIZES[-1]:
            print("-" * 100)


if __name__ == "__main__":
    main()
