"""
Experiment 6: Scaled LLM experiment with 50 agents across multiple cache sizes.

Runs a research decomposition task with depth=3, branching_factor=3 using a
real LLM (Gemini-2.0-flash) to stress-test BIC at scale.  Sweeps across
cache_sizes=[8, 16, 32] with 5 seeds per configuration to provide
robust error bars.

Reports:
  - Semantic reconstruction quality (TF-IDF cosine)
  - Query success rate
  - Wall-clock time per operation (spawn, query, evict)

Usage:
    export GEMINI_API_KEY="..."
    python -m experiments.run_llm_50agents
"""
from __future__ import annotations

import json
import os
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

from bounded_infinity.adapters.llm_client import (
    get_usage_tracker,
    make_llm_callable,
    reset_usage_tracker,
)
from experiments.baselines import BICBackend, make_backend
from experiments.harness import run_instrumented as _run_instrumented_base


def run_instrumented(*args, **kwargs):
    """Wrapper that always logs text pairs for offline semantic-metric scoring."""
    kwargs.setdefault("log_text_pairs", True)
    return _run_instrumented_base(*args, **kwargs)
from experiments.research_decomposition import TaskConfig

RESULTS_DIR = Path(__file__).resolve().parent / "results"

# --------------- Experiment parameters --------------- #
BRANCHING_FACTOR = 3
MAX_DEPTH = 3
# 3^0 + 3^1 + 3^2 + 3^3 = 1 + 3 + 9 + 27 = 40 tree nodes;
# with overhead from decomposition we target ~50 agents.
EXPECTED_AGENTS = sum(BRANCHING_FACTOR ** d for d in range(MAX_DEPTH + 1))  # 40
CACHE_SIZES = [8, 16, 32]
MAX_LLM_CALLS = 150  # per backend: 50 agents + question decomposition overhead

SEEDS = [42, 123, 7, 2024, 314]
BACKENDS = ["bic", "lru", "lru-summary", "lru-summary-aw", "lru-pin", "unbounded"]
QUESTION = "Survey the state of memory management in multi-agent LLM systems"


def _make_llm_executor(llm_callable):
    """Create an executor that uses a real LLM for agent responses."""

    def executor(agent_id, task, state, spawn):
        result = dict(state)
        question = task.get("question", "")
        depth = task.get("depth", 0)
        prompt = (
            f"You are a research agent investigating this question:\n"
            f"{question}\n\n"
            f"Provide a concise, factual answer in 2-3 sentences."
        )
        response = llm_callable(prompt)
        result["response"] = response
        result["question"] = question
        result["depth"] = depth
        result["completed"] = True
        result["_llm_generated"] = True
        return result

    return executor


def single_run(seed: int, cache_size: int,
               provider: str, api_key: str | None) -> list[dict]:
    """Run one experiment pass with the given seed and cache size."""
    results = []
    for bname in BACKENDS:
        llm = make_llm_callable(
            provider=provider,
            max_tokens=150,
            api_key=api_key,
            max_calls=MAX_LLM_CALLS,
        )
        llm_executor = _make_llm_executor(llm)

        config = TaskConfig(
            branching_factor=BRANCHING_FACTOR,
            max_depth=MAX_DEPTH,
            response_tokens=150,
            mode="llm",
            llm_callable=llm,
            seed=seed,
            question=QUESTION,
        )

        if bname == "bic":
            backend = BICBackend(cache_size=cache_size, executor=llm_executor)
        elif bname == "unbounded":
            backend = make_backend("unbounded", executor=llm_executor)
        else:
            backend = make_backend(bname, capacity=cache_size, executor=llm_executor)

        print(f"      {bname:14s} (seed={seed}, cache={cache_size}) ... ",
              end="", flush=True)
        t0 = time.perf_counter()
        metrics = run_instrumented(bname, backend, config)
        wall_time = time.perf_counter() - t0

        r = metrics.to_dict()
        r["seed"] = seed
        r["cache_size"] = cache_size
        r["wall_time_s"] = round(wall_time, 2)
        r["scale"] = {
            "branching": BRANCHING_FACTOR,
            "depth": MAX_DEPTH,
            "expected_agents": EXPECTED_AGENTS,
            "cache": cache_size,
        }
        # Extract per-operation timing for convenience
        r["timing"] = {
            "spawn_mean_s": r.get("latency", {}).get("spawn", {}).get("mean_s", 0),
            "spawn_count": r.get("latency", {}).get("spawn", {}).get("count", 0),
            "query_mean_s": r.get("latency", {}).get("query", {}).get("mean_s", 0),
            "query_count": r.get("latency", {}).get("query", {}).get("count", 0),
            "execute_mean_s": r.get("latency", {}).get("execute", {}).get("mean_s", 0),
            "execute_count": r.get("latency", {}).get("execute", {}).get("count", 0),
        }
        results.append(r)

        print(f"sem={metrics.semantic_reconstruction_quality:.3f} "
              f"qs={metrics.query_success_rate:.3f} "
              f"({wall_time:.1f}s)")
    return results


def aggregate_results(all_runs: list[list[dict]]) -> dict:
    """Compute mean +/- std for each (backend, cache_size) pair."""
    agg = {}
    for bname in BACKENDS:
        for cs in CACHE_SIZES:
            key = f"{bname}_cache{cs}"
            sem_vals, qs_vals, evict_vals = [], [], []
            spawn_times, query_times, exec_times = [], [], []
            wall_times = []

            for run in all_runs:
                for r in run:
                    if r["backend"] == bname and r.get("cache_size") == cs:
                        q = r.get("quality", {})
                        sem_vals.append(q.get("semantic_reconstruction_quality", 0))
                        qs_vals.append(q.get("query_success_rate", 0))
                        evict_vals.append(r.get("evictions", 0))
                        wall_times.append(r.get("wall_time_s", 0))
                        t = r.get("timing", {})
                        spawn_times.append(t.get("spawn_mean_s", 0))
                        query_times.append(t.get("query_mean_s", 0))
                        exec_times.append(t.get("execute_mean_s", 0))
                    # unbounded has no cache_size in the record for make_backend
                    elif (r["backend"] == bname == "unbounded"
                          and r.get("cache_size") == cs):
                        q = r.get("quality", {})
                        sem_vals.append(q.get("semantic_reconstruction_quality", 0))
                        qs_vals.append(q.get("query_success_rate", 0))
                        evict_vals.append(r.get("evictions", 0))
                        wall_times.append(r.get("wall_time_s", 0))
                        t = r.get("timing", {})
                        spawn_times.append(t.get("spawn_mean_s", 0))
                        query_times.append(t.get("query_mean_s", 0))
                        exec_times.append(t.get("execute_mean_s", 0))

            if not sem_vals:
                continue

            agg[key] = {
                "backend": bname,
                "cache_size": cs,
                "n_runs": len(sem_vals),
                "semantic_mean": round(float(np.mean(sem_vals)), 4),
                "semantic_std": round(float(np.std(sem_vals)), 4),
                "query_success_mean": round(float(np.mean(qs_vals)), 4),
                "query_success_std": round(float(np.std(qs_vals)), 4),
                "evictions_mean": round(float(np.mean(evict_vals)), 1),
                "wall_time_mean_s": round(float(np.mean(wall_times)), 2),
                "wall_time_std_s": round(float(np.std(wall_times)), 2),
                "timing": {
                    "spawn_mean_s": round(float(np.mean(spawn_times)), 6),
                    "spawn_std_s": round(float(np.std(spawn_times)), 6),
                    "query_mean_s": round(float(np.mean(query_times)), 6),
                    "query_std_s": round(float(np.std(query_times)), 6),
                    "execute_mean_s": round(float(np.mean(exec_times)), 6),
                    "execute_std_s": round(float(np.std(exec_times)), 6),
                },
            }
    return agg


def main():
    provider = os.environ.get("LLM_PROVIDER", "gemini")
    api_key = os.environ.get("GEMINI_API_KEY")

    total_runs = len(SEEDS) * len(CACHE_SIZES) * len(BACKENDS)
    est_calls = len(SEEDS) * len(CACHE_SIZES) * len(BACKENDS) * EXPECTED_AGENTS
    print("=" * 70)
    print(f"Experiment 6: 50-Agent LLM Experiment")
    print(f"  Tree: branching={BRANCHING_FACTOR}, depth={MAX_DEPTH} "
          f"-> ~{EXPECTED_AGENTS} agents/run")
    print(f"  Cache sizes: {CACHE_SIZES}")
    print(f"  Seeds: {SEEDS} ({len(SEEDS)} runs per config)")
    print(f"  Backends: {BACKENDS}")
    print(f"  Total configurations: {total_runs} "
          f"(~{est_calls} LLM calls)")
    print(f"  Provider: {provider}")
    print("=" * 70)

    reset_usage_tracker()
    all_runs: list[list[dict]] = []

    for cs_idx, cache_size in enumerate(CACHE_SIZES):
        print(f"\n  Cache size: {cache_size} "
              f"(ratio: {cache_size/EXPECTED_AGENTS:.1%})")
        print(f"  {'-' * 50}")

        for i, seed in enumerate(SEEDS):
            print(f"    Run {i+1}/{len(SEEDS)} (seed={seed}):")
            run_results = single_run(seed, cache_size, provider, api_key)
            all_runs.append(run_results)

            # Save partial results after each seed
            RESULTS_DIR.mkdir(parents=True, exist_ok=True)
            partial_path = RESULTS_DIR / "experiment_6_llm_50agents_partial.json"
            with open(partial_path, "w") as f:
                json.dump([r for run in all_runs for r in run], f, indent=2)
            print(f"      [saved partial: {len(all_runs)} runs "
                  f"-> {partial_path.name}]")

    tracker = get_usage_tracker()
    print(f"\n  Total LLM usage: {tracker.total_calls} calls, "
          f"~{tracker.total_input_tokens} input, "
          f"~{tracker.total_output_tokens} output, "
          f"{tracker.errors} errors")

    # Aggregate results
    agg = aggregate_results(all_runs)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Save raw results
    raw_path = RESULTS_DIR / "experiment_6_llm_50agents_raw.json"
    flat_results = [r for run in all_runs for r in run]
    with open(raw_path, "w") as f:
        json.dump(flat_results, f, indent=2)

    # Save aggregated results
    output = {
        "experiment": "6_llm_50agents",
        "provider": provider,
        "n_seeds": len(SEEDS),
        "seeds": SEEDS,
        "cache_sizes": CACHE_SIZES,
        "branching_factor": BRANCHING_FACTOR,
        "max_depth": MAX_DEPTH,
        "expected_agents": EXPECTED_AGENTS,
        "backends": BACKENDS,
        "aggregated": agg,
        "raw_runs": all_runs,
    }
    agg_path = RESULTS_DIR / "experiment_6_llm_50agents.json"
    with open(agg_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n-> Saved raw to {raw_path}")
    print(f"-> Saved aggregated to {agg_path}")

    # Summary table
    print("\n" + "=" * 90)
    print(f"EXPERIMENT 6 SUMMARY ({EXPECTED_AGENTS} agents, "
          f"{len(SEEDS)} seeds, cache_sizes={CACHE_SIZES})")
    print("=" * 90)
    print(f"{'Backend':14s} {'Cache':>6s} {'Semantic':>14s} "
          f"{'QuerySuccess':>14s} {'Evictions':>10s} "
          f"{'Spawn(ms)':>12s} {'Query(ms)':>12s}")
    print("-" * 90)
    for cs in CACHE_SIZES:
        for bname in BACKENDS:
            key = f"{bname}_cache{cs}"
            if key not in agg:
                continue
            a = agg[key]
            t = a["timing"]
            print(f"{bname:14s} {cs:6d} "
                  f"{a['semantic_mean']:.3f}+/-{a['semantic_std']:.3f}  "
                  f"{a['query_success_mean']:.3f}+/-{a['query_success_std']:.3f}  "
                  f"{a['evictions_mean']:10.1f} "
                  f"{t['spawn_mean_s']*1000:8.3f}+/-{t['spawn_std_s']*1000:.3f} "
                  f"{t['query_mean_s']*1000:8.3f}+/-{t['query_std_s']*1000:.3f}")
        if cs != CACHE_SIZES[-1]:
            print("-" * 90)


if __name__ == "__main__":
    main()
