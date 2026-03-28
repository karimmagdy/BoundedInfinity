"""
Scaled LLM experiment: branching_factor=3, max_depth=3 → 40 agents.

This stresses the cache more severely (40 agents in cache_size=8 slots),
providing a stronger signal for the BIC advantage over LRU.

Usage:
    export GEMINI_API_KEY="..."
    python -m experiments.run_llm_scaled
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
from experiments.harness import run_instrumented
from experiments.research_decomposition import TaskConfig

RESULTS_DIR = Path(__file__).resolve().parent / "results"

# Scaled parameters
BRANCHING_FACTOR = 3
MAX_DEPTH = 3
# 3^0 + 3^1 + 3^2 + 3^3 = 1 + 3 + 9 + 27 = 40 agents
EXPECTED_AGENTS = sum(BRANCHING_FACTOR ** d for d in range(MAX_DEPTH + 1))
CACHE_SIZE = 8  # Still severe: 8 slots for 40 agents
MAX_LLM_CALLS = 120  # per backend: 40 agents + question decomposition overhead

SEEDS = [42, 123]
BACKENDS = ["bic", "lru", "lru-summary", "unbounded"]
QUESTION = "Survey the state of memory management in multi-agent LLM systems"


def _make_llm_executor(llm_callable):
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


def single_run(seed: int, provider: str, api_key: str | None) -> list[dict]:
    """Run one experiment pass with the given seed."""
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
            backend = BICBackend(cache_size=CACHE_SIZE, executor=llm_executor)
        elif bname == "unbounded":
            backend = make_backend("unbounded", executor=llm_executor)
        else:
            backend = make_backend(bname, capacity=CACHE_SIZE, executor=llm_executor)

        print(f"    {bname:14s} (seed={seed}) ... ", end="", flush=True)
        t0 = time.perf_counter()
        metrics = run_instrumented(bname, backend, config)
        wall_time = time.perf_counter() - t0

        r = metrics.to_dict()
        r["seed"] = seed
        r["wall_time_s"] = round(wall_time, 2)
        r["scale"] = {"branching": BRANCHING_FACTOR, "depth": MAX_DEPTH,
                      "agents": EXPECTED_AGENTS, "cache": CACHE_SIZE}
        results.append(r)

        print(f"jacc={metrics.reconstruction_quality:.3f} "
              f"sem={metrics.semantic_reconstruction_quality:.3f} "
              f"qs={metrics.query_success_rate:.3f} "
              f"({wall_time:.1f}s)")
    return results


def main():
    provider = os.environ.get("LLM_PROVIDER", "gemini")
    api_key = os.environ.get("GEMINI_API_KEY")

    est_calls = len(SEEDS) * len(BACKENDS) * EXPECTED_AGENTS
    print("=" * 60)
    print(f"Scaled LLM Experiment: {BRANCHING_FACTOR}^{MAX_DEPTH+1}-tree → "
          f"{EXPECTED_AGENTS} agents/run")
    print(f"  3 seeds × 4 backends = ~{est_calls} LLM calls")
    print(f"  Cache size: {CACHE_SIZE} (ratio: {CACHE_SIZE/EXPECTED_AGENTS:.1%})")
    print(f"  Provider: {provider}")
    print("=" * 60)

    reset_usage_tracker()
    all_runs: list[list[dict]] = []

    for i, seed in enumerate(SEEDS):
        print(f"\n  Run {i+1}/{len(SEEDS)} (seed={seed}):")
        run_results = single_run(seed, provider, api_key)
        all_runs.append(run_results)

        # Save partial results after each seed to avoid losing progress
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        partial_path = RESULTS_DIR / "experiment_5_llm_scaled_partial.json"
        with open(partial_path, "w") as f:
            json.dump([r for run in all_runs for r in run], f, indent=2)
        print(f"    [saved partial: {len(all_runs)} seeds → {partial_path.name}]")

    tracker = get_usage_tracker()
    print(f"\n  Total LLM usage: {tracker.total_calls} calls, "
          f"~{tracker.total_input_tokens} input, "
          f"~{tracker.total_output_tokens} output, "
          f"{tracker.errors} errors")

    # Aggregate
    agg = {}
    for bname in BACKENDS:
        jacc_vals, sem_vals, qs_vals, evict_vals = [], [], [], []
        for run in all_runs:
            for r in run:
                if r["backend"] == bname:
                    q = r.get("quality", {})
                    jacc_vals.append(q.get("reconstruction_quality", 0))
                    sem_vals.append(q.get("semantic_reconstruction_quality", 0))
                    qs_vals.append(q.get("query_success_rate", 0))
                    evict_vals.append(r.get("evictions", 0))
        agg[bname] = {
            "jaccard_mean": round(float(np.mean(jacc_vals)), 4),
            "jaccard_std": round(float(np.std(jacc_vals)), 4),
            "semantic_mean": round(float(np.mean(sem_vals)), 4),
            "semantic_std": round(float(np.std(sem_vals)), 4),
            "query_success_mean": round(float(np.mean(qs_vals)), 4),
            "query_success_std": round(float(np.std(qs_vals)), 4),
            "evictions_mean": round(float(np.mean(evict_vals)), 1),
        }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Save raw results
    raw_path = RESULTS_DIR / "experiment_5_llm_scaled_raw.json"
    with open(raw_path, "w") as f:
        json.dump([r for run in all_runs for r in run], f, indent=2)

    # Save aggregated
    agg_path = RESULTS_DIR / "experiment_5_llm_scaled.json"
    with open(agg_path, "w") as f:
        json.dump(agg, f, indent=2)

    print(f"\n→ Saved to {raw_path} and {agg_path}")

    # Summary table
    print("\n" + "=" * 75)
    print(f"SCALED EXPERIMENT SUMMARY ({EXPECTED_AGENTS} agents, cache={CACHE_SIZE})")
    print("=" * 75)
    print(f"{'Backend':14s} {'Jaccard':>14s} {'Semantic':>14s} "
          f"{'QuerySuccess':>14s} {'Evictions':>10s}")
    print("-" * 75)
    for bname in BACKENDS:
        a = agg[bname]
        print(f"{bname:14s} "
              f"{a['jaccard_mean']:.3f}±{a['jaccard_std']:.3f}  "
              f"{a['semantic_mean']:.3f}±{a['semantic_std']:.3f}  "
              f"{a['query_success_mean']:.3f}±{a['query_success_std']:.3f}  "
              f"{a['evictions_mean']:10.1f}")


if __name__ == "__main__":
    main()
