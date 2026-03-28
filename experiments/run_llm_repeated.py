"""
Run LLM experiment 3 times with different seeds for error bars.
Saves individual run results and aggregated mean±std.
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

from bounded_infinity.adapters.llm_client import (
    get_usage_tracker,
    make_llm_callable,
    reset_usage_tracker,
)
from experiments.baselines import BICBackend, make_backend
from experiments.harness import run_instrumented
from experiments.research_decomposition import TaskConfig

RESULTS_DIR = Path(__file__).resolve().parent / "results"

SEEDS = [42, 123, 7]
BACKENDS = ["bic", "lru", "lru-summary", "unbounded"]
CACHE_SIZE = 4
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
            max_calls=50,
        )
        llm_executor = _make_llm_executor(llm)

        config = TaskConfig(
            branching_factor=2,
            max_depth=2,
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
        results.append(r)

        print(f"jacc={metrics.reconstruction_quality:.3f} "
              f"sem={metrics.semantic_reconstruction_quality:.3f} "
              f"qs={metrics.query_success_rate:.3f} "
              f"({wall_time:.1f}s)")
    return results


def main():
    provider = os.environ.get("LLM_PROVIDER", "gemini")
    api_key = os.environ.get("GEMINI_API_KEY")

    print("=" * 60)
    print(f"LLM Experiment: 3 runs × 4 backends = ~168 LLM calls")
    print(f"Provider: {provider}")
    print("=" * 60)

    reset_usage_tracker()
    all_runs: list[list[dict]] = []

    for i, seed in enumerate(SEEDS):
        print(f"\n  Run {i+1}/3 (seed={seed}):")
        run_results = single_run(seed, provider, api_key)
        all_runs.append(run_results)

    tracker = get_usage_tracker()
    print(f"\n  Total LLM usage: {tracker.total_calls} calls, "
          f"~{tracker.total_input_tokens} input, "
          f"~{tracker.total_output_tokens} output, "
          f"{tracker.errors} errors")

    # Aggregate: compute mean ± std for each backend
    agg = {}
    for bname in BACKENDS:
        jacc_vals, sem_vals, qs_vals = [], [], []
        for run in all_runs:
            for r in run:
                if r["backend"] == bname:
                    q = r.get("quality", {})
                    jacc_vals.append(q.get("reconstruction_quality", 0))
                    sem_vals.append(q.get("semantic_reconstruction_quality", 0))
                    qs_vals.append(q.get("query_success_rate", 0))
        agg[bname] = {
            "jaccard_mean": round(float(np.mean(jacc_vals)), 4),
            "jaccard_std": round(float(np.std(jacc_vals)), 4),
            "semantic_mean": round(float(np.mean(sem_vals)), 4),
            "semantic_std": round(float(np.std(sem_vals)), 4),
            "query_success_mean": round(float(np.mean(qs_vals)), 4),
            "query_success_std": round(float(np.std(qs_vals)), 4),
            "n_runs": len(jacc_vals),
            "raw_jaccard": [round(v, 4) for v in jacc_vals],
            "raw_semantic": [round(v, 4) for v in sem_vals],
            "raw_query_success": [round(v, 4) for v in qs_vals],
        }

    # Print summary
    print("\n" + "=" * 70)
    print("AGGREGATED RESULTS (mean ± std over 3 runs)")
    print("=" * 70)
    print(f"{'Backend':14s} {'Jacc (mean±std)':>18s} {'Sem (mean±std)':>18s} "
          f"{'QS (mean±std)':>18s}")
    print("-" * 70)
    for bname in BACKENDS:
        a = agg[bname]
        print(f"{bname:14s} "
              f"{a['jaccard_mean']:.4f}±{a['jaccard_std']:.4f}  "
              f"{a['semantic_mean']:.4f}±{a['semantic_std']:.4f}  "
              f"{a['query_success_mean']:.4f}±{a['query_success_std']:.4f}")

    # Save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        "experiment": "5_llm_repeated",
        "provider": provider,
        "n_runs": len(SEEDS),
        "seeds": SEEDS,
        "cache_size": CACHE_SIZE,
        "aggregated": agg,
        "raw_runs": all_runs,
    }
    outpath = RESULTS_DIR / "experiment_5_llm_repeated.json"
    with open(outpath, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n→ Saved to {outpath}")


if __name__ == "__main__":
    main()
