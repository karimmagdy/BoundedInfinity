"""
Experiment 5: Real LLM integration experiment.

Runs a small-scale research decomposition task with a real LLM
(Google Gemini by default — cheapest option) to validate that BIC
works end-to-end with non-synthetic responses.

Design constraints:
  - Small scale: depth=2, branching=2 → 7 agents → ~7 LLM calls per backend
  - Three backends: BIC, LRU, Unbounded (≤ 21 LLM calls total)
  - Cache size = 4 (severe constraint for 7 agents)
  - Measures both Jaccard and semantic (TF-IDF) reconstruction quality

Usage:
    # Set API key as environment variable:
    export GEMINI_API_KEY="..."
    python -m experiments.run_llm_experiment

    # Or specify provider:
    python -m experiments.run_llm_experiment --provider openai
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from bounded_infinity.adapters.llm_client import (
    get_usage_tracker,
    make_llm_callable,
    reset_usage_tracker,
)
from experiments.baselines import BICBackend, make_backend
from experiments.harness import RunMetrics, run_instrumented
from experiments.research_decomposition import TaskConfig, research_executor

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def _make_llm_executor(llm_callable):
    """Create an executor that uses a real LLM for agent responses."""

    def executor(agent_id: str, task: dict[str, Any],
                 state: dict[str, Any],
                 spawn) -> dict[str, Any]:
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


def run_llm_experiment(provider: str = "gemini",
                       api_key: str | None = None) -> list[dict[str, Any]]:
    """Run Experiment 5: real LLM integration."""
    print("\n" + "=" * 60)
    print(f"Experiment 5: LLM Integration ({provider})")
    print("=" * 60)

    reset_usage_tracker()

    expected = (2 ** 3 - 1)  # 7 agents
    cache_size = 4  # Severe constraint: 4 slots for 7 agents
    question = "Survey the state of memory management in multi-agent LLM systems"

    backend_names = ["bic", "lru", "lru-summary", "unbounded"]
    results: list[dict[str, Any]] = []

    for bname in backend_names:
        # Fresh LLM callable per backend to avoid hitting call limit
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
            seed=42,
            question=question,
        )

        if bname == "bic":
            backend = BICBackend(cache_size=cache_size, executor=llm_executor)
        elif bname == "unbounded":
            backend = make_backend("unbounded", executor=llm_executor)
        else:
            backend = make_backend(bname, capacity=cache_size, executor=llm_executor)

        print(f"  {bname:14s} | cache={cache_size} | ~{expected} agents ... ",
              end="", flush=True)

        t0 = time.perf_counter()
        metrics = run_instrumented(bname, backend, config)
        wall_time = time.perf_counter() - t0

        result = metrics.to_dict()
        result["wall_time_s"] = round(wall_time, 2)
        result["llm_provider"] = provider
        results.append(result)

        print(f"recon={metrics.reconstruction_quality:.3f} "
              f"semantic={metrics.semantic_reconstruction_quality:.3f} "
              f"success={metrics.query_success_rate:.3f} "
              f"time={wall_time:.1f}s")

    # Print usage summary
    tracker = get_usage_tracker()
    print(f"\n  LLM Usage: {tracker.total_calls} calls, "
          f"~{tracker.total_input_tokens} input tokens, "
          f"~{tracker.total_output_tokens} output tokens, "
          f"{tracker.errors} errors")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="BoundedInfinity LLM experiment")
    parser.add_argument("--provider", type=str, default="gemini",
                        choices=["gemini", "openai", "anthropic"],
                        help="LLM provider (default: gemini)")
    parser.add_argument("--api-key", type=str, default=None,
                        help="API key (overrides env var)")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    results = run_llm_experiment(provider=args.provider, api_key=args.api_key)

    outpath = RESULTS_DIR / "experiment_5_llm.json"
    with open(outpath, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n→ Saved {len(results)} results to {outpath}")

    # Print summary
    print("\n" + "=" * 70)
    print("EXPERIMENT 5 SUMMARY")
    print("=" * 70)
    print(f"{'Backend':14s} {'Recon(Jaccard)':>14s} {'Recon(Semantic)':>16s} "
          f"{'QuerySuccess':>13s} {'Evictions':>10s}")
    print("-" * 70)
    for r in results:
        q = r.get("quality", {})
        print(f"{r['backend']:14s} "
              f"{q.get('reconstruction_quality', 0):14.3f} "
              f"{q.get('semantic_reconstruction_quality', 0):16.3f} "
              f"{q.get('query_success_rate', 0):13.3f} "
              f"{r.get('evictions', 0):10d}")


if __name__ == "__main__":
    main()
