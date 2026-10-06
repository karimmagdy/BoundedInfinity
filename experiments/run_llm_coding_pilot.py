"""
Tier B / Item B3 — Multi-agent coding pilot (Q8).

Purpose: provide an *additional* real-LLM workload beyond the
research-decomposition headline experiment, to demonstrate that BIC's
coverage advantage is not specific to a single task family. This is an
intentionally small-scale pilot, not a full sweep — see the
SUBMISSION_NOTES "Tier B" section for scope rationale.

Design:
- Task: recursive multi-agent code generation. The root agent receives
  a small Python coding spec ("Write a function ..."). It decomposes
  into 3 sub-tasks (parse spec, implement, test); each sub-task agent
  may further decompose into 2-3 children. Tree depth 2,
  branching 3 ⇒ ~13 agents per run.
- Backends: BIC, LRU, LRU+Summary-AW (the strengthened baseline from
  Tier B / B1), Unbounded.
- Cache sizes: 2, 4, 8 (deliberately small to stress-test eviction).
- Seeds: 1 only (pilot, not full sweep).

Metric: query-success rate (proxy for whether the root has the
information it needs to compose the final answer). The reviewer
explicitly accepted "at least a small-scale pilot" as satisfying Q8.

Usage:
    export GEMINI_API_KEY="..."
    python -m experiments.run_llm_coding_pilot

Output:
    experiments/results/relogged/coding_pilot.json
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

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
from experiments.baselines import make_backend
from experiments.harness import run_instrumented
from experiments.research_decomposition import TaskConfig

RESULTS_DIR = Path(__file__).resolve().parent / "results" / "relogged"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# --------------- Pilot parameters --------------- #
BRANCHING_FACTOR = 3
MAX_DEPTH = 2
EXPECTED_AGENTS = sum(BRANCHING_FACTOR ** d for d in range(MAX_DEPTH + 1))  # 13
CACHE_SIZES = [2, 4, 8]
SEEDS = [42, 123, 7]          # reviewer R1-W3: was single-seed; now 3 seeds
SEED = SEEDS[0]               # mutated per-seed in main() (run_one reads this global)
# R1-W2/R1-W3: BIC (ancestor-walk eviction by default) vs the strongest combined
# baseline vs naive LRU vs the unbounded ceiling, on a second task family + model.
BACKENDS = ["bic", "lru", "lru-summary-aw-pin", "unbounded"]
MAX_LLM_CALLS = 60  # per backend: ~13 agents + decomposition overhead
# Provider/model env-driven (Gemini key retired -> GPT-5.4 via Azure).
PROVIDER = os.environ.get("LLM_PROVIDER", "gemini")
MODEL = os.environ.get("LLM_MODEL") or None

CODING_SPEC = (
    "Write a Python function fibonacci(n) that returns the nth Fibonacci "
    "number. fibonacci(0) = 0, fibonacci(1) = 1. The function should run "
    "in O(n) time and not use recursion."
)


def _make_coding_executor(llm_callable):
    """Coding-task executor: each agent receives a sub-task and produces text."""

    def executor(agent_id: str, task: dict[str, Any],
                 state: dict[str, Any], spawn) -> dict[str, Any]:
        result = dict(state)
        question = task.get("question", "")
        depth = task.get("depth", 0)
        prompt = (
            f"You are a sub-agent in a multi-agent code-generation pipeline.\n"
            f"Your sub-task at depth {depth}: {question}\n\n"
            f"Respond in 2-3 sentences with concrete content (not meta-commentary)."
        )
        try:
            response = llm_callable(prompt)
        except Exception as exc:  # noqa: BLE001
            response = f"[LLM Error: {type(exc).__name__}: {exc}]"
        result["response"] = response
        result["question"] = question
        result["depth"] = depth
        result["completed"] = True
        return result

    return executor


def run_one(backend_name: str, cache_size: int,
            llm_callable) -> dict[str, Any]:
    """Run one (backend, cache_size) configuration and return RunMetrics dict."""
    label = f"{backend_name}_codingpilot_M{cache_size}_seed{SEED}"
    print(f"\n  {backend_name:18s} (M={cache_size}, seed={SEED}) ...",
          end="", flush=True)

    executor = _make_coding_executor(llm_callable)
    backend = make_backend(backend_name, capacity=cache_size, executor=executor)
    config = TaskConfig(
        branching_factor=BRANCHING_FACTOR,
        max_depth=MAX_DEPTH,
        response_tokens=80,
        mode="real_llm",
        seed=SEED,
        question=CODING_SPEC,
    )

    t0 = time.perf_counter()
    metrics = run_instrumented(backend_name, backend, config,
                               log_text_pairs=True)
    elapsed = time.perf_counter() - t0

    out = metrics.to_dict()
    out["seed"] = SEED
    out["cache_size"] = cache_size
    out["wall_time_s"] = round(elapsed, 2)
    out["task_family"] = "coding_pilot"

    qs = metrics.query_success_rate
    sem = metrics.semantic_reconstruction_quality
    print(f" qs={qs:.3f} sem={sem:.3f} ({elapsed:.0f}s)")
    return out


def main():
    global SEED
    # api_key resolved from the provider's env var inside make_llm_callable
    # (OPENAI_API_KEY for openai/GPT-5.4, GEMINI_API_KEY for gemini).

    print("=" * 60)
    print("Tier B / B3: Multi-agent coding pilot")
    print(f"  Tree: branching={BRANCHING_FACTOR}, depth={MAX_DEPTH} ~> {EXPECTED_AGENTS} agents")
    print(f"  Cache sizes: {CACHE_SIZES}")
    print(f"  Backends:    {BACKENDS}")
    print(f"  Seed:        {SEED}")
    print(f"  Provider:    {PROVIDER} ({MODEL or 'default'})")
    print(f"  Spec:        {CODING_SPEC[:60]}...")
    print("=" * 60)

    all_results = []
    for seed_val in SEEDS:
        SEED = seed_val  # run_one reads this module global and stamps it into results
        print(f"\n{'='*60}\nSEED {seed_val}\n{'='*60}")
        for cache_size in CACHE_SIZES:
            print(f"\nCache size: {cache_size}")
            for backend_name in BACKENDS:
                llm = make_llm_callable(provider=PROVIDER, model=MODEL, max_tokens=120,
                                        max_calls=MAX_LLM_CALLS)
                try:
                    result = run_one(backend_name, cache_size, llm)
                    all_results.append(result)
                except Exception as exc:  # noqa: BLE001
                    print(f" FAILED: {exc}")
                    import traceback; traceback.print_exc()
        # Save partial after each seed (crash-safety)
        (RESULTS_DIR / "coding_pilot_partial.json").write_text(json.dumps(all_results, indent=2))

    out_path = RESULTS_DIR / "coding_pilot.json"
    out_path.write_text(json.dumps(all_results, indent=2))
    print(f"\n[saved] {out_path}")
    print(f"  {len(all_results)} results across {len(BACKENDS)} backends x {len(CACHE_SIZES)} cache sizes")


if __name__ == "__main__":
    main()
