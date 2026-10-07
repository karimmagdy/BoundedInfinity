"""
Experiment runner — configures and executes all four experiments.

Experiments:
  1. Scale:    Synthetic agents, varying cache sizes and agent counts
  2. Quality:  Synthetic agents, measure reconstruction quality vs baselines
  3. Ablation: BIC with/without Hilbert locality and summarization
  4. Stress:   50,000+ synthetic agents, cache=64

Usage:
  python -m experiments.run_experiments                 # run all
  python -m experiments.run_experiments --experiment 1  # run one
  python -m experiments.run_experiments --quick          # small configs for CI
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

# Ensure project root is on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from experiments.baselines import BICBackend, make_backend
from experiments.harness import RunMetrics, run_instrumented
from experiments.research_decomposition import TaskConfig, research_executor

RESULTS_DIR = Path(__file__).resolve().parent / "results"


# ------------------------------------------------------------------ #
# Experiment 1: Scale
# ------------------------------------------------------------------ #

def experiment_scale(quick: bool = False) -> list[dict[str, Any]]:
    """Synthetic agents at varying scales. BIC vs all baselines.

    Measures: memory, latency, info preservation.
    """
    print("\n" + "=" * 60)
    print("Experiment 1: Scale")
    print("=" * 60)

    if quick:
        cache_sizes = [8, 16]
        depths = [2, 3]
    else:
        cache_sizes = [8, 16, 32, 64, 128]
        depths = [3, 4, 5]

    backend_names = ["bic", "lru", "lru-summary", "tiered", "fifo", "random", "unbounded"]
    branching = 3
    results: list[dict[str, Any]] = []

    for depth in depths:
        for cache_size in cache_sizes:
            for bname in backend_names:
                config = TaskConfig(
                    branching_factor=branching,
                    max_depth=depth,
                    response_tokens=256,
                    seed=42,
                )
                # Expected agents: (k^(d+1) - 1) / (k - 1)
                expected = (branching ** (depth + 1) - 1) // (branching - 1)

                print(f"  {bname:10s} | cache={cache_size:4d} | depth={depth} | ~{expected} agents ... ",
                      end="", flush=True)

                backend = make_backend(
                    bname, capacity=cache_size,
                    executor=research_executor, seed=42,
                )
                metrics = run_instrumented(bname, backend, config)
                results.append(metrics.to_dict())

                print(f"mem={metrics.peak_memory_bytes // 1024:6d}KB "
                      f"evict={metrics.total_evictions:5d} "
                      f"recon={metrics.reconstruction_quality:.3f}")

    return results


# ------------------------------------------------------------------ #
# Experiment 2: Quality
# ------------------------------------------------------------------ #

def experiment_quality(quick: bool = False) -> list[dict[str, Any]]:
    """Measure reconstruction quality across cache sizes.

    All backends get the same task; we measure how much of the original
    agent answers are recoverable via query().
    """
    print("\n" + "=" * 60)
    print("Experiment 2: Quality")
    print("=" * 60)

    if quick:
        cache_sizes = [8, 16, 32]
    else:
        cache_sizes = [8, 16, 32, 64, 128, 256]

    backend_names = ["bic", "lru", "lru-summary", "tiered", "fifo", "random", "unbounded"]
    results: list[dict[str, Any]] = []

    config = TaskConfig(
        branching_factor=3,
        max_depth=4,
        response_tokens=512,
        seed=42,
    )
    expected = (3 ** 5 - 1) // 2  # 121 agents

    for cache_size in cache_sizes:
        for bname in backend_names:
            print(f"  {bname:10s} | cache={cache_size:4d} | {expected} agents ... ",
                  end="", flush=True)

            backend = make_backend(
                bname, capacity=cache_size,
                executor=research_executor, seed=42,
            )
            metrics = run_instrumented(bname, backend, config)
            results.append(metrics.to_dict())

            print(f"quality={metrics.reconstruction_quality:.3f} "
                  f"hit={metrics.cache_hit_rate:.3f} "
                  f"success={metrics.query_success_rate:.3f}")

    return results


# ------------------------------------------------------------------ #
# Experiment 3: Ablation
# ------------------------------------------------------------------ #

def experiment_ablation(quick: bool = False) -> list[dict[str, Any]]:
    """BIC variants: with/without Hilbert locality, with/without summarization.

    Uses the same task config for all; varies BIC parameters.
    """
    print("\n" + "=" * 60)
    print("Experiment 3: Ablation")
    print("=" * 60)

    config = TaskConfig(
        branching_factor=3,
        max_depth=4 if not quick else 3,
        response_tokens=512,
        seed=42,
    )
    cache_size = 32

    variants = [
        # (name, state_dimensions, use_summarization)
        ("bic-full", 3, True),
        ("bic-no-hilbert", 0, True),   # No Hilbert (hash fallback)
        ("bic-no-summary", 3, False),  # Hilbert but no summarization
        ("bic-minimal", 0, False),     # Neither
    ]

    results: list[dict[str, Any]] = []

    for vname, state_dims, use_summary in variants:
        print(f"  {vname:18s} | dims={state_dims} | summary={use_summary} ... ",
              end="", flush=True)

        kwargs: dict[str, Any] = {"state_dimensions": state_dims}
        if not use_summary:
            # Use a no-op summarizer that just keeps the parent unchanged
            kwargs["summarize"] = _noop_summarize

        backend = BICBackend(
            cache_size=cache_size,
            executor=research_executor,
            **kwargs,
        )
        metrics = run_instrumented(vname, backend, config)
        results.append(metrics.to_dict())

        print(f"recon={metrics.reconstruction_quality:.3f} "
              f"pres={metrics.info_preservation_ratio:.3f}")

    return results


def _noop_summarize(parent_state: dict[str, Any],
                    child_states: list[dict[str, Any]]) -> dict[str, Any]:
    """No-op summarizer: just return parent unchanged (children are lost)."""
    return dict(parent_state)


# ------------------------------------------------------------------ #
# Experiment 4: Stress
# ------------------------------------------------------------------ #

def experiment_stress(quick: bool = False) -> list[dict[str, Any]]:
    """Stress test: many agents, small cache. BIC stays flat; others crash or OOM.

    The unbounded backend is excluded (it would allocate too much memory).
    """
    print("\n" + "=" * 60)
    print("Experiment 4: Stress")
    print("=" * 60)

    if quick:
        max_depth = 5  # 3^6 ≈ 364 agents
    else:
        max_depth = 7  # 3^8 ≈ 3280 agents

    config = TaskConfig(
        branching_factor=3,
        max_depth=max_depth,
        response_tokens=256,
        seed=42,
    )

    cache_size = 64
    backend_names = ["bic", "lru", "lru-summary", "tiered", "fifo", "random"]
    results: list[dict[str, Any]] = []

    expected = (3 ** (max_depth + 1) - 1) // 2

    for bname in backend_names:
        print(f"  {bname:10s} | cache={cache_size} | ~{expected} agents ... ",
              end="", flush=True)

        backend = make_backend(
            bname, capacity=cache_size,
            executor=research_executor, seed=42,
        )
        t0 = time.perf_counter()
        metrics = run_instrumented(bname, backend, config)
        wall_time = time.perf_counter() - t0
        results.append(metrics.to_dict())

        print(f"run_mem={metrics.run_peak_memory_bytes // 1024:6d}KB "
              f"mem={metrics.peak_memory_bytes // 1024:6d}KB "
              f"time={wall_time:.2f}s "
              f"recon={metrics.reconstruction_quality:.3f}")

    return results


# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #

def main() -> None:
    parser = argparse.ArgumentParser(description="BoundedInfinity experiments")
    parser.add_argument("--experiment", type=int, default=0,
                        help="Run a specific experiment (1-4). 0 = all.")
    parser.add_argument("--quick", action="store_true",
                        help="Use smaller configs for fast CI testing")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_results: dict[str, Any] = {}
    experiments = {
        1: ("scale", experiment_scale),
        2: ("quality", experiment_quality),
        3: ("ablation", experiment_ablation),
        4: ("stress", experiment_stress),
    }

    to_run = [args.experiment] if args.experiment > 0 else [1, 2, 3, 4]

    for exp_num in to_run:
        if exp_num not in experiments:
            print(f"Unknown experiment {exp_num}")
            continue
        name, func = experiments[exp_num]
        results = func(quick=args.quick)
        all_results[name] = results

        # Save per-experiment results
        outpath = RESULTS_DIR / f"experiment_{exp_num}_{name}.json"
        with open(outpath, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  → Saved {len(results)} results to {outpath.relative_to(_PROJECT_ROOT)}")

    # Save combined results
    combined_path = RESULTS_DIR / "all_results.json"
    with open(combined_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n→ All results saved to {combined_path.relative_to(_PROJECT_ROOT)}")

    # Print summary table
    _print_summary(all_results)


def _print_summary(all_results: dict[str, Any]) -> None:
    """Print a concise summary table of key results."""
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for exp_name, runs in all_results.items():
        print(f"\n--- {exp_name.upper()} ---")
        print(f"{'Backend':12s} {'Agents':>7s} {'PeakMem(KB)':>12s} "
              f"{'Evictions':>10s} {'ReconQ':>7s} {'HitRate':>8s}")
        print("-" * 60)
        for run in runs:
            bname = run.get("backend", "?")
            agents = run.get("task", {}).get("total_agents", 0)
            mem_kb = run.get("memory", {}).get("peak_bytes", 0) // 1024
            evictions = run.get("evictions", 0)
            recon = run.get("quality", {}).get("reconstruction_quality", 0)
            hit = run.get("quality", {}).get("cache_hit_rate", 0)
            print(f"{bname:12s} {agents:7d} {mem_kb:12d} "
                  f"{evictions:10d} {recon:7.3f} {hit:8.3f}")


if __name__ == "__main__":
    main()
