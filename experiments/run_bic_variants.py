"""Focused real-LLM validation: can an improved BIC match/beat the combined
LRU+Summary+AncestorWalk+Pin baseline?

Reuses the EXACT task, executor, harness and metric of run_llm_50agents.py so
numbers are directly comparable to the main baseline table. Tests four BIC
variants (the contribution ablation) plus the combined baseline (target) and
unbounded (ceiling), at the two cache regimes where the gap is largest.

  bic-orig    : legacy fold into immediate cached parent only
  bic-aw      : + eviction-time ancestor-walk fold (no orphan drops)
  bic-pin     : + active-path pinning (depth<=1 spine never evicted)
  bic-aw-pin  : + both

Usage:
    export GEMINI_API_KEY=...
    python -m experiments.run_bic_variants
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
for p in (_PROJECT_ROOT, _PROJECT_ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from bounded_infinity.adapters.llm_client import (
    get_usage_tracker, make_llm_callable, reset_usage_tracker,
)
from experiments.baselines import BICBackend, make_backend
from experiments.harness import run_instrumented as _run_base
from experiments.research_decomposition import TaskConfig

RESULTS_DIR = Path(__file__).resolve().parent / "results"
BRANCHING_FACTOR, MAX_DEPTH = 3, 3
EXPECTED_AGENTS = sum(BRANCHING_FACTOR ** d for d in range(MAX_DEPTH + 1))  # 40
CACHE_SIZES = [8, 16]
SEEDS = [42, 123, 7]
MAX_LLM_CALLS = 150
QUESTION = "Survey the state of memory management in multi-agent LLM systems"

# (label, builder(cache, executor)) — BIC variants are the contribution ablation.
VARIANTS = [
    ("bic-orig",   lambda c, e: BICBackend(cache_size=c, executor=e,
                                            eviction_ancestor_walk=False, pin_depth=0)),
    ("bic-aw",     lambda c, e: BICBackend(cache_size=c, executor=e,
                                            eviction_ancestor_walk=True, pin_depth=0)),
    ("bic-pin",    lambda c, e: BICBackend(cache_size=c, executor=e,
                                            eviction_ancestor_walk=False, pin_depth=1)),
    ("bic-aw-pin", lambda c, e: BICBackend(cache_size=c, executor=e,
                                            eviction_ancestor_walk=True, pin_depth=1)),
    ("lru-summary-aw-pin", lambda c, e: make_backend("lru-summary-aw-pin", capacity=c, executor=e)),
    ("unbounded",  lambda c, e: make_backend("unbounded", executor=e)),
]


def _make_executor(llm):
    def executor(agent_id, task, state, spawn):
        result = dict(state)
        question = task.get("question", "")
        prompt = (f"You are a research agent investigating this question:\n{question}\n\n"
                  f"Provide a concise, factual answer in 2-3 sentences.")
        result["response"] = llm(prompt)
        result["question"] = question
        result["depth"] = task.get("depth", 0)
        result["completed"] = True
        result["_llm_generated"] = True
        return result
    return executor


def main():
    reset_usage_tracker()
    flat = []
    total = len(CACHE_SIZES) * len(SEEDS) * len(VARIANTS)
    done = 0
    print(f"BIC-variants validation: {total} runs "
          f"(caches={CACHE_SIZES}, seeds={SEEDS}, variants={[v[0] for v in VARIANTS]})")
    for cache in CACHE_SIZES:
        print(f"\n=== cache={cache} ({cache/EXPECTED_AGENTS:.0%}) ===")
        for seed in SEEDS:
            for label, build in VARIANTS:
                llm = make_llm_callable(provider="gemini", max_tokens=150,
                                        max_calls=MAX_LLM_CALLS)
                cfg = TaskConfig(branching_factor=BRANCHING_FACTOR, max_depth=MAX_DEPTH,
                                 response_tokens=150, mode="llm", llm_callable=llm,
                                 seed=seed, question=QUESTION)
                backend = build(cache, _make_executor(llm))
                t0 = time.perf_counter()
                m = _run_base(label, backend, cfg, log_text_pairs=True)
                dt = time.perf_counter() - t0
                r = m.to_dict()
                r["seed"] = seed
                r["cache_size"] = cache
                r["wall_time_s"] = round(dt, 2)
                flat.append(r)
                done += 1
                print(f"  [{done:2d}/{total}] {label:20s} seed={seed} "
                      f"sem={m.semantic_reconstruction_quality:.3f} "
                      f"qs={m.query_success_rate:.3f} ({dt:.0f}s)")
                RESULTS_DIR.mkdir(parents=True, exist_ok=True)
                with open(RESULTS_DIR / "bic_variants_partial.json", "w") as f:
                    json.dump(flat, f, indent=2)

    tr = get_usage_tracker()
    print(f"\nLLM usage: {tr.total_calls} calls, {tr.errors} errors")
    with open(RESULTS_DIR / "bic_variants_raw.json", "w") as f:
        json.dump(flat, f, indent=2)

    # Aggregate
    import statistics as st
    from collections import defaultdict
    agg = defaultdict(lambda: defaultdict(list))
    for r in flat:
        q = r.get("quality", {})
        agg[(r["backend"], r["cache_size"])]["sem"].append(q.get("semantic_reconstruction_quality"))
        agg[(r["backend"], r["cache_size"])]["qs"].append(q.get("query_success_rate"))
    print("\n" + "=" * 70)
    print(f"{'variant':20s} {'cache':>6s} {'sem_recon':>16s} {'qsucc':>8s}")
    print("-" * 70)
    for cache in CACHE_SIZES:
        comb = st.mean(agg[("lru-summary-aw-pin", cache)]["sem"])
        for label, _ in VARIANTS:
            a = agg[(label, cache)]
            sm = st.mean(a["sem"]); ss = st.pstdev(a["sem"]); qm = st.mean(a["qs"])
            tag = ""
            if label.startswith("bic"):
                tag = "  >=combined" if sm >= comb else f"  gap {comb-sm:+.3f}"
            print(f"{label:20s} {cache:6d}   {sm:.3f}+/-{ss:.3f}     {qm:.3f}{tag}")
        print("-" * 70)


if __name__ == "__main__":
    main()
