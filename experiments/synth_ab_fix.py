"""Synthetic (no-API) A/B: does the eviction ancestor-walk fix close the
gap between BIC and the combined LRU+Summary+AW+Pin baseline?

Same harness (run_instrumented) and metric (semantic_reconstruction_quality)
as the real LLM experiment, but mode='synthetic' (deterministic seeded text).
"""
import statistics as st
from collections import defaultdict

from experiments.baselines import make_backend
from experiments.harness import run_instrumented
from experiments.research_decomposition import TaskConfig


def _noop(agent_id, task, state, spawn):
    return {**state, "status": "done"}


SEEDS = [42, 123, 7]
SPECS = [
    ("BIC-fixed (AW-evict)", lambda c: make_backend("bic", capacity=c, executor=_noop, eviction_ancestor_walk=True)),
    ("BIC-original",         lambda c: make_backend("bic", capacity=c, executor=_noop, eviction_ancestor_walk=False)),
    ("LRU+Sum+AW+Pin (comb)",lambda c: make_backend("lru-summary-aw-pin", capacity=c, executor=_noop)),
    ("LRU+Pin",              lambda c: make_backend("lru-pin", capacity=c, executor=_noop)),
    ("LRU+Summary+AW",       lambda c: make_backend("lru-summary-aw", capacity=c, executor=_noop)),
    ("LRU",                  lambda c: make_backend("lru", capacity=c, executor=_noop)),
    ("Unbounded (ceiling)",  lambda c: make_backend("unbounded", executor=_noop)),
]


def run_cache(cache):
    agg = defaultdict(lambda: defaultdict(list))
    for name, ctor in SPECS:
        for seed in SEEDS:
            backend = ctor(cache)
            cfg = TaskConfig(branching_factor=3, max_depth=3, mode="synthetic", seed=seed)
            m = run_instrumented(name, backend, cfg).to_dict()
            q = m["quality"]
            agg[name]["qs"].append(q["query_success_rate"])
            agg[name]["sem"].append(q["semantic_reconstruction_quality"])
            agg[name]["recon"].append(q["reconstruction_quality"])
    return agg


for cache in [8, 16]:
    agg = run_cache(cache)
    print(f"\n=== cache={cache} (synthetic, 3 seeds, branching=3 depth=3) ===")
    print("%-24s %8s %10s %10s" % ("backend", "qs", "sem_recon", "jaccard"))
    for name, _ in SPECS:
        a = agg[name]
        print("%-24s %8.3f %10.4f %10.4f" % (
            name, st.mean(a["qs"]), st.mean(a["sem"]), st.mean(a["recon"])))
    fixed = st.mean(agg["BIC-fixed (AW-evict)"]["sem"])
    comb = st.mean(agg["LRU+Sum+AW+Pin (comb)"]["sem"])
    orig = st.mean(agg["BIC-original"]["sem"])
    print(f"--> BIC-fixed {fixed:.4f} vs combined {comb:.4f} vs BIC-orig {orig:.4f}  "
          f"| fixed beats combined: {fixed >= comb}  (orig beat combined: {orig >= comb})")
