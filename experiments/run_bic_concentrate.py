"""Option B test: does BIC with the heuristic's concentration eviction
(pin depth<=1 + LRU-of-the-rest) tie the combined baseline on quality while
keeping Cantor addressing + summarization (i.e. determinism + guarantees)?

bic-aw            : current best BIC (depth-priority + ancestor-walk evict)
bic-concentrated  : BIC + pin_depth=1 + eviction_policy=lru (the "combine")
lru-summary-aw-pin: the combined heuristic (target)

Same harness/metric as run_bic_confirm. Env-driven provider (GPT-5.4).
"""
from __future__ import annotations
import json, os, sys, time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
for p in (_ROOT, _ROOT / "src"):
    if str(p) not in sys.path: sys.path.insert(0, str(p))

from bounded_infinity.adapters.llm_client import get_usage_tracker, make_llm_callable, reset_usage_tracker
from experiments.baselines import BICBackend, make_backend
from experiments.harness import run_instrumented as _run
from experiments.research_decomposition import TaskConfig

RESULTS_DIR = _ROOT / "experiments" / "results"
BRANCHING, DEPTH = 3, 3
CACHE_SIZES = [8, 16, 32]
SEEDS = [42, 123, 7]
MAX_CALLS = 150
Q = "Survey the state of memory management in multi-agent LLM systems"
PROVIDER = os.environ.get("LLM_PROVIDER", "gemini")
MODEL = os.environ.get("LLM_MODEL") or None

VARIANTS = [
    ("bic-aw",           lambda c, e: BICBackend(cache_size=c, executor=e,
                                                  eviction_ancestor_walk=True, pin_depth=0)),
    ("bic-concentrated", lambda c, e: BICBackend(cache_size=c, executor=e,
                                                  eviction_ancestor_walk=True, pin_depth=1,
                                                  eviction_policy="lru")),
    ("lru-summary-aw-pin", lambda c, e: make_backend("lru-summary-aw-pin", capacity=c, executor=e)),
]


def _exec(llm):
    def ex(agent_id, task, state, spawn):
        r = dict(state)
        r["response"] = llm(f"You are a research agent investigating this question:\n{task.get('question','')}\n\nProvide a concise, factual answer in 2-3 sentences.")
        r["question"] = task.get("question", ""); r["depth"] = task.get("depth", 0)
        r["completed"] = True; r["_llm_generated"] = True
        return r
    return ex


def main():
    reset_usage_tracker(); flat = []
    total = len(CACHE_SIZES) * len(SEEDS) * len(VARIANTS); done = 0
    print(f"bic-concentrate test: {total} runs (caches={CACHE_SIZES}, seeds={SEEDS}, provider={PROVIDER} {MODEL})")
    for cache in CACHE_SIZES:
        for seed in SEEDS:
            for label, build in VARIANTS:
                llm = make_llm_callable(provider=PROVIDER, model=MODEL, max_tokens=150, max_calls=MAX_CALLS)
                cfg = TaskConfig(branching_factor=BRANCHING, max_depth=DEPTH, response_tokens=150,
                                 mode="llm", llm_callable=llm, seed=seed, question=Q)
                m = _run(label, build(cache, _exec(llm)), cfg, log_text_pairs=True)
                r = m.to_dict(); r["seed"] = seed; r["cache_size"] = cache
                flat.append(r); done += 1
                print(f"  [{done:2d}/{total}] {label:18s} c={cache} seed={seed} sem={m.semantic_reconstruction_quality:.3f} qs={m.query_success_rate:.3f}")
                RESULTS_DIR.mkdir(parents=True, exist_ok=True)
                (RESULTS_DIR / "bic_concentrate_partial.json").write_text(json.dumps(flat, indent=2))
    (RESULTS_DIR / "bic_concentrate_raw.json").write_text(json.dumps(flat, indent=2))
    print(f"\nLLM usage: {get_usage_tracker().total_calls} calls, {get_usage_tracker().errors} errors")
    print(f"[saved] {RESULTS_DIR/'bic_concentrate_raw.json'}")


if __name__ == "__main__":
    main()
