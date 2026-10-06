"""Validate the full-chain reconstruction fix: does walking past empty cached
ancestors (reconstruct_full_chain=True) drop BIC's empty reconstructions and
lift per-query quality toward the combined baseline?

bic-aw         : legacy reconstruction (stops at first cached ancestor)
bic-fullchain  : + reconstruct_full_chain (aggregate every cached ancestor)
Compare empties + quality; combined baseline numbers come from bic_confirm.
"""
from __future__ import annotations
import json, os, sys, time
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent
for p in (_ROOT, _ROOT / "src"):
    if str(p) not in sys.path: sys.path.insert(0, str(p))
from bounded_infinity.adapters.llm_client import get_usage_tracker, make_llm_callable, reset_usage_tracker
from experiments.baselines import BICBackend
from experiments.harness import run_instrumented as _run
from experiments.research_decomposition import TaskConfig

RESULTS_DIR = _ROOT / "experiments" / "results"
CACHE_SIZES = [8, 16, 32]
SEEDS = [42, 123, 7]
Q = "Survey the state of memory management in multi-agent LLM systems"
PROVIDER = os.environ.get("LLM_PROVIDER", "gemini")
MODEL = os.environ.get("LLM_MODEL") or None

VARIANTS = [
    ("bic-aw",        lambda c, e: BICBackend(cache_size=c, executor=e,
                                              eviction_ancestor_walk=True, reconstruct_full_chain=False)),
    ("bic-fullchain", lambda c, e: BICBackend(cache_size=c, executor=e,
                                              eviction_ancestor_walk=True, reconstruct_full_chain=True)),
]


def _exec(llm):
    def ex(a, t, s, sp):
        r = dict(s)
        r["response"] = llm(f"You are a research agent investigating this question:\n{t.get('question','')}\n\nProvide a concise, factual answer in 2-3 sentences.")
        r["question"] = t.get("question", ""); r["depth"] = t.get("depth", 0); r["completed"] = True
        return r
    return ex


def main():
    reset_usage_tracker(); flat = []
    total = len(CACHE_SIZES) * len(SEEDS) * len(VARIANTS); done = 0
    print(f"fullchain test: {total} runs (provider={PROVIDER} {MODEL})")
    for cache in CACHE_SIZES:
        for seed in SEEDS:
            for label, build in VARIANTS:
                llm = make_llm_callable(provider=PROVIDER, model=MODEL, max_tokens=150, max_calls=150)
                cfg = TaskConfig(branching_factor=3, max_depth=3, response_tokens=150,
                                 mode="llm", llm_callable=llm, seed=seed, question=Q)
                m = _run(label, build(cache, _exec(llm)), cfg, log_text_pairs=True)
                r = m.to_dict(); r["seed"] = seed; r["cache_size"] = cache
                # count empty reconstructions from logged pairs
                pairs = r.get("text_pairs") or []
                empt = sum(1 for p in pairs if not (p.get("reconstructed") or "").strip())
                flat.append(r); done += 1
                print(f"  [{done:2d}/{total}] {label:14s} c={cache} seed={seed} sem={m.semantic_reconstruction_quality:.3f} empty={empt}/{len(pairs)}")
                RESULTS_DIR.mkdir(parents=True, exist_ok=True)
                (RESULTS_DIR / "bic_fullchain_partial.json").write_text(json.dumps(flat, indent=2))
    (RESULTS_DIR / "bic_fullchain_raw.json").write_text(json.dumps(flat, indent=2))
    print(f"\nLLM usage: {get_usage_tracker().total_calls} calls, {get_usage_tracker().errors} errors")
    print(f"[saved] {RESULTS_DIR/'bic_fullchain_raw.json'}")


if __name__ == "__main__":
    main()
