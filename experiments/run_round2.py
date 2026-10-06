"""Round-2 (Oct 2026) LLM experiments for the Scientific Reports revision.

One driver for every LLM number in the revised manuscript, so all of them come
from a single code version, a single model configuration and the same seeds.

Experiments
-----------
smoke  : a handful of direct GPT-5.4 calls plus one tiny tree; checks the model
         configuration (empty answers, finish reasons, served model) before
         spending hours on the sweep.
sweep  : 50-agent research decomposition (b=3, d=3, 40 agents).
         {bic, bic-aw, lru, lru-summary-aw-pin} x cache {8,16,32} x 5 seeds,
         plus unbounded x 5 seeds (unbounded does not depend on cache size).
hash   : Cantor-vs-hash ablation. bic-hash x cache {8,16,32} x the same 5 seeds;
         compared against the sweep's bic-aw runs (same code, same seeds).
pilot  : coding pilot (b=3, d=2, 13 agents). {bic-aw, lru, lru-summary-aw-pin}
         x cache {2,4,8} x 3 seeds, plus unbounded x 3 seeds.

Arm definitions (explicit; make_backend("bic") defaults to ancestor-walk ON,
which is bic-aw, so "bic" here pins the original behaviour):
  bic                : original BIC, eviction folds into immediate cached parent
  bic-aw             : BIC + eviction-time ancestor walk
  bic-hash           : bic-aw with Cantor addressing replaced by an ancestry hash
  lru                : plain LRU, evicted states lost
  lru-summary-aw-pin : combined heuristic baseline
  unbounded          : stores everything (ceiling)

Output (experiments/results/round2/)
------------------------------------
<exp>.<shard>.jsonl  one JSON record per completed run, appended as runs
                     finish (one file per shard, so parallel shards never
                     write to the same file). Re-running skips any run already
                     recorded in any shard, so a crash or a dropped SSH
                     session loses at most one run.
<exp>_raw.json       all valid records from all shards, in plan order; rewritten
                     whenever a shard finishes, complete once the last one does.
<exp>_meta.<shard>.json  model configuration, seeds, code fingerprint, environment.

A run whose LLM calls still failed after retries is NOT recorded as valid
(the error text would have been stored as the agent's answer); it is written to
<exp>_rejected.jsonl and retried on the next invocation.

Usage (on the compute server)
-----------------------------
    . ~/.gpt54_env
    export LLM_PROVIDER=openai LLM_MODEL=gpt-5.4 PYTHONHASHSEED=0
    .venv/bin/python -u -m experiments.run_round2 smoke
    .venv/bin/python -u -m experiments.run_round2 sweep --caches 8
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
for _p in (_PROJECT_ROOT, _PROJECT_ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from bounded_infinity.adapters import llm_client
from experiments.baselines import BICBackend, make_backend
from experiments.harness import run_instrumented
from experiments.research_decomposition import TaskConfig

RESULTS_DIR = _PROJECT_ROOT / "experiments" / "results" / "round2"

SEEDS_5 = [42, 123, 7, 2024, 314]
SEEDS_3 = [42, 123, 7]

RESEARCH_QUESTION = "Survey the state of memory management in multi-agent LLM systems"
CODING_SPEC = (
    "Write a Python function fibonacci(n) that returns the nth Fibonacci "
    "number. fibonacci(0) = 0, fibonacci(1) = 1. The function should run "
    "in O(n) time and not use recursion."
)


# ------------------------------------------------------------------ #
# Arms
# ------------------------------------------------------------------ #

def build_backend(arm: str, cache: int | None, executor: Callable) -> Any:
    if arm == "bic":
        return BICBackend(cache_size=cache, executor=executor,
                          eviction_ancestor_walk=False, pin_depth=0)
    if arm == "unbounded":
        return make_backend("unbounded", executor=executor)
    return make_backend(arm, capacity=cache, executor=executor)


# ------------------------------------------------------------------ #
# Executors (prompts identical to run_llm_50agents / run_llm_coding_pilot)
# ------------------------------------------------------------------ #

def research_prompt(question: str, depth: int) -> str:
    return (f"You are a research agent investigating this question:\n{question}\n\n"
            f"Provide a concise, factual answer in 2-3 sentences.")


def coding_prompt(question: str, depth: int) -> str:
    return (f"You are a sub-agent in a multi-agent code-generation pipeline.\n"
            f"Your sub-task at depth {depth}: {question}\n\n"
            f"Respond in 2-3 sentences with concrete content (not meta-commentary).")


def make_executor(llm: Callable[[str], str], prompt_fn: Callable[[str, int], str]):
    def executor(agent_id, task, state, spawn):
        result = dict(state)
        question = task.get("question", "")
        depth = task.get("depth", 0)
        result["response"] = llm(prompt_fn(question, depth))
        result["question"] = question
        result["depth"] = depth
        result["completed"] = True
        result["_llm_generated"] = True
        return result
    return executor


# ------------------------------------------------------------------ #
# Experiment plans
# ------------------------------------------------------------------ #

EXPERIMENTS: dict[str, dict[str, Any]] = {
    "sweep": dict(arms=["bic", "bic-aw", "lru", "lru-summary-aw-pin"],
                  caches=[8, 16, 32], seeds=SEEDS_5, unbounded=True,
                  branching=3, depth=3, max_tokens=150, max_calls=150,
                  question=RESEARCH_QUESTION, prompt=research_prompt),
    "hash": dict(arms=["bic-hash"], caches=[8, 16, 32], seeds=SEEDS_5, unbounded=False,
                 branching=3, depth=3, max_tokens=150, max_calls=150,
                 question=RESEARCH_QUESTION, prompt=research_prompt),
    "pilot": dict(arms=["bic-aw", "lru", "lru-summary-aw-pin"],
                  caches=[2, 4, 8], seeds=SEEDS_3, unbounded=True,
                  branching=3, depth=2, max_tokens=120, max_calls=60,
                  question=CODING_SPEC, prompt=coding_prompt),
}


def plan_units(exp: str, caches: list[int] | None = None,
               seeds: list[int] | None = None,
               include_unbounded: bool = True) -> list[tuple[str, int | None, int]]:
    """List (arm, cache, seed) units, seed-major so partial results stay balanced."""
    spec = EXPERIMENTS[exp]
    caches = caches or spec["caches"]
    seeds = seeds or spec["seeds"]
    units: list[tuple[str, int | None, int]] = []
    for seed in seeds:
        if spec["unbounded"] and include_unbounded:
            units.append(("unbounded", None, seed))
        for cache in caches:
            for arm in spec["arms"]:
                units.append((arm, cache, seed))
    return units


def unit_key(arm: str, cache: int | None, seed: int) -> str:
    return f"{arm}|{cache}|{seed}"


# ------------------------------------------------------------------ #
# Provenance
# ------------------------------------------------------------------ #

def llm_config() -> dict[str, Any]:
    base = os.environ.get("OPENAI_BASE_URL", "")
    return {
        "provider": os.environ.get("LLM_PROVIDER"),
        "model": os.environ.get("LLM_MODEL"),
        # Host only: never record keys or full URLs with query strings.
        "endpoint_host": urlparse(base).hostname if base else None,
        "reasoning_effort": os.environ.get("LLM_REASONING_EFFORT") or "provider default",
        "temperature": "provider default (GPT-5.x rejects overrides)",
        "max_retries": int(os.environ.get("LLM_MAX_RETRIES", "4")),
    }


def code_fingerprint() -> str:
    """SHA-256 over the source and experiment code (the server copy has no .git)."""
    h = hashlib.sha256()
    for sub in ("src/bounded_infinity", "experiments"):
        for path in sorted((_PROJECT_ROOT / sub).rglob("*.py")):
            h.update(str(path.relative_to(_PROJECT_ROOT)).encode())
            h.update(path.read_bytes())
    return h.hexdigest()


def preflight() -> None:
    problems = []
    if not os.environ.get("LLM_PROVIDER") or not os.environ.get("LLM_MODEL"):
        problems.append("set LLM_PROVIDER and LLM_MODEL explicitly (no silent defaults)")
    if os.environ.get("PYTHONHASHSEED") is None:
        problems.append("set PYTHONHASHSEED (builtin hash() feeds BIC's Hilbert index)")
    try:
        import sklearn  # noqa: F401  (without it every semantic score is silently 0)
    except ImportError:
        problems.append("scikit-learn is not installed; semantic scores would all be 0")
    if problems:
        sys.exit("preflight failed:\n  - " + "\n  - ".join(problems))


def write_meta(exp: str, shard: str, units: list) -> None:
    spec = {k: v for k, v in EXPERIMENTS.get(exp, {}).items() if k != "prompt"}
    meta = {
        "experiment": exp,
        "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "llm": llm_config(),
        "spec": spec,
        "units": [unit_key(*u) for u in units],
        "code_sha256": code_fingerprint(),
        "python": platform.python_version(),
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        "host": platform.node(),
    }
    path = RESULTS_DIR / f"{exp}_meta.{shard}.json"
    if path.exists():
        # Keep the original start; append resumes so the history is visible.
        old = json.loads(path.read_text())
        old.setdefault("resumes", []).append(meta)
        meta = old
    path.write_text(json.dumps(meta, indent=2))


# ------------------------------------------------------------------ #
# Running
# ------------------------------------------------------------------ #

def run_unit(exp: str, arm: str, cache: int | None, seed: int) -> dict[str, Any]:
    spec = EXPERIMENTS[exp]
    llm_client.reset_usage_tracker()
    llm = llm_client.make_llm_callable(
        provider=os.environ["LLM_PROVIDER"], model=os.environ["LLM_MODEL"],
        max_tokens=spec["max_tokens"], max_calls=spec["max_calls"])
    executor = make_executor(llm, spec["prompt"])
    cfg = TaskConfig(branching_factor=spec["branching"], max_depth=spec["depth"],
                     response_tokens=spec["max_tokens"], mode="llm",
                     llm_callable=llm, seed=seed, question=spec["question"])
    backend = build_backend(arm, cache, executor)
    t0 = time.perf_counter()
    metrics = run_instrumented(arm, backend, cfg, log_text_pairs=True)
    record = metrics.to_dict()
    record.update({
        "experiment": exp,
        "seed": seed,
        "cache_size": cache,
        "wall_time_s": round(time.perf_counter() - t0, 2),
        "llm_usage": llm_client.get_usage_tracker().summary(),
        "llm": llm_config(),
        "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    })
    record["valid"] = record["llm_usage"]["errors"] == 0
    return record


def load_done(exp: str) -> dict[str, dict]:
    done: dict[str, dict] = {}
    for path in sorted(RESULTS_DIR.glob(f"{exp}.*.jsonl")):
        for line in path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done[unit_key(r["backend"], r["cache_size"], r["seed"])] = r
    return done


def run_experiment(exp: str, caches: list[int] | None, seeds: list[int] | None,
                   include_unbounded: bool) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    units = plan_units(exp, caches, seeds, include_unbounded)
    shard = "-".join(
        [f"c{'_'.join(map(str, caches))}" if caches else "call",
         f"s{'_'.join(map(str, seeds))}" if seeds else "sall"]
        + ([] if include_unbounded else ["nounb"]))
    out = RESULTS_DIR / f"{exp}.{shard}.jsonl"
    rejected = RESULTS_DIR / f"{exp}_rejected.{shard}.jsonl.txt"
    done = load_done(exp)
    todo = [u for u in units if unit_key(*u) not in done]
    write_meta(exp, shard, units)
    print(f"[{exp}] {len(units)} units, {len(units) - len(todo)} already done, "
          f"{len(todo)} to run. LLM: {llm_config()}", flush=True)

    for i, (arm, cache, seed) in enumerate(todo, 1):
        r = run_unit(exp, arm, cache, seed)
        u = r["llm_usage"]
        q = r["quality"]
        status = "ok" if r["valid"] else "REJECTED (LLM errors)"
        print(f"  [{i}/{len(todo)}] {arm:20s} cache={cache} seed={seed} "
              f"sem={q['semantic_reconstruction_quality']:.3f} "
              f"nonempty={q['nonempty_success_rate']:.3f} "
              f"empty_gen={r['task']['empty_generations']} "
              f"calls={u['total_calls']} retries={u['retries']} errors={u['errors']} "
              f"({r['wall_time_s']:.0f}s) {status}", flush=True)
        with open(out if r["valid"] else rejected, "a") as f:
            f.write(json.dumps(r) + "\n")

    done = load_done(exp)
    mine = sum(unit_key(*u) in done for u in units)
    full = plan_units(exp)
    records = [done[unit_key(*u)] for u in full if unit_key(*u) in done]
    (RESULTS_DIR / f"{exp}_raw.json").write_text(json.dumps(records, indent=2))
    print(f"[{exp}/{shard}] finished: {mine}/{len(units)} valid runs in this shard; "
          f"{len(records)}/{len(full)} across the whole experiment"
          + ("" if mine == len(units) else " (re-run the same command for the rest)"),
          flush=True)


def smoke(n_calls: int = 6) -> None:
    """Probe the model configuration before the long runs."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {"llm": llm_config(), "probes": {}}
    for label, prompt_fn, max_tokens in (("research", research_prompt, 150),
                                         ("coding", coding_prompt, 120)):
        llm_client.reset_usage_tracker()
        llm = llm_client.make_llm_callable(
            provider=os.environ["LLM_PROVIDER"], model=os.environ["LLM_MODEL"],
            max_tokens=max_tokens, max_calls=n_calls)
        texts, t0 = [], time.perf_counter()
        for i in range(n_calls):
            texts.append(llm(prompt_fn(f"{RESEARCH_QUESTION} (aspect {i})", 1)))
        report["probes"][label] = {
            "max_completion_tokens": max_tokens,
            "usage": llm_client.get_usage_tracker().summary(),
            "mean_latency_s": round((time.perf_counter() - t0) / n_calls, 2),
            "mean_words": round(sum(len(t.split()) for t in texts) / n_calls, 1),
            "samples": texts[:2],
        }
    # One tiny end-to-end tree through the real harness (b=2, d=1: 3 agents).
    llm_client.reset_usage_tracker()
    llm = llm_client.make_llm_callable(provider=os.environ["LLM_PROVIDER"],
                                       model=os.environ["LLM_MODEL"],
                                       max_tokens=150, max_calls=10)
    cfg = TaskConfig(branching_factor=2, max_depth=1, response_tokens=150, mode="llm",
                     llm_callable=llm, seed=42, question=RESEARCH_QUESTION)
    m = run_instrumented("unbounded", build_backend("unbounded", None,
                         make_executor(llm, research_prompt)), cfg)
    report["tree_unbounded_semantic"] = m.semantic_reconstruction_quality
    report["tree_unbounded_nonempty"] = m.nonempty_success_rate
    (RESULTS_DIR / "smoke.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    bad = [k for k, p in report["probes"].items()
           if p["usage"]["errors"] or p["usage"]["empty_responses"]]
    if bad or m.semantic_reconstruction_quality < 0.999:
        sys.exit(f"SMOKE FAILED: probes with errors/empty answers: {bad}; "
                 f"unbounded semantic={m.semantic_reconstruction_quality:.3f} (expected 1.0)")
    print("SMOKE OK")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("experiment", choices=["smoke", *EXPERIMENTS])
    ap.add_argument("--caches", type=int, nargs="+", help="subset of cache sizes (sharding)")
    ap.add_argument("--seeds", type=int, nargs="+", help="subset of seeds")
    ap.add_argument("--no-unbounded", action="store_true",
                    help="skip unbounded units (run them in exactly one shard)")
    args = ap.parse_args(argv)
    preflight()
    if args.experiment == "smoke":
        smoke()
    else:
        run_experiment(args.experiment, args.caches, args.seeds, not args.no_unbounded)


if __name__ == "__main__":
    main()
