"""Access-order sensitivity probe (synthetic, no API calls).

Same tree content under three traversal orders (bfs / dfs / random); only
cache dynamics differ. Re-run after the Oct 2026 metric fix: the original
probe scored against a separately generated reference text.

Setup: b=3, d=4 (121 agents), cache 16, deterministic_content=True, 5 seeds.

    PYTHONHASHSEED=0 python -m experiments.run_order_probe
Output: experiments/results/round2/order_probe_raw.json
"""
from __future__ import annotations

import json
import os
import statistics as st
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
for _p in (_PROJECT_ROOT, _PROJECT_ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from experiments.harness import run_instrumented
from experiments.research_decomposition import TaskConfig, research_executor
from experiments.run_round2 import SEEDS_5, build_backend

OUT = _PROJECT_ROOT / "experiments" / "results" / "round2" / "order_probe_raw.json"
ARMS = ["bic", "bic-aw", "lru", "lru-summary-aw-pin", "unbounded"]
ORDERS = ["bfs", "dfs", "random"]
CACHE = 16


def main() -> None:
    if os.environ.get("PYTHONHASHSEED") is None:
        sys.exit("set PYTHONHASHSEED (the synthetic executor seeds text with hash())")
    records = []
    for seed in SEEDS_5:
        for order in ORDERS:
            for arm in ARMS:
                cfg = TaskConfig(branching_factor=3, max_depth=4, response_tokens=150,
                                 mode="synthetic", seed=seed, access_order=order,
                                 deterministic_content=True)
                cache = None if arm == "unbounded" else CACHE
                m = run_instrumented(arm, build_backend(arm, cache, research_executor), cfg)
                q = m.to_dict()["quality"]
                records.append({"arm": arm, "order": order, "seed": seed, "cache_size": cache,
                                "pythonhashseed": os.environ["PYTHONHASHSEED"], **q})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(records, indent=2))

    print(f"{'arm':20s} " + " ".join(f"{o:>14s}" for o in ORDERS) + "   spread  nonempty(bfs)")
    for arm in ARMS:
        means = []
        for o in ORDERS:
            v = [r["semantic_reconstruction_quality"] for r in records
                 if r["arm"] == arm and r["order"] == o]
            means.append((st.mean(v), st.pstdev(v)))
        ne = st.mean(r["nonempty_success_rate"] for r in records
                     if r["arm"] == arm and r["order"] == "bfs")
        print(f"{arm:20s} " + " ".join(f"{m:.3f}+/-{s:.3f}" for m, s in means)
              + f"   {max(m for m, _ in means) - min(m for m, _ in means):.3f}   {ne:.3f}")
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
