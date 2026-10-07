"""Working-state memory vs swarm size N (synthetic, no LLM calls).

For each tree depth d (N = (3^(d+1) - 1) / 2 agents) and each backend, run the
research-decomposition swarm breadth-first and record the peak traced memory
during the run. Unlike run_research_task, this loop keeps no per-agent answers,
so the measurement is what the backend itself holds: Unbounded grows with every
stored response, the bounded backends only with their per-agent bookkeeping.

    PYTHONHASHSEED=0 .venv/bin/python -m experiments.run_memory_scaling
"""
from __future__ import annotations

import json
import sys
import time
import tracemalloc
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from experiments.baselines import make_backend
from experiments.research_decomposition import _decompose_question, research_executor

QUESTION = "Survey the state of memory management in multi-agent LLM systems"
BACKENDS = ["bic-aw", "lru", "lru-summary-aw-pin", "unbounded"]
DEPTHS = [3, 4, 5, 6, 7, 8]
CACHE = 64
BRANCHING = 3
OUT = _PROJECT_ROOT / "experiments" / "results" / "synthetic_fixed" / "memory_scaling.json"


def run(backend_name: str, depth: int) -> dict:
    backend = make_backend(backend_name, capacity=CACHE, executor=research_executor, seed=42)
    tracemalloc.start()
    t0 = time.perf_counter()
    root = backend.spawn(task={"question": QUESTION, "depth": 0, "max_depth": depth})
    queue = [(root, QUESTION, 0)]
    n = 1
    while queue:
        agent_id, question, d = queue.pop(0)
        backend.execute(agent_id)          # result deliberately not kept
        if d < depth:
            for sub_q in _decompose_question(question, BRANCHING, hash(agent_id) & 0x7FFFFFFF):
                child = backend.spawn(parent_id=agent_id,
                                      task={"question": sub_q, "depth": d + 1, "max_depth": depth})
                queue.append((child, sub_q, d + 1))
                n += 1
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {"backend": backend_name, "depth": depth, "agents": n, "cache_size": CACHE,
            "run_peak_bytes": peak, "final_bytes": current,
            "seconds": round(time.perf_counter() - t0, 2)}


def main() -> None:
    results = []
    for depth in DEPTHS:
        for name in BACKENDS:
            r = run(name, depth)
            results.append(r)
            print(f"{name:20s} N={r['agents']:6d} peak={r['run_peak_bytes'] / 1024:9.0f} KB "
                  f"final={r['final_bytes'] / 1024:9.0f} KB ({r['seconds']}s)", flush=True)
            OUT.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
