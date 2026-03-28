"""
Unified measurement harness for BoundedInfinity experiments.

Collects per-run metrics across all backends:
  - Memory usage over time (via tracemalloc)
  - Task completion quality (reconstruction accuracy)
  - Information preservation ratio
  - Latency per operation (spawn/execute/query)
  - Cache hit rate

All results are returned as structured dicts for JSON serialization.
"""

from __future__ import annotations

import time
import tracemalloc
from dataclasses import dataclass, field
from typing import Any

from experiments.research_decomposition import (
    SwarmBackend,
    TaskConfig,
    TaskResult,
    run_research_task,
    _synthetic_response,
)


# ------------------------------------------------------------------ #
# Metric data structures
# ------------------------------------------------------------------ #

@dataclass
class MemorySample:
    """A single memory snapshot."""
    step: int
    timestamp: float
    current_bytes: int
    peak_bytes: int
    agent_count: int  # total agents spawned so far


@dataclass
class LatencyStats:
    """Aggregated latency statistics for an operation type."""
    operation: str
    count: int = 0
    total_seconds: float = 0.0
    min_seconds: float = float("inf")
    max_seconds: float = 0.0

    @property
    def mean_seconds(self) -> float:
        return self.total_seconds / max(self.count, 1)

    def record(self, elapsed: float) -> None:
        self.count += 1
        self.total_seconds += elapsed
        self.min_seconds = min(self.min_seconds, elapsed)
        self.max_seconds = max(self.max_seconds, elapsed)

    def to_dict(self) -> dict[str, Any]:
        return {
            "operation": self.operation,
            "count": self.count,
            "total_s": round(self.total_seconds, 6),
            "mean_s": round(self.mean_seconds, 6),
            "min_s": round(self.min_seconds, 6) if self.count > 0 else None,
            "max_s": round(self.max_seconds, 6),
        }


@dataclass
class RunMetrics:
    """All metrics for a single experiment run."""
    backend_name: str
    config: dict[str, Any]
    task_result: dict[str, Any]

    # Memory
    peak_memory_bytes: int = 0
    memory_samples: list[dict[str, Any]] = field(default_factory=list)

    # Latency
    spawn_latency: dict[str, Any] = field(default_factory=dict)
    execute_latency: dict[str, Any] = field(default_factory=dict)
    query_latency: dict[str, Any] = field(default_factory=dict)

    # Quality
    cache_hit_rate: float = 0.0  # fraction of queries served from cache
    query_success_rate: float = 0.0  # fraction of queries that returned non-None
    reconstruction_quality: float = 0.0  # average token overlap for reconstructed states
    semantic_reconstruction_quality: float = 0.0  # TF-IDF cosine similarity
    info_preservation_ratio: float = 0.0  # avg ρ from BIC eviction stats

    # Eviction
    total_evictions: int = 0
    final_memory_size: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend_name,
            "config": self.config,
            "task": self.task_result,
            "memory": {
                "peak_bytes": self.peak_memory_bytes,
                "final_size": self.final_memory_size,
                "samples": self.memory_samples,
            },
            "latency": {
                "spawn": self.spawn_latency,
                "execute": self.execute_latency,
                "query": self.query_latency,
            },
            "quality": {
                "cache_hit_rate": round(self.cache_hit_rate, 4),
                "query_success_rate": round(self.query_success_rate, 4),
                "reconstruction_quality": round(self.reconstruction_quality, 4),
                "semantic_reconstruction_quality": round(self.semantic_reconstruction_quality, 4),
                "info_preservation_ratio": round(self.info_preservation_ratio, 4),
            },
            "evictions": self.total_evictions,
        }


# ------------------------------------------------------------------ #
# Instrumented backend wrapper
# ------------------------------------------------------------------ #

class InstrumentedBackend:
    """Wraps any SwarmBackend to collect latency and memory metrics."""

    def __init__(self, inner: SwarmBackend, sample_interval: int = 50) -> None:
        self._inner = inner
        self._sample_interval = sample_interval
        self._step = 0

        self.spawn_latency = LatencyStats("spawn")
        self.execute_latency = LatencyStats("execute")
        self.query_latency = LatencyStats("query")
        self.memory_samples: list[MemorySample] = []

        self._queries_total = 0
        self._queries_hit = 0  # returned non-None
        self._queries_cached = 0  # state was NOT reconstructed

    def spawn(self, parent_id: str | None = None,
              task: dict[str, Any] | None = None) -> str:
        t0 = time.perf_counter()
        result = self._inner.spawn(parent_id=parent_id, task=task)
        self.spawn_latency.record(time.perf_counter() - t0)
        self._step += 1
        self._maybe_sample()
        return result

    def execute(self, agent_id: str) -> dict[str, Any]:
        t0 = time.perf_counter()
        result = self._inner.execute(agent_id)
        self.execute_latency.record(time.perf_counter() - t0)
        self._step += 1
        self._maybe_sample()
        return result

    def query(self, agent_id: str) -> dict[str, Any] | None:
        t0 = time.perf_counter()
        result = self._inner.query(agent_id)
        self.query_latency.record(time.perf_counter() - t0)
        self._queries_total += 1
        if result is not None:
            self._queries_hit += 1
            if not result.get("_reconstructed", False):
                self._queries_cached += 1
        return result

    def terminate(self, agent_id: str) -> None:
        self._inner.terminate(agent_id)

    @property
    def cache_hit_rate(self) -> float:
        if self._queries_total == 0:
            return 1.0
        return self._queries_cached / self._queries_total

    @property
    def query_success_rate(self) -> float:
        if self._queries_total == 0:
            return 1.0
        return self._queries_hit / self._queries_total

    def _maybe_sample(self) -> None:
        if self._step % self._sample_interval == 0:
            current, peak = tracemalloc.get_traced_memory()
            self.memory_samples.append(MemorySample(
                step=self._step,
                timestamp=time.monotonic(),
                current_bytes=current,
                peak_bytes=peak,
                agent_count=self._step,
            ))


# ------------------------------------------------------------------ #
# Reconstruction quality measurement
# ------------------------------------------------------------------ #

def _token_overlap(original: str, reconstructed: str) -> float:
    """Jaccard similarity of token sets between two strings."""
    if not original and not reconstructed:
        return 1.0
    orig_tokens = set(original.lower().split())
    recon_tokens = set(reconstructed.lower().split())
    if not orig_tokens:
        return 0.0
    intersection = orig_tokens & recon_tokens
    union = orig_tokens | recon_tokens
    return len(intersection) / len(union) if union else 0.0


def _semantic_similarity(original: str, reconstructed: str) -> float:
    """TF-IDF cosine similarity between two strings.

    More meaningful than raw Jaccard: weights terms by importance
    and captures semantic overlap via cosine in TF-IDF space.
    Returns 0.0 if either string is empty or sklearn is unavailable.
    """
    if not original or not reconstructed:
        return 0.0
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        vectorizer = TfidfVectorizer()
        tfidf = vectorizer.fit_transform([original, reconstructed])
        sim = cosine_similarity(tfidf[0:1], tfidf[1:2])[0, 0]
        return float(sim)
    except ImportError:
        # sklearn not installed — fall back to Jaccard
        return _token_overlap(original, reconstructed)


def _extract_reconstructed_text(state: dict[str, Any]) -> str:
    """Extract reconstructed text from a query result."""
    if state is None:
        return ""
    if state.get("_reconstructed"):
        summaries = state.get("_ancestor_summaries", [])
        return " ".join(
            str(s) for chain in summaries for s in chain.get("summaries", [])
        )
    return state.get("response", "")


def measure_reconstruction_quality(
    backend: SwarmBackend,
    agent_answers: dict[str, str],
) -> tuple[float, float]:
    """Query every agent and measure how much of the original answer is recoverable.

    For cached agents: the state should contain the full response → high overlap.
    For evicted agents: only summaries available → lower overlap.
    For lost agents (baselines with no summarization): query returns None → 0.

    Returns (jaccard_quality, semantic_quality) averaged across all agents.
    """
    if not agent_answers:
        return 0.0, 0.0

    total_jaccard = 0.0
    total_semantic = 0.0
    count = 0

    for agent_id, original_answer in agent_answers.items():
        state = backend.query(agent_id)
        if state is None:
            pass  # 0.0 for both metrics
        elif state.get("_reconstructed"):
            reconstructed_text = _extract_reconstructed_text(state)
            total_jaccard += _token_overlap(original_answer, reconstructed_text)
            total_semantic += _semantic_similarity(original_answer, reconstructed_text)
        else:
            cached_response = state.get("response", "")
            if cached_response:
                total_jaccard += _token_overlap(original_answer, cached_response)
                total_semantic += _semantic_similarity(original_answer, cached_response)
            else:
                total_jaccard += 0.5
                total_semantic += 0.5
        count += 1

    n = max(count, 1)
    return total_jaccard / n, total_semantic / n


# ------------------------------------------------------------------ #
# Full experiment runner
# ------------------------------------------------------------------ #

def run_instrumented(
    backend_name: str,
    backend: SwarmBackend,
    config: TaskConfig,
) -> RunMetrics:
    """Run a research-decomposition task with full instrumentation.

    Wraps the backend, runs the task, measures reconstruction quality,
    and returns a RunMetrics with everything.
    """
    tracemalloc.start()
    instrumented = InstrumentedBackend(backend)

    # Run the task
    task_result = run_research_task(instrumented, config)

    # Measure reconstruction quality (query every agent)
    recon_quality, semantic_quality = measure_reconstruction_quality(
        instrumented, task_result.agent_answers
    )

    # Collect memory info
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Get BIC-specific metrics if available
    info_pres = 0.0
    total_evictions = 0
    final_size = 0

    from experiments.baselines import BICBackend
    if isinstance(backend, BICBackend):
        stats = backend.eviction_stats
        info_pres = stats.avg_preservation_ratio
        total_evictions = stats.total_evictions
        final_size = backend.memory_size
    else:
        final_size = getattr(backend, "memory_size", 0)
        total_evictions = getattr(backend, "total_evictions", 0)

    metrics = RunMetrics(
        backend_name=backend_name,
        config={
            "question": config.question,
            "branching_factor": config.branching_factor,
            "max_depth": config.max_depth,
            "response_tokens": config.response_tokens,
            "mode": config.mode,
            "seed": config.seed,
        },
        task_result={
            "total_agents": task_result.total_agents,
            "tree_depth": task_result.tree_depth,
            "elapsed_seconds": round(task_result.elapsed_seconds, 4),
            "answer_length": len(task_result.final_answer),
        },
        peak_memory_bytes=peak_mem,
        memory_samples=[
            {
                "step": s.step,
                "current_bytes": s.current_bytes,
                "peak_bytes": s.peak_bytes,
            }
            for s in instrumented.memory_samples
        ],
        spawn_latency=instrumented.spawn_latency.to_dict(),
        execute_latency=instrumented.execute_latency.to_dict(),
        query_latency=instrumented.query_latency.to_dict(),
        cache_hit_rate=instrumented.cache_hit_rate,
        query_success_rate=instrumented.query_success_rate,
        reconstruction_quality=recon_quality,
        semantic_reconstruction_quality=semantic_quality,
        info_preservation_ratio=info_pres,
        total_evictions=total_evictions,
        final_memory_size=final_size,
    )

    return metrics
