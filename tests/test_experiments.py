"""
Tests for experiment infrastructure — baselines, harness, and task runner.
"""

import pytest

from experiments.baselines import (
    BICBackend,
    FIFOBackend,
    LRUBackend,
    RandomBackend,
    UnboundedBackend,
    make_backend,
)
from experiments.research_decomposition import (
    TaskConfig,
    _decompose_question,
    _synthetic_response,
    research_executor,
    run_research_task,
)


# ------------------------------------------------------------------ #
# Synthetic text generation
# ------------------------------------------------------------------ #

def test_synthetic_response_deterministic() -> None:
    r1 = _synthetic_response("test question", 100, seed=42)
    r2 = _synthetic_response("test question", 100, seed=42)
    assert r1 == r2


def test_synthetic_response_varies_by_seed() -> None:
    r1 = _synthetic_response("test question", 100, seed=42)
    r2 = _synthetic_response("test question", 100, seed=99)
    assert r1 != r2


def test_decompose_question() -> None:
    subs = _decompose_question("memory management in agents", 3, seed=42)
    assert len(subs) == 3
    assert all(isinstance(s, str) for s in subs)


# ------------------------------------------------------------------ #
# All backends pass the same interface test
# ------------------------------------------------------------------ #

@pytest.mark.parametrize("backend_name", ["bic", "unbounded", "lru", "fifo", "random"])
def test_backend_interface(backend_name: str) -> None:
    """Every backend should support spawn/execute/query/terminate."""
    backend = make_backend(backend_name, capacity=8, executor=research_executor, seed=42)

    # Spawn root
    root_id = backend.spawn(task={"question": "test", "depth": 0, "max_depth": 1})
    assert root_id is not None

    # Execute
    state = backend.execute(root_id)
    assert state is not None

    # Spawn child
    child_id = backend.spawn(parent_id=root_id, task={"question": "sub", "depth": 1, "max_depth": 1})
    backend.execute(child_id)

    # Query
    root_state = backend.query(root_id)
    # Root might be evicted in small caches, but shouldn't crash
    # For unbounded, it should always be available
    if backend_name == "unbounded":
        assert root_state is not None

    # Terminate
    backend.terminate(child_id)


@pytest.mark.parametrize("backend_name", ["bic", "unbounded", "lru", "fifo", "random"])
def test_backend_stays_bounded(backend_name: str) -> None:
    """Bounded backends should not exceed capacity."""
    capacity = 8
    backend = make_backend(backend_name, capacity=capacity, seed=42)

    for i in range(50):
        backend.spawn(task={"i": i})

    if backend_name != "unbounded":
        assert backend.memory_size <= capacity


# ------------------------------------------------------------------ #
# Task runner
# ------------------------------------------------------------------ #

def test_run_research_task_synthetic() -> None:
    """Run a small synthetic task on BIC."""
    config = TaskConfig(
        branching_factor=2,
        max_depth=2,
        response_tokens=64,
        seed=42,
    )
    backend = BICBackend(cache_size=16, executor=research_executor)
    result = run_research_task(backend, config)

    # 2^0 + 2^1 + 2^2 = 7 agents
    assert result.total_agents == 7
    assert result.tree_depth == 2
    assert len(result.final_answer) > 0
    assert len(result.agent_answers) == 7


def test_run_research_task_all_backends() -> None:
    """All backends should complete the same small task."""
    config = TaskConfig(
        branching_factor=2,
        max_depth=2,
        response_tokens=64,
        seed=42,
    )
    for name in ["bic", "unbounded", "lru", "fifo", "random"]:
        backend = make_backend(name, capacity=16, executor=research_executor, seed=42)
        result = run_research_task(backend, config)
        assert result.total_agents == 7, f"{name} spawned {result.total_agents}"


# ------------------------------------------------------------------ #
# Harness
# ------------------------------------------------------------------ #

def test_run_instrumented() -> None:
    """Full instrumented run should produce valid metrics."""
    from experiments.harness import run_instrumented

    config = TaskConfig(
        branching_factor=2,
        max_depth=2,
        response_tokens=64,
        seed=42,
    )
    backend = BICBackend(cache_size=16, executor=research_executor)
    metrics = run_instrumented("bic", backend, config)

    d = metrics.to_dict()
    assert d["backend"] == "bic"
    assert d["task"]["total_agents"] == 7
    assert d["memory"]["peak_bytes"] > 0
    assert d["latency"]["spawn"]["count"] > 0
    assert d["latency"]["execute"]["count"] > 0
    assert 0.0 <= d["quality"]["cache_hit_rate"] <= 1.0
    assert 0.0 <= d["quality"]["query_success_rate"] <= 1.0


def test_instrumented_quality_bic_vs_lru() -> None:
    """BIC with summarization should have >= reconstruction quality vs LRU."""
    from experiments.harness import run_instrumented

    config = TaskConfig(
        branching_factor=3,
        max_depth=3,
        response_tokens=128,
        seed=42,
    )

    bic = BICBackend(cache_size=8, executor=research_executor)
    lru = make_backend("lru", capacity=8, executor=research_executor, seed=42)

    bic_metrics = run_instrumented("bic", bic, config)
    lru_metrics = run_instrumented("lru", lru, config)

    # BIC should have better query success rate (summarization preserves info)
    assert bic_metrics.query_success_rate >= lru_metrics.query_success_rate


# ------------------------------------------------------------------ #
# make_backend factory
# ------------------------------------------------------------------ #

def test_make_backend_unknown_raises() -> None:
    with pytest.raises(ValueError):
        make_backend("nonexistent")


def test_unbounded_is_exact_ceiling_and_empty_scores_zero():
    """Unbounded must reconstruct its own stored answers exactly (SR round-2 review, R5),
    and a record with no text must score 0, not a neutral 0.5."""
    from experiments.baselines import make_backend
    from experiments.harness import run_instrumented
    from experiments.research_decomposition import TaskConfig, research_executor

    cfg = TaskConfig(branching_factor=3, max_depth=2, seed=1)
    q = run_instrumented("unbounded", make_backend("unbounded", executor=research_executor), cfg).to_dict()["quality"]
    assert q["semantic_reconstruction_quality"] == 1.0
    assert q["nonempty_success_rate"] == 1.0

    q = run_instrumented("unbounded", make_backend("unbounded"), cfg).to_dict()["quality"]  # noop: no text stored
    assert q["semantic_reconstruction_quality"] == 0.0
    assert q["nonempty_success_rate"] == 0.0
