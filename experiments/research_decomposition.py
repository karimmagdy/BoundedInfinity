"""
Recursive Research Decomposition — the main experiment task.

A "research agent" receives a broad research question and decomposes it
into sub-questions, spawning child agents for each.  Each child
investigates its sub-question (optionally calling an LLM), and may
further decompose.  The final answer is assembled by aggregating results
up the tree.

Two execution modes:
  - **synthetic**: Uses deterministic pseudo-text (fast, reproducible, for
    scale/memory experiments).
  - **llm**: Calls a real LLM API for investigation (for quality experiments).

The task interface is backend-agnostic: any system that exposes
spawn/execute/query/terminate can run it.
"""

from __future__ import annotations

import hashlib
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol


# ------------------------------------------------------------------ #
# Backend protocol — what every system under test must expose
# ------------------------------------------------------------------ #

class SwarmBackend(Protocol):
    """Minimal interface that BIC + every baseline must implement."""

    def spawn(self, parent_id: str | None = None,
              task: dict[str, Any] | None = None) -> str: ...

    def execute(self, agent_id: str) -> dict[str, Any]: ...

    def query(self, agent_id: str) -> dict[str, Any] | None: ...

    def terminate(self, agent_id: str) -> None: ...


# ------------------------------------------------------------------ #
# Task configuration
# ------------------------------------------------------------------ #

@dataclass
class TaskConfig:
    """Configuration for a research-decomposition run."""

    question: str = "Survey the state of memory management in multi-agent LLM systems"
    branching_factor: int = 3        # children per non-leaf
    max_depth: int = 4               # tree depth (0 = root only)
    response_tokens: int = 512       # approximate tokens per agent response
    mode: str = "synthetic"          # "synthetic" | "llm"
    seed: int = 42
    llm_callable: Any = None         # Callable[[str], str] for LLM mode
    access_order: str = "bfs"        # "bfs" | "dfs" | "random" — agent spawn/exec order
    deterministic_content: bool = False  # fix tree content across orders (robustness probe)


@dataclass
class TaskResult:
    """Aggregated result of a research-decomposition run."""

    root_id: str
    total_agents: int
    tree_depth: int
    final_answer: str
    agent_answers: dict[str, str] = field(default_factory=dict)
    elapsed_seconds: float = 0.0
    empty_generations: int = 0  # agents whose stored response was empty


# ------------------------------------------------------------------ #
# Synthetic text generation (deterministic, seeded)
# ------------------------------------------------------------------ #

_VOCABULARY = (
    "the agent explores memory management techniques including caching "
    "eviction summarization hierarchical compression bounded state "
    "retrieval preservation information loss multi-agent coordination "
    "recursive decomposition sub-question investigation aggregation "
    "Hilbert curve Cantor pairing function space-filling bijection "
    "transformer attention mechanism embedding LLM token context "
    "window sliding eviction policy depth-first breadth-first "
    "fragmentation compaction defragmentation slot allocation "
    "open addressing linear probing hash collision resolution"
).split()


def _synthetic_response(question: str, n_tokens: int, seed: int) -> str:
    """Generate deterministic pseudo-text for a research sub-question."""
    h = hashlib.sha256(f"{question}:{seed}".encode()).hexdigest()
    rng = random.Random(int(h[:16], 16))
    # ~4 chars/token on average
    words = [rng.choice(_VOCABULARY) for _ in range(n_tokens)]
    return " ".join(words)


def _decompose_question(question: str, n_children: int, seed: int) -> list[str]:
    """Break a research question into sub-questions (deterministic)."""
    prefixes = [
        "What are the key techniques for",
        "How does the literature address",
        "What are the tradeoffs in",
        "What open problems remain in",
        "What empirical evidence exists for",
        "How do existing frameworks handle",
        "What theoretical foundations underlie",
        "Compare and contrast approaches to",
    ]
    rng = random.Random(seed)
    # Extract a topic fragment from the question
    tokens = question.split()
    topic_start = min(3, len(tokens) - 1)
    topic = " ".join(tokens[topic_start:topic_start + 5]) if len(tokens) > topic_start else question

    subs = []
    for i in range(n_children):
        prefix = rng.choice(prefixes)
        variant = f"{prefix} {topic} (sub-question {i + 1})"
        subs.append(variant)
    return subs


# ------------------------------------------------------------------ #
# Core task runner (backend-agnostic)
# ------------------------------------------------------------------ #

def run_research_task(
    backend: SwarmBackend,
    config: TaskConfig,
) -> TaskResult:
    """Execute the recursive research decomposition task on any backend.

    Returns a TaskResult with the final aggregated answer and per-agent
    answers (where available via query).
    """
    rng = random.Random(config.seed)
    start = time.perf_counter()

    # -- Spawn and execute agents level by level (BFS) -----------------
    root_id = backend.spawn(task={
        "question": config.question,
        "depth": 0,
        "max_depth": config.max_depth,
    })

    # Frontier order controls which agents are "recently used" when the cache
    # fills, exercising recency-based policies (LRU) differently. BIC's
    # depth-priority eviction is order-invariant by construction, so this is a
    # fair robustness probe: the same tree, only the traversal order changes.
    order = getattr(config, "access_order", "bfs")
    queue: list[tuple[str, str, int]] = [(root_id, config.question, 0)]
    agent_answers: dict[str, str] = {}
    total_agents = 1
    empty_generations = 0

    while queue:
        if order == "dfs":
            agent_id, question, depth = queue.pop()          # LIFO: depth-first
        elif order == "random":
            agent_id, question, depth = queue.pop(rng.randrange(len(queue)))
        else:
            agent_id, question, depth = queue.pop(0)         # FIFO: breadth-first

        # -- Generate response for this agent -------------------------
        # By default child_seed is drawn sequentially (legacy). For the
        # access-order robustness probe we derive it deterministically from the
        # node's question+depth so that tree CONTENT is identical across
        # traversal orders and only cache dynamics differ (a clean A/B).
        if getattr(config, "deterministic_content", False):
            child_seed = int(hashlib.sha256(
                f"{question}:{depth}:{config.seed}".encode()).hexdigest()[:8], 16)
        else:
            child_seed = rng.randint(0, 2**31)
        # -- Execute, and take the reference answer from what the backend
        # stored. Previously the runner generated its OWN answer here (a
        # second, independent LLM call with a different prompt) and used it as
        # the reference, so even the Unbounded backend scored ~0.25 against
        # it. The reference must be the exact text the backend was given.
        state = backend.execute(agent_id)
        response = (state or {}).get("response", "")
        if not response and config.mode != "llm":
            # Executor stored no text (e.g. noop executor): fall back to a
            # locally generated answer, as before. In LLM mode an empty
            # generation stays empty: a second, differently-prompted LLM call
            # would reintroduce the reference mismatch fixed above. Empty
            # generations are counted in TaskResult.empty_generations.
            response = _synthetic_response(question, config.response_tokens, child_seed)
        if not response:
            empty_generations += 1
        agent_answers[agent_id] = response

        # -- Spawn children if not at max depth -----------------------
        if depth < config.max_depth:
            sub_questions = _decompose_question(question, config.branching_factor, child_seed)
            for i, sub_q in enumerate(sub_questions):
                child_id = backend.spawn(
                    parent_id=agent_id,
                    task={
                        "question": sub_q,
                        "depth": depth + 1,
                        "max_depth": config.max_depth,
                    },
                )
                queue.append((child_id, sub_q, depth + 1))
                total_agents += 1

    # -- Aggregate final answer from root's state ---------------------
    root_state = backend.query(root_id)
    final_answer = _aggregate_answers(root_id, agent_answers, backend)

    elapsed = time.perf_counter() - start

    return TaskResult(
        root_id=root_id,
        total_agents=total_agents,
        tree_depth=config.max_depth,
        final_answer=final_answer,
        agent_answers=agent_answers,
        elapsed_seconds=elapsed,
        empty_generations=empty_generations,
    )


def _aggregate_answers(root_id: str, answers: dict[str, str],
                       backend: SwarmBackend) -> str:
    """Build the final aggregated answer from per-agent answers.

    Uses the root's reconstructed state (will contain summaries of
    evicted children for BIC) plus all available agent answers.
    """
    parts = []
    root_state = backend.query(root_id)
    if root_state is not None and "_summaries" in root_state:
        parts.append(f"[Summaries from {len(root_state['_summaries'])} evicted children]")
    if root_id in answers:
        parts.append(answers[root_id])

    # Collect up to 20 non-root answers for final aggregation
    count = 0
    for aid, answer in answers.items():
        if aid != root_id and count < 20:
            parts.append(f"[{aid[:8]}]: {answer[:200]}")
            count += 1

    return "\n---\n".join(parts) if parts else "(no answers collected)"


# ------------------------------------------------------------------ #
# Executor for BIC runtime integration
# ------------------------------------------------------------------ #

def research_executor(
    agent_id: str,
    task: dict[str, Any],
    state: dict[str, Any],
    spawn: Callable[[dict[str, Any]], str],
) -> dict[str, Any]:
    """Executor for use with BoundedInfinityRuntime.

    Generates a synthetic response and stores it in the agent's state.
    Spawning is handled externally by the task runner, so this executor
    just processes the current agent's question.
    """
    result = dict(state)
    question = task.get("question", "")
    depth = task.get("depth", 0)
    seed = hash(agent_id) & 0x7FFFFFFF
    result["response"] = _synthetic_response(question, 512, seed)
    result["question"] = question
    result["depth"] = depth
    result["completed"] = True
    return result
