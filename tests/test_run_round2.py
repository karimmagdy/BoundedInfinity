"""Tests for the round-2 experiment driver, with a fake LLM (no network)."""

from __future__ import annotations

import json

import pytest

from bounded_infinity.adapters import llm_client
from experiments import run_round2 as r2


@pytest.fixture
def fake_llm(monkeypatch, tmp_path):
    """Route provider 'openai' to a deterministic fake; isolate the results dir."""
    calls = {"n": 0, "fail": False}

    def fake_call(prompt, model, max_tokens, api_key):
        calls["n"] += 1
        if calls["fail"]:
            raise TimeoutError("simulated outage")
        text = f"answer {calls['n']} about {prompt.splitlines()[1][:40]}"
        llm_client._tracker.record(len(prompt.split()), len(text.split()))
        llm_client._tracker.record_response(text, finish_reason="stop", served_model=model)
        return text

    monkeypatch.setitem(llm_client._CALLERS, "openai", fake_call)
    monkeypatch.setattr(llm_client.time, "sleep", lambda s: None)
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("LLM_MODEL", "gpt-5.4")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-not-real")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.openai.azure.com/openai/v1/?x=1")
    monkeypatch.setenv("PYTHONHASHSEED", "0")
    monkeypatch.setenv("LLM_MAX_RETRIES", "1")
    monkeypatch.setattr(r2, "RESULTS_DIR", tmp_path)
    return calls


def _records(tmp_path, exp):
    return json.loads((tmp_path / f"{exp}_raw.json").read_text())


def test_plans_match_the_requested_design() -> None:
    sweep = r2.plan_units("sweep")
    assert len(sweep) == 4 * 3 * 5 + 5
    assert {a for a, _, _ in sweep} == {"bic", "bic-aw", "lru", "lru-summary-aw-pin", "unbounded"}
    assert [u for u in sweep if u[0] == "unbounded"] == [("unbounded", None, s) for s in r2.SEEDS_5]
    hash_units = r2.plan_units("hash")
    assert len(hash_units) == 3 * 5 and {a for a, _, _ in hash_units} == {"bic-hash"}
    # The ablation reuses the sweep's seeds and caches, so bic-aw is its partner.
    assert {(c, s) for _, c, s in hash_units} == {(c, s) for a, c, s in sweep if a == "bic-aw"}
    pilot = r2.plan_units("pilot")
    assert len(pilot) == 3 * 3 * 3 + 3
    assert {s for _, _, s in pilot} == set(r2.SEEDS_3)


def test_bic_arm_is_original_and_bic_aw_is_ancestor_walk() -> None:
    noop = lambda *a: {}
    assert r2.build_backend("bic", 8, noop).runtime.eviction_manager.eviction_ancestor_walk is False
    assert r2.build_backend("bic-aw", 8, noop).runtime.eviction_manager.eviction_ancestor_walk is True
    assert r2.build_backend("bic-hash", 8, noop).runtime.registry.addressing == "hash"


def test_pilot_runs_records_and_resumes(fake_llm, tmp_path) -> None:
    r2.run_experiment("pilot", caches=[2], seeds=[42], include_unbounded=True)
    recs = _records(tmp_path, "pilot")
    assert [(r["backend"], r["cache_size"], r["seed"]) for r in recs] == [
        ("unbounded", None, 42), ("bic-aw", 2, 42), ("lru", 2, 42), ("lru-summary-aw-pin", 2, 42)]
    by_arm = {r["backend"]: r for r in recs}
    assert by_arm["unbounded"]["quality"]["semantic_reconstruction_quality"] == 1.0
    assert by_arm["unbounded"]["quality"]["nonempty_success_rate"] == 1.0
    for r in recs:
        assert r["valid"] and r["task"]["empty_generations"] == 0
        assert r["llm"]["model"] == "gpt-5.4"
        assert r["llm"]["endpoint_host"] == "example.openai.azure.com"
        assert r["llm_usage"]["served_models"] == {"gpt-5.4": r["llm_usage"]["total_calls"]}
        assert "test-key-not-real" not in json.dumps(r)
    meta = json.loads(next(tmp_path.glob("pilot_meta.*.json")).read_text())
    assert meta["pythonhashseed"] == "0" and len(meta["code_sha256"]) == 64

    n_calls = fake_llm["n"]
    r2.run_experiment("pilot", caches=[2], seeds=[42], include_unbounded=True)
    assert fake_llm["n"] == n_calls  # everything already done: no new LLM calls


def test_runs_with_llm_errors_are_rejected_and_retried_later(fake_llm, tmp_path) -> None:
    fake_llm["fail"] = True
    r2.run_experiment("pilot", caches=[2], seeds=[42], include_unbounded=False)
    assert _records(tmp_path, "pilot") == []
    rejected = next(tmp_path.glob("pilot_rejected.*"))
    assert all(not json.loads(l)["valid"] for l in rejected.read_text().splitlines())

    fake_llm["fail"] = False
    r2.run_experiment("pilot", caches=[2], seeds=[42], include_unbounded=False)
    assert len(_records(tmp_path, "pilot")) == 3


def test_shards_merge_into_one_raw_file(fake_llm, tmp_path) -> None:
    r2.run_experiment("pilot", caches=[2], seeds=[42], include_unbounded=True)
    r2.run_experiment("pilot", caches=[4], seeds=[42], include_unbounded=False)
    assert len(list(tmp_path.glob("pilot.*.jsonl"))) == 2
    recs = _records(tmp_path, "pilot")
    assert len(recs) == 7 and len({(r["backend"], r["cache_size"]) for r in recs}) == 7


def test_preflight_refuses_missing_model_and_hashseed(monkeypatch) -> None:
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.delenv("PYTHONHASHSEED", raising=False)
    with pytest.raises(SystemExit) as e:
        r2.preflight()
    assert "LLM_MODEL" in str(e.value) and "PYTHONHASHSEED" in str(e.value)


def test_empty_generation_is_not_replaced_by_a_second_llm_call(fake_llm, monkeypatch) -> None:
    from experiments.harness import run_instrumented
    from experiments.research_decomposition import TaskConfig

    second_calls = []
    llm = lambda p: second_calls.append(p) or "a different answer"
    executor = lambda aid, task, state, spawn: {**state, "response": ""}
    cfg = TaskConfig(branching_factor=2, max_depth=1, mode="llm", llm_callable=llm)
    m = run_instrumented("unbounded", r2.build_backend("unbounded", None, executor), cfg)
    assert second_calls == []
    assert m.task_result["empty_generations"] == 3
    assert m.nonempty_success_rate == 0.0


def test_openai_caller_sends_reasoning_effort_and_tracks_length_cutoffs(monkeypatch) -> None:
    import types

    import openai

    sent = {}

    class FakeCompletions:
        def create(self, **kw):
            sent.update(kw)
            usage = types.SimpleNamespace(
                prompt_tokens=10, completion_tokens=150,
                completion_tokens_details=types.SimpleNamespace(reasoning_tokens=150))
            choice = types.SimpleNamespace(
                message=types.SimpleNamespace(content=""), finish_reason="length")
            return types.SimpleNamespace(choices=[choice], usage=usage, model="gpt-5.4-2026-03-05")

    class FakeClient:
        def __init__(self, **kw):
            self.chat = types.SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setattr(openai, "OpenAI", FakeClient)
    monkeypatch.setenv("LLM_REASONING_EFFORT", "none")
    llm_client.reset_usage_tracker()
    assert llm_client._call_openai("hi", "gpt-5.4", 150, "k") == ""
    assert sent["reasoning_effort"] == "none" and sent["max_completion_tokens"] == 150
    s = llm_client.get_usage_tracker().summary()
    assert s["empty_responses"] == 1 and s["finish_reasons"] == {"length": 1}
    assert s["reasoning_tokens"] == 150 and s["served_models"] == {"gpt-5.4-2026-03-05": 1}
