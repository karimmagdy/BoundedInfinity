"""Tests for the Cantor-vs-hash ablation backend (bic-hash)."""

from __future__ import annotations

import pytest

from bounded_infinity.agent_registry import AgentRegistry, ancestry_hash
from experiments.baselines import HashAncestryBackend, make_backend


def _executor(agent_id, task, state, spawn):
    return {**state, "result": f"{agent_id}:{task.get('goal', '')}"}


def _run_tree(backend, fanout: int = 3, depth: int = 3) -> list[str]:
    """Spawn and execute a full tree breadth-first; return all agent IDs."""
    root = backend.spawn(None, {"goal": "root"})
    backend.execute(root)
    ids, frontier = [root], [root]
    for d in range(depth):
        nxt = []
        for parent in frontier:
            for i in range(fanout):
                child = backend.spawn(parent, {"goal": f"d{d + 1}-{i}"})
                backend.execute(child)
                nxt.append(child)
        ids.extend(nxt)
        frontier = nxt
    return ids


def test_ancestry_hash_is_deterministic_and_path_sensitive() -> None:
    assert ancestry_hash((1, 2, 3)) == ancestry_hash([1, 2, 3])
    assert ancestry_hash((1, 2, 3)) != ancestry_hash((3, 2, 1))
    assert ancestry_hash((1, 2)) != ancestry_hash((1, 2, 0))
    # Fixed 64-bit width regardless of depth (Cantor grows without bound).
    assert ancestry_hash(tuple(range(50))) < 2 ** 64


def test_registry_hash_mode_uses_hash_for_slot() -> None:
    reg = AgentRegistry(cache_capacity=16, addressing="hash")
    root = reg.register()
    child = reg.register(parent_id=root.agent_id)
    grandchild = reg.register(parent_id=child.agent_id)
    for rec in (root, child, grandchild):
        assert rec.cantor_address is None
        assert rec.address == ancestry_hash(rec.tree_path)
        assert rec.preferred_slot == rec.address % 16


def test_registry_cantor_mode_unchanged() -> None:
    reg = AgentRegistry(cache_capacity=16)
    root = reg.register()
    child = reg.register(parent_id=root.agent_id)
    assert root.cantor_address == root.address == 0
    assert child.cantor_address == child.address == 1


def test_registry_rejects_unknown_addressing() -> None:
    with pytest.raises(ValueError):
        AgentRegistry(cache_capacity=8, addressing="hilbert")


def test_bic_hash_matches_bic_aw_config_except_addressing() -> None:
    aw = make_backend("bic-aw", capacity=16, executor=_executor).runtime
    hb = make_backend("bic-hash", capacity=16, executor=_executor).runtime
    assert isinstance(make_backend("bic-hash", capacity=16), HashAncestryBackend)
    assert aw.registry.addressing == "cantor"
    assert hb.registry.addressing == "hash"
    for rt in (aw, hb):
        em = rt.eviction_manager
        assert em.eviction_ancestor_walk is True
        assert em.pin_depth == 0
        assert em.eviction_policy == "depth"
        assert em.eviction_batch_size == 2  # cache_size // 8
        assert em.idle_threshold == 0.0
        assert rt.reconstruct_full_chain is False


def test_bic_hash_forwards_pinning_and_refuses_cantor() -> None:
    hb = make_backend("bic-hash", capacity=16, pin_depth=1).runtime
    assert hb.eviction_manager.pin_depth == 1
    assert hb.registry.addressing == "hash"
    with pytest.raises(ValueError):
        HashAncestryBackend(cache_size=16, addressing="cantor")


def test_bic_hash_differs_from_bic_aw_only_in_slot_placement() -> None:
    aw = make_backend("bic-aw", capacity=8, executor=_executor)
    hb = make_backend("bic-hash", capacity=8, executor=_executor)
    ids_aw = _run_tree(aw)
    ids_hb = _run_tree(hb)
    assert ids_aw == ids_hb

    # The addressing really is different...
    slots_aw = [aw.runtime.registry.get(a).preferred_slot for a in ids_aw]
    slots_hb = [hb.runtime.registry.get(a).preferred_slot for a in ids_hb]
    assert slots_aw != slots_hb

    # ...but the cache bound and the eviction/reconstruction behaviour match.
    for b in (aw, hb):
        assert b.memory_size <= 8
    hist = lambda b: [e["evicted"] for e in b.eviction_stats.eviction_history]
    assert hist(aw) == hist(hb)
    assert aw.runtime.cache.agent_ids() == hb.runtime.cache.agent_ids()
    for a in ids_aw:
        assert aw.query(a) == hb.query(a)
