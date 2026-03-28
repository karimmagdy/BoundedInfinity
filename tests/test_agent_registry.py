"""
Tests for AgentRegistry.
"""

import pytest

from bounded_infinity.agent_registry import AgentRegistry, AgentStatus


def test_register_root() -> None:
    reg = AgentRegistry(cache_capacity=1024)
    rec = reg.register(parent_id=None, task={"goal": "X"})
    assert rec.parent_id is None
    assert rec.tree_path == ()
    assert rec.depth == 0
    assert rec.cantor_address == 0
    assert rec.status == AgentStatus.PENDING


def test_register_children() -> None:
    reg = AgentRegistry(cache_capacity=1024)
    root = reg.register()
    child1 = reg.register(parent_id=root.agent_id, task={"sub": 1})
    child2 = reg.register(parent_id=root.agent_id, task={"sub": 2})
    assert child1.depth == 1
    assert child2.depth == 1
    assert child1.tree_path == (0,)
    assert child2.tree_path == (1,)
    assert child1.cantor_address != child2.cantor_address


def test_register_deep_tree() -> None:
    reg = AgentRegistry(cache_capacity=1024)
    current = reg.register()
    for depth in range(1, 6):
        current = reg.register(parent_id=current.agent_id)
        assert current.depth == depth


def test_preferred_slot_bounded() -> None:
    """Preferred slot should always be in [0, capacity)."""
    reg = AgentRegistry(cache_capacity=16)
    root = reg.register()
    for _ in range(50):
        child = reg.register(parent_id=root.agent_id)
        assert 0 <= child.preferred_slot < 16


def test_children_of() -> None:
    reg = AgentRegistry(cache_capacity=1024)
    root = reg.register()
    c1 = reg.register(parent_id=root.agent_id)
    c2 = reg.register(parent_id=root.agent_id)
    children = reg.children_of(root.agent_id)
    assert len(children) == 2
    assert {c.agent_id for c in children} == {c1.agent_id, c2.agent_id}


def test_subtree() -> None:
    reg = AgentRegistry(cache_capacity=1024)
    root = reg.register()
    c1 = reg.register(parent_id=root.agent_id)
    c2 = reg.register(parent_id=root.agent_id)
    gc1 = reg.register(parent_id=c1.agent_id)
    subtree = reg.subtree(root.agent_id)
    assert len(subtree) == 3
    ids = {r.agent_id for r in subtree}
    assert ids == {c1.agent_id, c2.agent_id, gc1.agent_id}


def test_set_status() -> None:
    reg = AgentRegistry(cache_capacity=1024)
    rec = reg.register()
    reg.set_status(rec.agent_id, AgentStatus.RUNNING)
    assert reg.get(rec.agent_id).status == AgentStatus.RUNNING


def test_set_status_unknown_raises() -> None:
    reg = AgentRegistry(cache_capacity=1024)
    with pytest.raises(ValueError):
        reg.set_status("no-such-agent", AgentStatus.RUNNING)


def test_register_unknown_parent_raises() -> None:
    reg = AgentRegistry(cache_capacity=1024)
    with pytest.raises(ValueError):
        reg.register(parent_id="no-such-parent")


def test_depth_range() -> None:
    reg = AgentRegistry(cache_capacity=1024)
    root = reg.register()
    c1 = reg.register(parent_id=root.agent_id)
    gc1 = reg.register(parent_id=c1.agent_id)
    gc2 = reg.register(parent_id=c1.agent_id)

    depth_1 = reg.depth_range(min_depth=1, max_depth=1)
    assert len(depth_1) == 1
    depth_2 = reg.depth_range(min_depth=2)
    assert len(depth_2) == 2


def test_active_agents() -> None:
    reg = AgentRegistry(cache_capacity=1024)
    r1 = reg.register()
    r2 = reg.register(parent_id=r1.agent_id)
    reg.set_status(r1.agent_id, AgentStatus.COMPLETED)
    active = reg.active_agents()
    assert len(active) == 1
    assert active[0].agent_id == r2.agent_id


def test_total_agents_and_max_depth() -> None:
    reg = AgentRegistry(cache_capacity=1024)
    root = reg.register()
    c = reg.register(parent_id=root.agent_id)
    _ = reg.register(parent_id=c.agent_id)
    assert reg.total_agents == 3
    assert reg.max_depth == 2
