"""
Tests for BoundedCache — the fixed-size state cache.
"""

import pytest

from bounded_infinity.bounded_cache import BoundedCache, CacheFull


def test_basic_put_get() -> None:
    cache = BoundedCache(capacity=4)
    slot = cache.put("a1", {"x": 1})
    entry = cache.get("a1")
    assert entry is not None
    assert entry.state == {"x": 1}
    assert entry.agent_id == "a1"
    assert cache.size == 1


def test_update_existing() -> None:
    cache = BoundedCache(capacity=4)
    cache.put("a1", {"x": 1})
    cache.put("a1", {"x": 2})
    assert cache.size == 1
    assert cache.get("a1").state == {"x": 2}


def test_remove() -> None:
    cache = BoundedCache(capacity=4)
    cache.put("a1", {"x": 1})
    entry = cache.remove("a1")
    assert entry is not None
    assert entry.state == {"x": 1}
    assert cache.size == 0
    assert cache.get("a1") is None


def test_remove_nonexistent() -> None:
    cache = BoundedCache(capacity=4)
    assert cache.remove("nope") is None


def test_cache_full() -> None:
    cache = BoundedCache(capacity=2)
    cache.put("a1", {"x": 1})
    cache.put("a2", {"x": 2})
    with pytest.raises(CacheFull):
        cache.put("a3", {"x": 3})


def test_capacity_after_remove() -> None:
    cache = BoundedCache(capacity=2)
    cache.put("a1", {"x": 1})
    cache.put("a2", {"x": 2})
    cache.remove("a1")
    cache.put("a3", {"x": 3})  # Should succeed
    assert cache.size == 2


def test_preferred_slot() -> None:
    cache = BoundedCache(capacity=10)
    slot = cache.put("a1", {"x": 1}, preferred_slot=7)
    assert slot == 7


def test_preferred_slot_collision() -> None:
    cache = BoundedCache(capacity=10)
    cache.put("a1", {"x": 1}, preferred_slot=5)
    slot2 = cache.put("a2", {"x": 2}, preferred_slot=5)
    # Should be placed nearby via linear probing
    assert slot2 != 5
    assert cache.size == 2


def test_contains() -> None:
    cache = BoundedCache(capacity=4)
    cache.put("a1", {"x": 1})
    assert cache.contains("a1")
    assert not cache.contains("a2")


def test_agent_ids() -> None:
    cache = BoundedCache(capacity=4)
    cache.put("a1", {"x": 1})
    cache.put("a2", {"x": 2})
    assert cache.agent_ids() == {"a1", "a2"}


def test_entries() -> None:
    cache = BoundedCache(capacity=4)
    cache.put("a1", {"x": 1})
    cache.put("a2", {"x": 2})
    entries = cache.entries()
    assert len(entries) == 2
    ids = {e.agent_id for e in entries}
    assert ids == {"a1", "a2"}


# ------------------------------------------------------------------ #
# Compaction & Fragmentation
# ------------------------------------------------------------------ #

def test_compact_no_fragmentation() -> None:
    """A full cache with no gaps should not move anything."""
    cache = BoundedCache(capacity=3)
    cache.put("a1", {"x": 1})
    cache.put("a2", {"x": 2})
    cache.put("a3", {"x": 3})
    moved = cache.compact()
    assert moved == 0


def test_compact_after_removal() -> None:
    cache = BoundedCache(capacity=4)
    cache.put("a1", {"x": 1}, preferred_slot=0)
    cache.put("a2", {"x": 2}, preferred_slot=1)
    cache.put("a3", {"x": 3}, preferred_slot=2)
    cache.put("a4", {"x": 4}, preferred_slot=3)
    cache.remove("a2")  # Creates a gap at slot 1
    cache.remove("a3")  # Creates a gap at slot 2
    moved = cache.compact()
    assert moved > 0
    # After compaction, all entries should be contiguous from slot 0
    assert cache._slots[0] is not None
    assert cache._slots[1] is not None
    assert cache._slots[2] is None
    assert cache._slots[3] is None


def test_fragmentation_ratio() -> None:
    cache = BoundedCache(capacity=10)
    assert cache.fragmentation_ratio == 0.0

    cache.put("a1", {"x": 1}, preferred_slot=0)
    cache.put("a2", {"x": 2}, preferred_slot=5)
    # 2 entries, first 2 slots check: slot 0 occupied, slot 1 empty
    assert cache.fragmentation_ratio > 0.0

    cache.compact()
    assert cache.fragmentation_ratio == 0.0


# ------------------------------------------------------------------ #
# Metrics
# ------------------------------------------------------------------ #

def test_occupancy() -> None:
    cache = BoundedCache(capacity=4)
    assert cache.occupancy == 0.0
    cache.put("a1", {})
    assert cache.occupancy == 0.25
    cache.put("a2", {})
    assert cache.occupancy == 0.5


def test_is_full() -> None:
    cache = BoundedCache(capacity=2)
    assert not cache.is_full
    cache.put("a1", {})
    cache.put("a2", {})
    assert cache.is_full


def test_invalid_capacity() -> None:
    with pytest.raises(ValueError):
        BoundedCache(capacity=0)
