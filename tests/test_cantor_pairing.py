"""
Tests for Cantor pairing functions.

Includes property-based tests (hypothesis) verifying bijectivity over large
random inputs, plus targeted edge-case tests.
"""

from hypothesis import given, settings
from hypothesis import strategies as st
import pytest

from bounded_infinity.cantor_pairing import (
    cantor_pair,
    cantor_unpair,
    cantor_pair_n,
    cantor_unpair_n,
    tree_address,
    tree_address_to_path,
    _isqrt,
)


# ------------------------------------------------------------------ #
# cantor_pair / cantor_unpair bijectivity
# ------------------------------------------------------------------ #

@given(k1=st.integers(min_value=0, max_value=10_000),
       k2=st.integers(min_value=0, max_value=10_000))
@settings(max_examples=5_000)
def test_pair_unpair_roundtrip(k1: int, k2: int) -> None:
    """π⁻¹(π(k1, k2)) == (k1, k2) for all non-negative k1, k2."""
    z = cantor_pair(k1, k2)
    assert z >= 0
    assert cantor_unpair(z) == (k1, k2)


@given(z=st.integers(min_value=0, max_value=100_000))
@settings(max_examples=5_000)
def test_unpair_pair_roundtrip(z: int) -> None:
    """π(π⁻¹(z)) == z for all non-negative z."""
    k1, k2 = cantor_unpair(z)
    assert k1 >= 0
    assert k2 >= 0
    assert cantor_pair(k1, k2) == z


def test_pair_known_values() -> None:
    """Spot-check the standard Cantor pairing sequence."""
    assert cantor_pair(0, 0) == 0
    assert cantor_pair(1, 0) == 1
    assert cantor_pair(0, 1) == 2
    assert cantor_pair(2, 0) == 3
    assert cantor_pair(1, 1) == 4
    assert cantor_pair(0, 2) == 5


def test_pair_negative_raises() -> None:
    with pytest.raises(ValueError):
        cantor_pair(-1, 0)
    with pytest.raises(ValueError):
        cantor_pair(0, -1)


def test_unpair_negative_raises() -> None:
    with pytest.raises(ValueError):
        cantor_unpair(-1)


# ------------------------------------------------------------------ #
# cantor_pair_n / cantor_unpair_n (n-tuple generalization)
# ------------------------------------------------------------------ #

@given(values=st.lists(st.integers(min_value=0, max_value=500),
                       min_size=1, max_size=6))
@settings(max_examples=5_000)
def test_pair_n_roundtrip(values: list[int]) -> None:
    """pair_n and unpair_n are inverses for any tuple length."""
    z = cantor_pair_n(values)
    assert z >= 0
    recovered = cantor_unpair_n(z, len(values))
    assert recovered == tuple(values)


def test_pair_n_single() -> None:
    assert cantor_pair_n([42]) == 42
    assert cantor_unpair_n(42, 1) == (42,)


def test_pair_n_empty_raises() -> None:
    with pytest.raises(ValueError):
        cantor_pair_n([])


def test_pair_n_negative_raises() -> None:
    with pytest.raises(ValueError):
        cantor_pair_n([-1, 2])


# ------------------------------------------------------------------ #
# tree_address / tree_address_to_path
# ------------------------------------------------------------------ #

def test_tree_address_root() -> None:
    assert tree_address([]) == 0


def test_tree_address_first_child() -> None:
    addr = tree_address([0])
    assert addr >= 1  # Non-zero since we shift by +1


@given(path=st.lists(st.integers(min_value=0, max_value=100),
                     min_size=0, max_size=5))
@settings(max_examples=3_000)
def test_tree_address_roundtrip(path: list[int]) -> None:
    """tree_address_to_path(tree_address(path), len(path)) == path."""
    addr = tree_address(path)
    assert addr >= 0
    recovered = tree_address_to_path(addr, len(path))
    assert recovered == tuple(path)


def test_tree_address_unique() -> None:
    """Different paths should (almost always) give different addresses."""
    paths = [
        [], [0], [1], [0, 0], [0, 1], [1, 0], [1, 1],
        [0, 0, 0], [2, 3, 1],
    ]
    addresses = [tree_address(p) for p in paths]
    assert len(set(addresses)) == len(addresses), "Cantor addresses should be unique"


# ------------------------------------------------------------------ #
# _isqrt correctness
# ------------------------------------------------------------------ #

@given(n=st.integers(min_value=0, max_value=10**12))
@settings(max_examples=2_000)
def test_isqrt(n: int) -> None:
    s = _isqrt(n)
    assert s * s <= n
    assert (s + 1) * (s + 1) > n


def test_isqrt_zero() -> None:
    assert _isqrt(0) == 0


def test_isqrt_negative_raises() -> None:
    with pytest.raises(ValueError):
        _isqrt(-1)
