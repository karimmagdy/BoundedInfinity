"""
Generalized Cantor pairing and unpairing functions.

The Cantor pairing function provides a bijection π: N×N → N, mapping two
natural numbers to a single natural number.  We generalize to n-tuples by
recursive application, yielding a bijection N^k → N.

This module is the foundation of the agent addressing scheme: each agent's
position in the spawn tree (depth, sibling indices) is encoded as a single
natural number, which is then mapped to a cache slot.

Mathematical properties (verified by property-based tests):
  - Bijectivity:  cantor_unpair(cantor_pair(a, b)) == (a, b)  ∀ a,b ∈ N
  - Bijectivity:  cantor_unpair_n(cantor_pair_n(xs)) == xs     ∀ xs ∈ N^k, k≥1
"""

from __future__ import annotations


def cantor_pair(k1: int, k2: int) -> int:
    """Cantor pairing function π(k1, k2) = (k1+k2)(k1+k2+1)/2 + k2.

    Bijection N×N → N.

    >>> cantor_pair(0, 0)
    0
    >>> cantor_pair(1, 0)
    1
    >>> cantor_pair(0, 1)
    2
    >>> cantor_pair(1, 1)
    4
    """
    if k1 < 0 or k2 < 0:
        raise ValueError(f"Inputs must be non-negative, got ({k1}, {k2})")
    s = k1 + k2
    return s * (s + 1) // 2 + k2


def cantor_unpair(z: int) -> tuple[int, int]:
    """Inverse of cantor_pair.  Returns (k1, k2) such that π(k1,k2) = z.

    Uses the standard inversion via the triangular root.

    >>> cantor_unpair(0)
    (0, 0)
    >>> cantor_unpair(4)
    (1, 1)
    """
    if z < 0:
        raise ValueError(f"Input must be non-negative, got {z}")
    # w = floor((sqrt(8z+1) - 1) / 2)  — triangular root
    # We use integer arithmetic to avoid floating-point error.
    w = _isqrt(8 * z + 1)
    # w is floor(sqrt(8z+1)); we need floor((w-1)/2)
    w = (w - 1) // 2
    t = w * (w + 1) // 2
    k2 = z - t
    k1 = w - k2
    return (k1, k2)


def cantor_pair_n(values: tuple[int, ...] | list[int]) -> int:
    """Generalized Cantor pairing for n-tuples (n ≥ 1).

    Encodes (x_1, x_2, …, x_n) by folding from the right:
      pair_n([a])       = a
      pair_n([a, b])    = π(a, b)
      pair_n([a, b, c]) = π(a, π(b, c))

    This is a bijection N^k → N for any fixed k ≥ 1.

    >>> cantor_pair_n([3])
    3
    >>> cantor_pair_n([1, 2])
    8
    >>> cantor_pair_n([1, 2, 3])
    >>> cantor_unpair_n(cantor_pair_n([1, 2, 3]), 3) == (1, 2, 3)
    True
    """
    if not values:
        raise ValueError("Need at least one value")
    for v in values:
        if v < 0:
            raise ValueError(f"All values must be non-negative, got {v}")
    result = values[-1]
    for v in reversed(values[:-1]):
        result = cantor_pair(v, result)
    return result


def cantor_unpair_n(z: int, n: int) -> tuple[int, ...]:
    """Inverse of cantor_pair_n.  Returns n values from a single natural number.

    >>> cantor_unpair_n(cantor_pair_n([1, 2, 3]), 3)
    (1, 2, 3)
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    if n == 1:
        return (z,)
    result: list[int] = []
    current = z
    for _ in range(n - 1):
        k1, k2 = cantor_unpair(current)
        result.append(k1)
        current = k2
    result.append(current)
    return tuple(result)


def tree_address(path: tuple[int, ...] | list[int]) -> int:
    """Encode a tree path (sequence of sibling indices from root to leaf)
    into a single natural number via generalized Cantor pairing.

    The path encodes the agent's position in the spawn tree.
    E.g. path=(2, 0, 3) means: root's 3rd child → that node's 1st child →
    that node's 4th child.

    An empty path (root agent) maps to 0.

    >>> tree_address([])
    0
    >>> tree_address([0])
    1
    >>> tree_address([2, 0, 3])
    >>> tree_address([2, 0, 3]) >= 0
    True
    """
    if not path:
        return 0
    # Prefix each index with +1 to reserve 0 for root, then pair
    shifted = [x + 1 for x in path]
    return cantor_pair_n(shifted)


def tree_address_to_path(addr: int, depth: int) -> tuple[int, ...]:
    """Inverse of tree_address for a known depth.

    >>> tree_address_to_path(0, 0)
    ()
    >>> path = (2, 0, 3)
    >>> tree_address_to_path(tree_address(path), len(path)) == path
    True
    """
    if depth == 0:
        return ()
    shifted = cantor_unpair_n(addr, depth)
    return tuple(x - 1 for x in shifted)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _isqrt(n: int) -> int:
    """Integer square root (floor).  Works for arbitrarily large n."""
    if n < 0:
        raise ValueError("Square root of negative number")
    if n == 0:
        return 0
    x = n
    y = (x + 1) // 2
    while y < x:
        x = y
        y = (x + n // x) // 2
    return x
