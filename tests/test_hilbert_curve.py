"""
Tests for Hilbert curve encoder/decoder and StateCompressor.
"""

from hypothesis import given, settings, assume
from hypothesis import strategies as st
import numpy as np
import pytest

from bounded_infinity.hilbert_curve import HilbertCurve, StateCompressor


# ------------------------------------------------------------------ #
# HilbertCurve roundtrip
# ------------------------------------------------------------------ #

@pytest.mark.parametrize("p,n", [(1, 2), (2, 2), (3, 2), (2, 3), (3, 3), (4, 2)])
def test_roundtrip_exhaustive_small(p: int, n: int) -> None:
    """For small curves, verify every index roundtrips exactly."""
    hc = HilbertCurve(p=p, n=n)
    for idx in range(hc.max_index + 1):
        point = hc.index_to_point(idx)
        assert len(point) == n
        for c in point:
            assert 0 <= c < hc.side
        recovered_idx = hc.point_to_index(point)
        assert recovered_idx == idx, f"Failed at index {idx}: {point} -> {recovered_idx}"


@given(idx=st.integers(min_value=0, max_value=(1 << 12) - 1))
@settings(max_examples=2_000)
def test_roundtrip_2d_p6(idx: int) -> None:
    """Roundtrip for a 2D curve with p=6 (4096 indices)."""
    hc = HilbertCurve(p=6, n=2)
    assume(idx <= hc.max_index)
    point = hc.index_to_point(idx)
    assert hc.point_to_index(point) == idx


@given(idx=st.integers(min_value=0, max_value=(1 << 9) - 1))
@settings(max_examples=1_000)
def test_roundtrip_3d_p3(idx: int) -> None:
    """Roundtrip for a 3D curve with p=3 (512 indices)."""
    hc = HilbertCurve(p=3, n=3)
    assume(idx <= hc.max_index)
    point = hc.index_to_point(idx)
    assert hc.point_to_index(point) == idx


def test_bijective_small_2d() -> None:
    """All indices map to unique points (2D, p=3)."""
    hc = HilbertCurve(p=3, n=2)
    points = set()
    for idx in range(hc.max_index + 1):
        pt = hc.index_to_point(idx)
        assert pt not in points, f"Duplicate point {pt} at index {idx}"
        points.add(pt)
    assert len(points) == hc.max_index + 1


def test_locality_2d() -> None:
    """Adjacent Hilbert indices should map to spatially adjacent points."""
    hc = HilbertCurve(p=4, n=2)
    for idx in range(hc.max_index):
        p1 = hc.index_to_point(idx)
        p2 = hc.index_to_point(idx + 1)
        dist = sum(abs(a - b) for a, b in zip(p1, p2))
        assert dist == 1, f"Non-adjacent points at indices {idx},{idx+1}: {p1}, {p2}"


def test_invalid_params() -> None:
    with pytest.raises(ValueError):
        HilbertCurve(p=0, n=2)
    with pytest.raises(ValueError):
        HilbertCurve(p=2, n=0)


def test_out_of_range() -> None:
    hc = HilbertCurve(p=2, n=2)
    with pytest.raises(ValueError):
        hc.point_to_index([4, 0])  # 4 >= 2^2
    with pytest.raises(ValueError):
        hc.index_to_point(16)  # 16 > 15


# ------------------------------------------------------------------ #
# StateCompressor
# ------------------------------------------------------------------ #

def test_compressor_roundtrip() -> None:
    """Compress then decompress should be approximately the original."""
    comp = StateCompressor(dimensions=3, resolution=8)
    state = np.array([0.25, 0.5, 0.75])
    idx = comp.compress(state)
    recovered = comp.decompress(idx)
    # With 8-bit resolution, error should be < 1/256 ≈ 0.004
    np.testing.assert_allclose(recovered, state, atol=1.0 / 255 + 1e-10)


def test_compressor_boundary_values() -> None:
    comp = StateCompressor(dimensions=2, resolution=4)
    # Corners of the unit square
    for corner in [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]:
        idx = comp.compress(corner)
        recovered = comp.decompress(idx)
        np.testing.assert_allclose(recovered, corner, atol=1.0 / 15 + 1e-10)


@given(state=st.lists(st.floats(min_value=0.0, max_value=1.0,
                                allow_nan=False, allow_infinity=False),
                      min_size=3, max_size=3))
@settings(max_examples=500)
def test_compressor_roundtrip_property(state: list[float]) -> None:
    """Any state in [0,1]^3 roundtrips within quantization error."""
    comp = StateCompressor(dimensions=3, resolution=8)
    idx = comp.compress(state)
    recovered = comp.decompress(idx)
    np.testing.assert_allclose(recovered, state, atol=1.0 / 255 + 1e-10)


def test_compressor_locality() -> None:
    """Similar states should have nearby Hilbert indices on average."""
    comp = StateCompressor(dimensions=2, resolution=10)
    # Compute average index distance for nearby vs. distant point pairs
    import random
    rng = random.Random(42)

    near_dists = []
    far_dists = []
    for _ in range(100):
        # Nearby: two points within 0.01 of each other
        x = rng.uniform(0.05, 0.95)
        y = rng.uniform(0.05, 0.95)
        idx_a = comp.compress([x, y])
        idx_b = comp.compress([x + rng.uniform(-0.01, 0.01),
                               y + rng.uniform(-0.01, 0.01)])
        near_dists.append(abs(idx_a - idx_b))

        # Distant: two random points
        idx_c = comp.compress([rng.random(), rng.random()])
        idx_d = comp.compress([rng.random(), rng.random()])
        far_dists.append(abs(idx_c - idx_d))

    avg_near = sum(near_dists) / len(near_dists)
    avg_far = sum(far_dists) / len(far_dists)
    # On average, nearby points should have much smaller Hilbert distance
    assert avg_near < avg_far


# ------------------------------------------------------------------ #
# Vectorized operations
# ------------------------------------------------------------------ #

def test_batch_distances() -> None:
    hc = HilbertCurve(p=3, n=2)
    points = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    distances = hc.distances_from_points(points)
    assert distances.shape == (4,)
    # Verify each matches scalar version
    for i, row in enumerate(points):
        assert distances[i] == hc.point_to_index(tuple(int(c) for c in row))


def test_batch_points() -> None:
    hc = HilbertCurve(p=3, n=2)
    indices = np.array([0, 1, 2, 3])
    points = hc.points_from_distances(indices)
    assert points.shape == (4, 2)
    for i, idx in enumerate(indices):
        assert tuple(points[i]) == hc.index_to_point(int(idx))
