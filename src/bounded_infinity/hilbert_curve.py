"""
Hilbert space-filling curve encoder/decoder for state compression.

Maps points in an n-dimensional hypercube [0, 2^p)^n to a 1D Hilbert index
and back.  The key property exploited by BoundedInfinity is **locality
preservation**: points close in n-D space map to nearby 1D indices, making
cache eviction of "similar" agent states efficient.

Implementation follows the algorithm from:
  "Programming the Hilbert Curve" — John Skilling, AIP Conf. Proc. 707, 2004.

This is a pure-Python implementation operating on arbitrary-precision integers
so it works for any dimensionality and resolution without overflow.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class HilbertCurve:
    """Hilbert curve mapping between n-D coordinates and 1-D indices.

    Parameters
    ----------
    p : int
        Iteration order (resolution).  Coordinates range in [0, 2^p).
    n : int
        Number of dimensions.
    """

    def __init__(self, p: int, n: int) -> None:
        if p < 1:
            raise ValueError(f"Order p must be >= 1, got {p}")
        if n < 1:
            raise ValueError(f"Dimensions n must be >= 1, got {n}")
        self.p = p
        self.n = n
        self.side = 1 << p          # 2^p
        self.max_index = (1 << (p * n)) - 1  # 2^(p*n) - 1

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def point_to_index(self, coords: tuple[int, ...] | list[int]) -> int:
        """Convert n-D integer coordinates to a 1-D Hilbert index.

        Each coordinate must be in [0, 2^p).

        >>> hc = HilbertCurve(p=2, n=2)
        >>> hc.point_to_index([0, 0])
        0
        >>> hc.point_to_index([1, 1])
        2
        """
        if len(coords) != self.n:
            raise ValueError(f"Expected {self.n} coordinates, got {len(coords)}")
        x = list(coords)
        for c in x:
            if c < 0 or c >= self.side:
                raise ValueError(f"Coordinate {c} out of range [0, {self.side})")
        return self._coords_to_index(x)

    def index_to_point(self, index: int) -> tuple[int, ...]:
        """Convert a 1-D Hilbert index to n-D integer coordinates.

        >>> hc = HilbertCurve(p=2, n=2)
        >>> hc.index_to_point(0)
        (0, 0)
        """
        if index < 0 or index > self.max_index:
            raise ValueError(f"Index {index} out of range [0, {self.max_index}]")
        return self._index_to_coords(index)

    def distances_from_points(self, points: NDArray[np.integer]) -> NDArray[np.int64]:
        """Vectorized point_to_index for a batch of points.

        Parameters
        ----------
        points : ndarray of shape (k, n)
            Each row is an n-D coordinate.

        Returns
        -------
        ndarray of shape (k,) — the Hilbert indices.
        """
        return np.array([self.point_to_index(tuple(int(c) for c in row)) for row in points], dtype=np.int64)

    def points_from_distances(self, indices: NDArray[np.integer]) -> NDArray[np.int64]:
        """Vectorized index_to_point."""
        return np.array([self.index_to_point(int(i)) for i in indices], dtype=np.int64)

    # ------------------------------------------------------------------
    # Hilbert curve via "transpose" representation (Skilling 2004).
    #
    # The Wikipedia C code for TransposetoAxes / AxestoTranspose is the
    # reference.  We store the Hilbert index as a set of n p-bit words
    # (the "transpose").  Conversion to/from a single integer is just
    # bit-interleaving / de-interleaving.
    # ------------------------------------------------------------------

    @staticmethod
    def _transpose_to_axes(x: list[int], n: int, p: int) -> None:
        """In-place TransposetoAxes (index-transpose → coordinates)."""
        # Gray decode
        N = 1 << p  # 2^p
        t = x[n - 1] >> 1
        for i in range(n - 1, 0, -1):
            x[i] ^= x[i - 1]
        x[0] ^= t

        # Undo excess work
        M = 2
        while M != N:
            P = M - 1
            for i in range(n - 1, -1, -1):
                if x[i] & M:
                    x[0] ^= P  # invert
                else:
                    t = (x[0] ^ x[i]) & P
                    x[0] ^= t
                    x[i] ^= t  # exchange
            M <<= 1

    @staticmethod
    def _axes_to_transpose(x: list[int], n: int, p: int) -> None:
        """In-place AxestoTranspose (coordinates → index-transpose)."""
        M = 1 << (p - 1)

        # Inverse undo
        Q = M
        while Q > 1:
            P = Q - 1
            for i in range(n):
                if x[i] & Q:
                    x[0] ^= P  # invert
                else:
                    t = (x[0] ^ x[i]) & P
                    x[0] ^= t
                    x[i] ^= t  # exchange
            Q >>= 1

        # Gray encode
        for i in range(1, n):
            x[i] ^= x[i - 1]
        t = 0
        Q = M
        while Q > 1:
            if x[n - 1] & Q:
                t ^= Q - 1
            Q >>= 1
        for i in range(n):
            x[i] ^= t

    def _coords_to_index(self, x_in: list[int]) -> int:
        """Encode coordinates to Hilbert index."""
        n = self.n
        p = self.p
        x = list(x_in)
        self._axes_to_transpose(x, n, p)
        # Interleave the n p-bit words into one integer
        index = 0
        for bit in range(p - 1, -1, -1):
            for dim in range(n):
                index = (index << 1) | ((x[dim] >> bit) & 1)
        return index

    def _index_to_coords(self, index: int) -> tuple[int, ...]:
        """Decode Hilbert index to coordinates."""
        n = self.n
        p = self.p
        # De-interleave the integer into n p-bit words
        x = [0] * n
        total_bits = p * n
        for i in range(total_bits):
            src_bit = total_bits - 1 - i
            dim = i % n
            bit_pos = p - 1 - (i // n)
            if index & (1 << src_bit):
                x[dim] |= (1 << bit_pos)
        self._transpose_to_axes(x, n, p)
        return tuple(x)


class StateCompressor:
    """Compress a continuous agent state vector into a compact Hilbert index.

    Quantizes a floating-point state vector in [0, 1]^n to integer coordinates
    in [0, 2^p)^n, then maps to a 1-D Hilbert index.  This preserves locality:
    agents with similar states get nearby indices.

    Parameters
    ----------
    dimensions : int
        Dimensionality of the state vectors.
    resolution : int
        Bits of precision per dimension. Higher = more precision, larger indices.
    """

    def __init__(self, dimensions: int, resolution: int = 16) -> None:
        self.dimensions = dimensions
        self.resolution = resolution
        self.curve = HilbertCurve(p=resolution, n=dimensions)
        self._side = 1 << resolution

    def compress(self, state: NDArray[np.floating] | list[float] | tuple[float, ...]) -> int:
        """Compress a state vector in [0,1]^n to a Hilbert index."""
        arr = np.asarray(state, dtype=np.float64)
        if arr.shape != (self.dimensions,):
            raise ValueError(f"Expected shape ({self.dimensions},), got {arr.shape}")
        # Clamp to [0, 1] to handle numerical noise
        arr = np.clip(arr, 0.0, 1.0)
        # Quantize to [0, 2^p - 1]
        coords = np.floor(arr * (self._side - 1)).astype(int)
        return self.curve.point_to_index(tuple(int(c) for c in coords))

    def decompress(self, index: int) -> NDArray[np.float64]:
        """Decompress a Hilbert index back to an approximate state vector in [0,1]^n."""
        coords = self.curve.index_to_point(index)
        return np.array(coords, dtype=np.float64) / (self._side - 1)
