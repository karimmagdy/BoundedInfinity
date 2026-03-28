"""
BoundedInfinity — Bounding Unbounded Agentic Swarms.

A mathematically rigorous memory management architecture that uses Cantor
pairing functions and Hilbert space-filling curves to compress the state of
an unbounded multi-agent swarm into a finite, fixed-size cache with formal
guarantees of bounded memory, zero fragmentation, and indefinite execution.
"""

from bounded_infinity.cantor_pairing import cantor_pair, cantor_unpair, cantor_pair_n, cantor_unpair_n
from bounded_infinity.hilbert_curve import HilbertCurve, StateCompressor
from bounded_infinity.bounded_cache import BoundedCache
from bounded_infinity.eviction import EvictionManager
from bounded_infinity.agent_registry import AgentRegistry
from bounded_infinity.runtime import BoundedInfinityRuntime
from bounded_infinity.metrics import structural_info, token_info, preservation_ratio

__all__ = [
    "cantor_pair",
    "cantor_unpair",
    "cantor_pair_n",
    "cantor_unpair_n",
    "HilbertCurve",
    "StateCompressor",
    "BoundedCache",
    "EvictionManager",
    "AgentRegistry",
    "BoundedInfinityRuntime",
    "structural_info",
    "token_info",
    "preservation_ratio",
]

__version__ = "0.1.0"
