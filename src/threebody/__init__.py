"""Symplectic simulations of Ekeland's symmetric three-body problem.

Quick start::

    from threebody import SymmetricThreeBody, SimulationConfig, simulate

    result = simulate(SymmetricThreeBody(v0=0.17), SimulationConfig(periods=40))
"""

from threebody.integrators import Method
from threebody.metrics import summarize
from threebody.simulate import SimulationConfig, SimulationResult, simulate
from threebody.systems import (
    DynamicalSystem,
    HarmonicOscillator,
    NBody,
    SpringPendulum,
    SymmetricThreeBody,
)

__all__ = [
    "DynamicalSystem",
    "HarmonicOscillator",
    "Method",
    "NBody",
    "SimulationConfig",
    "SimulationResult",
    "SpringPendulum",
    "SymmetricThreeBody",
    "simulate",
    "summarize",
]
__version__ = "2.0.0"
