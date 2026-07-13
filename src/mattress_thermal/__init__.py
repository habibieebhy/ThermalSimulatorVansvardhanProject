"""Thermal performance simulator for mattress prototype evaluation."""

from .simulation import (
    SimulationConfig,
    SimulationResult,
    build_prototypes,
    run_mattress_simulation,
)

__all__ = [
    "SimulationConfig",
    "SimulationResult",
    "build_prototypes",
    "run_mattress_simulation",
]

