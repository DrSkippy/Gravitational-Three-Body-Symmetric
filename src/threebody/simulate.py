"""High-level driver: run a model and collect the result."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

from threebody.integrators import (
    FloatArray,
    Method,
    integrate_adaptive,
    integrate_fixed,
)
from threebody.systems import DynamicalSystem, EventTable


@dataclass(frozen=True)
class SimulationConfig:
    """How to integrate a model.

    Attributes:
        periods: Run length in units of the model's characteristic time.
        steps_per_period: Fixed-step resolution (ignored by ``dop853``).
        method: Integration method.
        max_samples: Upper bound on stored samples (the integrator itself
            always runs at full resolution).
        lyapunov: Estimate the maximal Lyapunov exponent with a shadow orbit.
        lyapunov_renorm_per_period: Shadow-orbit renormalisations per period.
        rtol: Relative tolerance for ``dop853``.
        atol: Absolute tolerance for ``dop853``.
        seed: Seed for the shadow-orbit perturbation direction.
    """

    periods: float = 20.0
    steps_per_period: int = 2000
    method: Method = Method.YOSHIDA4
    max_samples: int = 20_000
    lyapunov: bool = True
    lyapunov_renorm_per_period: int = 4
    rtol: float = 1e-12
    atol: float = 1e-12
    seed: int = 0


@dataclass(frozen=True)
class SimulationResult:
    """A finished run: samples, events, Lyapunov estimate and timing."""

    system: DynamicalSystem
    config: SimulationConfig
    t: FloatArray
    q: FloatArray
    v: FloatArray
    events: EventTable
    lyapunov_t: FloatArray
    lyapunov: FloatArray
    n_steps: int
    n_force_evaluations: int
    wall_time: float
    dt: float | None
    extras: dict[str, float] = field(default_factory=dict)

    @property
    def t_end(self) -> float:
        return float(self.t[-1])


def simulate(system: DynamicalSystem, config: SimulationConfig) -> SimulationResult:
    """Integrate ``system`` according to ``config``."""
    period = system.characteristic_time()
    t_end = config.periods * period
    q0, v0 = system.initial_state()
    n_events = len(system.event_names)
    started = time.perf_counter()

    if config.method is Method.DOP853:
        n_samples = config.max_samples
        lyap_interval = (
            period / config.lyapunov_renorm_per_period if config.lyapunov else 0.0
        )
        raw = integrate_adaptive(
            system.kernel,
            n_events,
            system.params,
            q0,
            v0,
            t_end,
            n_samples,
            lyapunov_interval=lyap_interval,
            rtol=config.rtol,
            atol=config.atol,
            seed=config.seed,
        )
        dt = None
    else:
        n_steps = max(1, round(config.periods * config.steps_per_period))
        sample_every = max(1, math.ceil(n_steps / (config.max_samples - 1)))
        # A whole number of sampling strides keeps the sample grid uniform and
        # guarantees the final state (t = t_end) is stored.
        n_steps = math.ceil(n_steps / sample_every) * sample_every
        dt = t_end / n_steps
        lyap_every = (
            max(1, config.steps_per_period // config.lyapunov_renorm_per_period)
            if config.lyapunov
            else 0
        )
        raw = integrate_fixed(
            system.kernel,
            n_events,
            system.params,
            q0,
            v0,
            dt,
            n_steps,
            method=config.method,
            sample_every=sample_every,
            lyapunov_every=lyap_every,
            seed=config.seed,
        )
    wall = time.perf_counter() - started

    return SimulationResult(
        system=system,
        config=config,
        t=raw.t,
        q=raw.q,
        v=raw.v,
        events=EventTable.from_rows(raw.events, q0.shape[0]),
        lyapunov_t=raw.lyapunov_t,
        lyapunov=raw.lyapunov,
        n_steps=raw.n_steps,
        n_force_evaluations=raw.n_force_evaluations,
        wall_time=wall,
        dt=dt,
    )
