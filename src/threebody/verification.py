"""Verification of the integrators against the simple harmonic oscillator.

Three independent checks:

1. **Convergence order.** The global error at a fixed time must fall as
   ``dt^p`` with ``p`` the method's order.
2. **Exact discrete dispersion.** Kick-drift-kick leapfrog applied to
   ``x'' = -w^2 x`` is a linear map whose rotation angle per step obeys
   ``cos(W dt) = 1 - (w dt)^2 / 2``.  The numerical solution must match
   ``x0 cos(W t)`` (up to the map's amplitude modulation) to round-off,
   which checks the implementation, not just its order.
3. **Bounded energy error.** A symplectic method conserves a shadow
   Hamiltonian, so the energy error oscillates but does not grow.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from threebody.integrators import Method
from threebody.simulate import SimulationConfig, simulate
from threebody.systems import HarmonicOscillator


@dataclass(frozen=True)
class ConvergenceRow:
    """Global error at the end of a run for one step size."""

    method: Method
    steps_per_period: int
    dt: float
    max_error: float
    observed_order: float | None


def convergence_study(
    method: Method,
    steps: tuple[int, ...] = (16, 32, 64, 128),
    periods: float = 10.0,
) -> list[ConvergenceRow]:
    """Max position error vs exact solution for a sequence of step sizes."""
    sho = HarmonicOscillator(omega=1.0, x0=1.0, v0=0.5)
    rows: list[ConvergenceRow] = []
    prev: tuple[float, float] | None = None
    for n in steps:
        cfg = SimulationConfig(
            periods=periods, steps_per_period=n, method=method, lyapunov=False
        )
        res = simulate(sho, cfg)
        x_exact, _ = sho.exact(res.t)
        err = float(np.max(np.abs(res.q[:, 0] - x_exact)))
        dt = sho.characteristic_time() / n
        order = (
            math.log(prev[1] / err) / math.log(prev[0] / dt)
            if prev is not None and err > 0
            else None
        )
        rows.append(ConvergenceRow(method, n, dt, err, order))
        prev = (dt, err)
    return rows


def leapfrog_dispersion_error(
    omega: float = 1.0, steps_per_period: int = 20, periods: float = 50.0
) -> float:
    """Max deviation of leapfrog from its exact discrete solution.

    For KDK leapfrog with ``x0 = 1, v0 = 0`` the iterates are exactly
    ``x_n = cos(n W dt)`` with ``cos(W dt) = 1 - (omega dt)^2 / 2``.
    """
    sho = HarmonicOscillator(omega=omega, x0=1.0, v0=0.0)
    cfg = SimulationConfig(
        periods=periods,
        steps_per_period=steps_per_period,
        method=Method.LEAPFROG,
        lyapunov=False,
    )
    res = simulate(sho, cfg)
    dt = sho.characteristic_time() / steps_per_period
    big_w = math.acos(1.0 - 0.5 * (omega * dt) ** 2) / dt
    return float(np.max(np.abs(res.q[:, 0] - np.cos(big_w * res.t))))


def energy_error_growth(
    method: Method, steps_per_period: int = 32
) -> tuple[float, float]:
    """Max relative energy error over 10 and over 1000 periods."""
    sho = HarmonicOscillator(omega=1.0, x0=1.0, v0=0.5)
    out = []
    for periods in (10.0, 1000.0):
        cfg = SimulationConfig(
            periods=periods,
            steps_per_period=steps_per_period,
            method=method,
            lyapunov=False,
            max_samples=200_000,
        )
        res = simulate(sho, cfg)
        e = sho.invariants(res.q, res.v)["energy"]
        out.append(float(np.max(np.abs(e - e[0])) / e[0]))
    return out[0], out[1]


def report() -> str:
    """Human-readable verification report."""
    lines = ["Simple harmonic oscillator verification (omega = 1, 10 periods)", ""]
    lines.append(
        f"{'method':<10} {'steps/T':>8} {'dt':>10} {'max |x - x_exact|':>18} {'order':>7}"
    )
    for method in (Method.LEAPFROG, Method.YOSHIDA4, Method.YOSHIDA6):
        steps = (16, 32, 64, 128) if method is not Method.YOSHIDA6 else (8, 16, 32, 64)
        for row in convergence_study(method, steps):
            order = f"{row.observed_order:7.3f}" if row.observed_order else "      -"
            lines.append(
                f"{method.value:<10} {row.steps_per_period:>8} {row.dt:>10.4g} "
                f"{row.max_error:>18.3e} {order}"
            )
        lines.append(f"{'':<10} expected order {method.order}")
    lines += [
        "",
        "Leapfrog vs its exact discrete solution cos(W n dt), "
        "cos(W dt) = 1 - (w dt)^2/2:",
        f"  max deviation over 50 periods at 20 steps/period: "
        f"{leapfrog_dispersion_error():.2e}",
        "",
        "Energy error growth (symplectic => bounded, adaptive => drifts):",
    ]
    for method in Method:
        short, long = energy_error_growth(method)
        lines.append(
            f"  {method.value:<10} 10 periods: {short:.2e}   1000 periods: {long:.2e}"
        )
    return "\n".join(lines)
