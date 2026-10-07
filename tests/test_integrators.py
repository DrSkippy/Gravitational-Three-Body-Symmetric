"""Integrator verification on the simple harmonic oscillator."""

from __future__ import annotations

import math

import numpy as np
import pytest

from threebody.integrators import Method
from threebody.simulate import SimulationConfig, simulate
from threebody.systems import HarmonicOscillator
from threebody.verification import (
    convergence_study,
    energy_error_growth,
    leapfrog_dispersion_error,
)

SYMPLECTIC = [Method.LEAPFROG, Method.YOSHIDA4, Method.YOSHIDA6]


@pytest.mark.parametrize("method", SYMPLECTIC)
def test_convergence_order(method: Method) -> None:
    steps = (16, 32, 64) if method is not Method.YOSHIDA6 else (16, 32)
    rows = convergence_study(method, steps)
    observed = rows[-1].observed_order
    assert observed is not None
    assert observed == pytest.approx(method.order, abs=0.1)


def test_leapfrog_matches_exact_discrete_solution() -> None:
    assert leapfrog_dispersion_error() < 1e-11


@pytest.mark.parametrize("method", SYMPLECTIC)
def test_symplectic_energy_error_is_bounded(method: Method) -> None:
    short, long = energy_error_growth(method)
    assert long < 1.05 * short


def test_dop853_is_accurate() -> None:
    sho = HarmonicOscillator(x0=1.0, v0=-0.3)
    res = simulate(sho, SimulationConfig(periods=20, method=Method.DOP853))
    x, v = sho.exact(res.t)
    assert np.max(np.abs(res.q[:, 0] - x)) < 1e-9
    assert np.max(np.abs(res.v[:, 0] - v)) < 1e-9


# Event times differ from the exact pi/2 + k pi only by each method's own
# global phase error after 5 periods at 400 steps per period.
EVENT_TOLERANCE = {
    Method.LEAPFROG: 1e-3,
    Method.YOSHIDA4: 1e-6,
    Method.YOSHIDA6: 1e-10,
    Method.DOP853: 1e-9,
}


@pytest.mark.parametrize("method", list(EVENT_TOLERANCE))
def test_event_location(method: Method) -> None:
    sho = HarmonicOscillator(x0=1.0, v0=0.0)
    res = simulate(
        sho, SimulationConfig(periods=5, steps_per_period=400, method=method)
    )
    ev = res.events.select(0)
    assert len(ev) == 10
    assert list(ev.direction[:2]) == [-1.0, 1.0]
    # The interpolated state lies on the section ...
    assert np.max(np.abs(ev.q[:, 0])) < 1e-12
    # ... and its time matches the exact crossing up to the method's accuracy.
    expected = math.pi / 2 + math.pi * np.arange(len(ev))
    assert np.max(np.abs(ev.t - expected)) < EVENT_TOLERANCE[method]


def test_lyapunov_vanishes_for_linear_oscillator() -> None:
    res = simulate(HarmonicOscillator(), SimulationConfig(periods=100))
    assert abs(res.lyapunov[-1]) < 1e-2


def test_sampling_respects_max_samples() -> None:
    res = simulate(
        HarmonicOscillator(),
        SimulationConfig(periods=10, steps_per_period=1000, max_samples=101),
    )
    assert res.t.shape[0] <= 101
    assert res.t[-1] == pytest.approx(10 * 2 * math.pi)
    assert res.n_steps == 10_000
