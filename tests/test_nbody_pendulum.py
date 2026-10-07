"""General N-body presets and the elastic pendulum."""

from __future__ import annotations

import math

import numpy as np
import pytest

from threebody.demos import (
    PERIODIC_ORBITS,
    broken_symmetry,
    periodic_three_body,
    pythagorean,
)
from threebody.integrators import Method
from threebody.simulate import SimulationConfig, simulate
from threebody.systems import NBody, SpringPendulum


@pytest.mark.parametrize("name", sorted(PERIODIC_ORBITS))
def test_periodic_orbits_close(name: str) -> None:
    model = periodic_three_body(name)
    res = simulate(
        model,
        SimulationConfig(
            periods=1, method=Method.DOP853, lyapunov=False, rtol=1e-13, atol=1e-13
        ),
    )
    start = np.hstack([res.q[0], res.v[0]])
    end = np.hstack([res.q[-1], res.v[-1]])
    assert np.linalg.norm(end - start) < 1e-7


def test_figure8_with_symplectic_integrator() -> None:
    model = periodic_three_body("figure8")
    res = simulate(
        model,
        SimulationConfig(periods=20, steps_per_period=2000, method=Method.YOSHIDA6),
    )
    inv = model.invariants(res.q, res.v)
    assert np.max(np.abs(inv["energy"] / inv["energy"][0] - 1)) < 1e-11
    assert np.max(inv["|linear momentum|"]) < 1e-12
    assert res.t[-1] == pytest.approx(20 * model.characteristic_time())
    assert np.linalg.norm(res.q[-1] - res.q[0]) < 1e-6


def test_momentum_and_angular_momentum_conserved() -> None:
    model = NBody(
        masses=(1.0, 0.5, 0.2, 0.01),
        positions=((0, 0, 0), (1, 0, 0.1), (0, 2, -0.2), (3, 0, 0)),
        velocities=((0, -0.1, 0), (0, 1, 0), (-0.7, 0, 0.05), (0, 0.55, 0)),
    )
    res = simulate(model, SimulationConfig(periods=5, steps_per_period=4000))
    inv = model.invariants(res.q, res.v)
    for key in ("angular momentum Lz", "|angular momentum|"):
        assert np.max(np.abs(inv[key] - inv[key][0])) < 1e-12
    assert (
        np.max(np.abs(inv["|linear momentum|"] - inv["|linear momentum|"][0])) < 1e-12
    )


def test_pythagorean_problem_ejects_lightest_body() -> None:
    model = pythagorean()
    res = simulate(
        model, SimulationConfig(periods=70, method=Method.DOP853, lyapunov=False)
    )
    energy = model.invariants(res.q, res.v)["energy"]
    assert np.max(np.abs(energy / energy[0] - 1)) < 1e-7
    d = model.distances(res.q[-1:])
    # The m=4, m=5 pair is bound tightly; the m=3 body has escaped.
    assert d["|m = 4 - m = 5|"][0] < 1.0
    assert d["|m = 3 - m = 4|"][0] > 10.0


def test_broken_symmetry_leaves_the_axis() -> None:
    model = broken_symmetry()
    res = simulate(
        model, SimulationConfig(periods=60, method=Method.DOP853, lyapunov=False)
    )
    comet = model.body_positions(res.q)[:, 2]
    off_axis = np.hypot(comet[:, 0], comet[:, 1])
    assert off_axis[0] == pytest.approx(1e-3)
    assert off_axis.max() > 1.0


def test_spring_pendulum_vertical_bounce_is_harmonic() -> None:
    model = SpringPendulum(theta0=0.0, omega0=0.0, stretch0=0.05, ratio=3.0)
    res = simulate(model, SimulationConfig(periods=5, steps_per_period=2000))
    ell = np.linalg.norm(res.q, axis=1)
    t_spring = 2 * math.pi / math.sqrt(model.k_over_m)
    expected = model.equilibrium_length + 0.05 * np.cos(2 * math.pi * res.t / t_spring)
    assert np.max(np.abs(ell - expected)) < 1e-8


def test_spring_pendulum_conserves_energy_and_lz() -> None:
    model = SpringPendulum(theta0=0.4, omega0=0.5, stretch0=0.1, v_perp0=0.3)
    res = simulate(model, SimulationConfig(periods=40, steps_per_period=2000))
    inv = model.invariants(res.q, res.v)
    energy = inv["energy (per unit mass)"]
    lz = inv["vertical angular momentum Lz"]
    assert np.max(np.abs(energy / energy[0] - 1)) < 1e-9
    assert np.max(np.abs(lz - lz[0])) < 1e-13


def test_spring_pendulum_small_swing_period() -> None:
    # Very stiff spring: the bob behaves as a rigid pendulum.
    model = SpringPendulum(ratio=60.0, theta0=0.01, omega0=0.0)
    res = simulate(model, SimulationConfig(periods=6, steps_per_period=20_000))
    periods = np.diff(res.events.select(0, 1.0).t)
    assert np.mean(periods) == pytest.approx(model.characteristic_time(), rel=1e-4)
