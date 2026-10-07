"""Physics checks for Ekeland's symmetric three-body problem."""

from __future__ import annotations

import math

import numpy as np
import pytest

from threebody.integrators import Method
from threebody.metrics import summarize
from threebody.simulate import SimulationConfig, simulate
from threebody.systems import SymmetricThreeBody, comet_period


def test_initial_state_is_centre_of_mass_frame() -> None:
    model = SymmetricThreeBody(mu=0.3, v0=0.1, z0=0.2)
    q, v = model.initial_state()
    assert model.mu * q[3] + 2 * q[2] == pytest.approx(0.0, abs=1e-15)
    assert model.mu * v[3] + 2 * v[2] == pytest.approx(0.0, abs=1e-15)
    assert q[3] - q[2] == pytest.approx(0.2)
    assert v[3] - v[2] == pytest.approx(0.1)


def test_circular_binary_stays_circular() -> None:
    model = SymmetricThreeBody(mu=0.0, v0=0.1)
    res = simulate(model, SimulationConfig(periods=5, lyapunov=False))
    rho = np.hypot(res.q[:, 0], res.q[:, 1])
    assert np.max(np.abs(rho - model.r0)) < 1e-9


def test_binary_force_uses_full_separation() -> None:
    # Regression for the original script's G M / (2 r^2) radial force: the
    # binary period must be Kepler's for separation 2 r0 and total mass 2 M.
    model = SymmetricThreeBody(mu=0.0, eccentricity=0.3)
    res = simulate(model, SimulationConfig(periods=10, lyapunov=False))
    phase0 = res.events.select(1, 1.0).t
    assert np.mean(np.diff(phase0)) == pytest.approx(model.binary_period(), rel=1e-9)


@pytest.mark.parametrize("fraction", [0.2, 0.6, 0.9, 0.97])
def test_comet_period_matches_quadrature(fraction: float) -> None:
    model = SymmetricThreeBody(
        mu=0.0, v0=fraction * SymmetricThreeBody().escape_velocity
    )
    expected = model.comet_period()
    res = simulate(
        model,
        SimulationConfig(
            periods=3.5 * expected / model.binary_period(),
            steps_per_period=4000,
            method=Method.YOSHIDA6,
            lyapunov=False,
        ),
    )
    measured = np.diff(res.events.select(0, 1.0).t)
    assert measured.size >= 2
    assert np.max(np.abs(measured / expected - 1.0)) < 1e-9


def test_small_amplitude_period_limit() -> None:
    model = SymmetricThreeBody(v0=1e-6)
    assert comet_period(model.G, model.M, model.r0, 1e-6, 0.0) == pytest.approx(
        model.small_oscillation_period(), rel=1e-9
    )


def test_escape_velocity_gives_infinite_period() -> None:
    model = SymmetricThreeBody()
    assert math.isinf(
        comet_period(model.G, model.M, model.r0, model.escape_velocity, 0)
    )


@pytest.mark.parametrize("ratio", [0.4, 1.0, 3.0])
def test_resonant_solution(ratio: float) -> None:
    model = SymmetricThreeBody.resonant(ratio)
    assert model.comet_period() == pytest.approx(
        ratio * model.binary_period(), rel=1e-12
    )


def test_resonant_rejects_ratio_below_small_oscillation_limit() -> None:
    with pytest.raises(ValueError, match="small-oscillation"):
        SymmetricThreeBody.resonant(0.3)


def test_conservation_with_heavy_comet() -> None:
    model = SymmetricThreeBody(mu=0.5, v0=0.12, eccentricity=0.2)
    res = simulate(model, SimulationConfig(periods=50, lyapunov=False))
    inv = model.invariants(res.q, res.v)
    energy = inv["energy (per unit M)"]
    lz = inv["angular momentum Lz"]
    assert np.max(np.abs(energy / energy[0] - 1)) < 1e-8
    # Central forces + leapfrog-type kicks/drifts conserve Lz to round-off.
    assert np.max(np.abs(lz / lz[0] - 1)) < 1e-12
    axial = model.mu * res.v[:, 3] + 2 * res.v[:, 2]
    assert np.max(np.abs(axial)) < 1e-14


def test_original_csv_columns() -> None:
    model = SymmetricThreeBody()
    q, v = model.initial_state()
    cols = model.columns(np.zeros(1), q[None], v[None])
    assert list(cols) == ["t", "r", "vr", "z", "vz", "y", "vy", "theta", "w"]


def test_chaos_classification() -> None:
    v_esc = SymmetricThreeBody().escape_velocity
    cfg = SimulationConfig(periods=300, steps_per_period=1000, max_samples=2000)
    regular = summarize(simulate(SymmetricThreeBody(v0=0.6 * v_esc), cfg))
    chaotic = summarize(
        simulate(SymmetricThreeBody(v0=0.9 * v_esc, eccentricity=0.1), cfg)
    )
    assert regular["chaos"]["verdict"].startswith("regular")
    assert chaotic["chaos"]["verdict"].startswith("chaotic")


def test_invalid_parameters() -> None:
    with pytest.raises(ValueError):
        SymmetricThreeBody(eccentricity=1.0)
    with pytest.raises(ValueError):
        SymmetricThreeBody(mu=-1.0)
