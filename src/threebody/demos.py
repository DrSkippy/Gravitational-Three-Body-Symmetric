"""Curated demonstration runs: periodic, quasi-periodic and chaotic.

The periodic three-body initial conditions are the Suvakov-Dmitrasinovic
(2013) families in their convention (unit masses, ``G = 1``, bodies at
``(-1, 0)``, ``(1, 0)``, ``(0, 0)`` with velocities ``(p1, p2)``,
``(p1, p2)``, ``(-2 p1, -2 p2)``), re-converged here by Newton shooting so
that each orbit closes to ~1e-11 after one period (the published 6-digit
values close only to ~1e-3).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

from threebody.integrators import Method
from threebody.simulate import SimulationConfig
from threebody.systems import (
    DynamicalSystem,
    HarmonicOscillator,
    NBody,
    SpringPendulum,
    SymmetricThreeBody,
)

# (p1, p2, period), refined by shooting; see module docstring.
PERIODIC_ORBITS: dict[str, tuple[float, float, float]] = {
    "figure8": (0.347116888119226, 0.532724945387937, 6.325913982926752),
    "butterfly1": (0.306893420445669, 0.125506567019571, 6.234674838573193),
    "butterfly2": (0.392955493720964, 0.097578968412215, 7.003709600896517),
    "butterfly3": (0.405915567134907, 0.230163125980364, 13.867123435718113),
    "bumblebee": (0.184278499160169, 0.587188172191679, 63.534353339945049),
    "moth1": (0.464445172817842, 0.396060014652845, 14.894305175017841),
    "moth2": (0.439165917887362, 0.452967643191304, 28.669270914934728),
    "goggles": (0.083300071849282, 0.127889255520081, 10.464849525874627),
    "dragonfly": (0.080584225540029, 0.588836089777925, 21.272337394569096),
    "yinyang1a": (0.513938537459262, 0.304735919346950, 17.328834018504512),
    "yinyang1b": (0.282702090433725, 0.327208971522662, 10.963303088019396),
}


def periodic_three_body(name: str) -> NBody:
    """One of the :data:`PERIODIC_ORBITS` as an :class:`NBody` model."""
    p1, p2, period = PERIODIC_ORBITS[name]
    return NBody(
        masses=(1.0, 1.0, 1.0),
        positions=((-1.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        velocities=((p1, p2, 0.0), (p1, p2, 0.0), (-2.0 * p1, -2.0 * p2, 0.0)),
        period=period,
    )


def pythagorean() -> NBody:
    """Burrau's problem: masses 3, 4, 5 at rest on a 3-4-5 triangle."""
    return NBody(
        masses=(3.0, 4.0, 5.0),
        positions=((1.0, 3.0, 0.0), (-2.0, -1.0, 0.0), (1.0, -1.0, 0.0)),
        velocities=((0.0, 0.0, 0.0),) * 3,
        period=1.0,
        labels=("m = 3", "m = 4", "m = 5"),
    )


def broken_symmetry(
    mu: float = 0.1, offset: float = 1e-3, speed_fraction: float = 0.9
) -> NBody:
    """Ekeland's configuration as a full 3D N-body problem.

    Two unit stars (``G = 1``) on a circular orbit of radius 1/2 and a comet
    of mass ``mu`` launched along the axis at ``speed_fraction`` of the
    escape speed, displaced ``offset`` off the axis.  The axis is unstable
    for these energies: the tiny offset grows until the comet is flung out
    sideways.
    """
    r = 0.5
    # Radial acceleration of a star: G M / (2r)^2 from its partner plus the
    # comet's pull G mu / r^2 (comet in the plane, on the axis).
    omega = math.sqrt((0.25 / r**2 + mu / r**2) / r)
    v0 = speed_fraction * math.sqrt(4.0 / r)
    vz = 2.0 * v0 / (2.0 + mu)
    vy = -mu * v0 / (2.0 + mu)
    return NBody(
        masses=(1.0, 1.0, mu),
        positions=((r, 0.0, 0.0), (-r, 0.0, 0.0), (offset, 0.0, 0.0)),
        velocities=((0.0, r * omega, vy), (0.0, -r * omega, vy), (0.0, 0.0, vz)),
        period=2.0 * math.pi / omega,
        labels=("star A", "star B", "comet"),
    )


@dataclass(frozen=True)
class Demo:
    """A named, reproducible run."""

    title: str
    description: str
    build: Callable[[], DynamicalSystem]
    config: SimulationConfig
    animate_periods: float | None = 10.0


_SYM_V_ESC = SymmetricThreeBody().escape_velocity

DEMOS: dict[str, Demo] = {
    "sho": Demo(
        "Verification: simple harmonic oscillator",
        "x'' = -x integrated with 4th-order Yoshida at 64 steps per period. The "
        "global error, period and energy are compared with the exact solution.",
        HarmonicOscillator,
        SimulationConfig(periods=50, steps_per_period=64),
    ),
    "ekeland-coil": Demo(
        "Ekeland's comet in 3:1 resonance",
        "Circular binary, massless comet whose period is exactly three binary "
        "periods (v0 solved by quadrature). Drawn on the cylinder swept by the "
        "stars, the comet's height closes into a coil after one comet period.",
        lambda: SymmetricThreeBody.resonant(3.0),
        SimulationConfig(periods=300, steps_per_period=1500),
        animate_periods=6.0,
    ),
    "ekeland-weave": Demo(
        "Ekeland's comet in 2:5 resonance",
        "Five comet oscillations for every two binary turns: a closed five-lobed "
        "weave on the stellar cylinder. Strictly periodic, line spectrum.",
        lambda: SymmetricThreeBody.resonant(0.4),
        SimulationConfig(periods=300, steps_per_period=1500),
        animate_periods=4.0,
    ),
    "ekeland-quasiperiodic": Demo(
        "Ekeland's original run (v0 = 0.1817)",
        "The README's example: an irrational period ratio, so the comet's trace "
        "fills a band on the stellar cylinder without closing. Regular, not chaotic.",
        lambda: SymmetricThreeBody(v0=0.1817),
        SimulationConfig(periods=300, steps_per_period=1500),
    ),
    "sitnikov-chaos": Demo(
        "Sitnikov chaos: eccentric binary (e = 0.2)",
        "A 20% eccentric binary turns the integrable comet into the classic chaotic "
        "Sitnikov problem. Launched at 84% of escape speed, the comet wanders "
        "chaotically for hundreds of binary periods: its period jumps erratically "
        "and the stroboscopic section is a scattered cloud. A chaotic orbit near "
        "escape energy is eventually ejected, at a time that depends on round-off "
        "(see `threebody portrait` for the whole phase space).",
        lambda: SymmetricThreeBody(v0=0.84 * _SYM_V_ESC, eccentricity=0.2),
        SimulationConfig(periods=300, steps_per_period=1000),
    ),
    "heavy-comet": Demo(
        "Heavy comet: coupled chaos (mu = 0.1)",
        "A comet a tenth as heavy as each star squeezes the binary every time it "
        "passes, so the binary breathes and the stellar plane recoils. Energy "
        "sloshes between the degrees of freedom chaotically.",
        lambda: SymmetricThreeBody(v0=0.9 * _SYM_V_ESC, mu=0.1),
        SimulationConfig(periods=300, steps_per_period=1500),
    ),
    "broken-symmetry": Demo(
        "Ekeland's configuration, symmetry broken (full 3D N-body)",
        "The same stars and comet integrated as a general 3D three-body problem "
        "with the comet 0.001 off the axis. The offset grows exponentially and the "
        "comet is eventually thrown out sideways.",
        broken_symmetry,
        SimulationConfig(periods=60, method=Method.DOP853, max_samples=12_000),
        animate_periods=None,
    ),
    "figure8": Demo(
        "Figure-eight choreography",
        "Chenciner-Montgomery's figure-eight: three equal masses chase each other "
        "on one curve. Linearly stable, so it survives 200 periods with the energy "
        "conserved to round-off by the 6th-order symplectic integrator.",
        lambda: periodic_three_body("figure8"),
        SimulationConfig(periods=200, steps_per_period=2000, method=Method.YOSHIDA6),
        animate_periods=2.0,
    ),
    "butterfly": Demo(
        "Butterfly I",
        "Suvakov-Dmitrasinovic periodic orbit I.A.1, re-converged by shooting. "
        "Like most of this family it is linearly unstable: run it for many "
        "periods and round-off grows until the choreography breaks up.",
        lambda: periodic_three_body("butterfly1"),
        SimulationConfig(periods=2, method=Method.DOP853, max_samples=12_000),
        animate_periods=None,
    ),
    "moth": Demo(
        "Moth I",
        "Suvakov-Dmitrasinovic periodic orbit I.B.1, re-converged by shooting.",
        lambda: periodic_three_body("moth1"),
        SimulationConfig(periods=2, method=Method.DOP853, max_samples=12_000),
        animate_periods=None,
    ),
    "yin-yang": Demo(
        "Yin-yang Ia",
        "Suvakov-Dmitrasinovic periodic orbit II.C.2a, re-converged by shooting.",
        lambda: periodic_three_body("yinyang1a"),
        SimulationConfig(periods=2, method=Method.DOP853, max_samples=12_000),
        animate_periods=None,
    ),
    "bumblebee": Demo(
        "Bumblebee",
        "Suvakov-Dmitrasinovic orbit I.A.3: a long, intricate period.",
        lambda: periodic_three_body("bumblebee"),
        SimulationConfig(periods=1, method=Method.DOP853, max_samples=20_000),
        animate_periods=None,
    ),
    "pythagorean": Demo(
        "Pythagorean three-body problem (Burrau)",
        "Masses 3, 4 and 5 released from rest at the vertices of a 3-4-5 triangle. "
        "A long chaotic dance with near-collisions ends when the two heavier "
        "bodies form a tight binary and the lightest is ejected.",
        pythagorean,
        SimulationConfig(
            periods=70,
            method=Method.DOP853,
            max_samples=20_000,
            lyapunov_renorm_per_period=1,
        ),
        animate_periods=None,
    ),
    "swing-spring": Demo(
        "Swing spring: 2:1 resonant elastic pendulum",
        "Spring frequency twice the pendulum frequency. Energy pumps from bouncing "
        "to swinging and back, and the swing plane precesses in discrete steps "
        "(watch the top view).",
        lambda: SpringPendulum(theta0=0.03, omega0=0.0, stretch0=0.1, v_perp0=0.02),
        SimulationConfig(periods=80, steps_per_period=1000),
        animate_periods=24.0,
    ),
}
