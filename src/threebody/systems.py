"""Dynamical systems: equations of motion, invariants and plotting hooks.

Each model is a frozen dataclass deriving from :class:`DynamicalSystem`.
Models are written in coordinates with a diagonal (Cartesian) kinetic
energy, so ``q'' = a(q)`` and the symplectic integrators apply directly.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any, ClassVar

import numpy as np
import numpy.typing as npt
from scipy.integrate import quad
from scipy.optimize import brentq

from threebody import _compiled
from threebody.integrators import FloatArray

IntArray = npt.NDArray[np.int64]


@dataclass(frozen=True)
class PhasePortrait:
    """A pair of sampled series to be drawn against each other."""

    title: str
    x_label: str
    y_label: str
    x: FloatArray
    y: FloatArray
    periodic_x: float | None = None


@dataclass(frozen=True)
class Section:
    """A Poincare section or return map built from recorded events."""

    title: str
    x_label: str
    y_label: str
    x: FloatArray
    y: FloatArray
    t: FloatArray


@dataclass(frozen=True)
class EventTable:
    """Events recorded during integration (sign changes of event scalars)."""

    event_id: IntArray
    direction: FloatArray
    t: FloatArray
    q: FloatArray
    v: FloatArray

    @classmethod
    def from_rows(cls, rows: FloatArray, dim: int) -> EventTable:
        rows = rows.reshape(-1, 3 + 2 * dim)
        return cls(
            rows[:, 0].astype(np.int64),
            rows[:, 1],
            rows[:, 2],
            rows[:, 3 : 3 + dim],
            rows[:, 3 + dim :],
        )

    def select(self, event_id: int, direction: float | None = None) -> EventTable:
        """Return the events of one type, optionally in one direction."""
        mask = self.event_id == event_id
        if direction is not None:
            mask &= self.direction == direction
        return EventTable(
            self.event_id[mask],
            self.direction[mask],
            self.t[mask],
            self.q[mask],
            self.v[mask],
        )

    def __len__(self) -> int:
        return int(self.t.shape[0])


class DynamicalSystem(ABC):
    """Interface shared by all models."""

    name: ClassVar[str]
    kernel: ClassVar[int]
    event_names: ClassVar[tuple[str, ...]]
    body_labels: ClassVar[tuple[str, ...]]
    links: ClassVar[tuple[tuple[int, int], ...]] = ()

    @property
    @abstractmethod
    def params(self) -> FloatArray:
        """Parameter vector handed to the compiled kernels."""

    @abstractmethod
    def initial_state(self) -> tuple[FloatArray, FloatArray]:
        """Initial positions and velocities."""

    @abstractmethod
    def characteristic_time(self) -> float:
        """A natural time unit (orbital period, oscillator period, ...)."""

    @abstractmethod
    def invariants(self, q: FloatArray, v: FloatArray) -> dict[str, FloatArray]:
        """Conserved quantities evaluated on samples; the first is energy."""

    @abstractmethod
    def body_positions(self, q: FloatArray) -> FloatArray:
        """Body positions in 3D, shape ``(n_samples, n_bodies, 3)``."""

    @abstractmethod
    def body_velocities(self, q: FloatArray, v: FloatArray) -> FloatArray:
        """Body velocities in 3D, shape ``(n_samples, n_bodies, 3)``."""

    @abstractmethod
    def columns(
        self, t: FloatArray, q: FloatArray, v: FloatArray
    ) -> dict[str, FloatArray]:
        """Named columns written to ``trajectory.csv``."""

    @abstractmethod
    def phase_portraits(
        self, t: FloatArray, q: FloatArray, v: FloatArray
    ) -> list[PhasePortrait]:
        """Phase-space projections worth plotting."""

    @abstractmethod
    def sections(self, events: EventTable) -> list[Section]:
        """Poincare sections / return maps built from events."""

    def body_masses(self) -> FloatArray:
        """Relative body masses (used for marker sizes)."""
        return np.ones(len(self.body_labels))

    def extra_curves(
        self, t: FloatArray, q: FloatArray, v: FloatArray
    ) -> dict[str, FloatArray]:
        """Additional 3D curves for the orbit view, each ``(n_samples, 3)``."""
        return {}

    def distances(self, q: FloatArray) -> dict[str, FloatArray]:
        """Pairwise body separations."""
        pos = self.body_positions(q)
        out: dict[str, FloatArray] = {}
        n = pos.shape[1]
        for i in range(n):
            for j in range(i + 1, n):
                key = f"|{self.body_labels[i]} - {self.body_labels[j]}|"
                out[key] = np.linalg.norm(pos[:, i] - pos[:, j], axis=1)
        return out

    def spectrum_signal(self, q: FloatArray, v: FloatArray) -> tuple[str, FloatArray]:
        """A scalar signal whose power spectrum characterises the motion."""
        return "q[0]", q[:, 0]

    def describe(self) -> dict[str, Any]:
        """JSON-friendly description of the model parameters."""
        d = asdict(self)  # type: ignore[call-overload]
        return {"model": self.name, **{k: _jsonable(x) for k, x in d.items()}}


def _jsonable(x: Any) -> Any:
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, tuple):
        return [_jsonable(i) for i in x]
    return x


# ==========================================================================
# Simple harmonic oscillator (verification problem)
# ==========================================================================


@dataclass(frozen=True)
class HarmonicOscillator(DynamicalSystem):
    """``x'' = -omega^2 x``: the verification problem with a known solution."""

    omega: float = 1.0
    x0: float = 1.0
    v0: float = 0.0

    name: ClassVar[str] = "harmonic"
    kernel: ClassVar[int] = _compiled.HARMONIC
    event_names: ClassVar[tuple[str, ...]] = ("x = 0 crossing",)
    body_labels: ClassVar[tuple[str, ...]] = ("mass",)

    @property
    def params(self) -> FloatArray:
        return np.array([self.omega])

    def initial_state(self) -> tuple[FloatArray, FloatArray]:
        return np.array([self.x0]), np.array([self.v0])

    def characteristic_time(self) -> float:
        return 2.0 * math.pi / self.omega

    def exact(self, t: FloatArray) -> tuple[FloatArray, FloatArray]:
        """Exact position and velocity at times ``t``."""
        w = self.omega
        x = self.x0 * np.cos(w * t) + self.v0 / w * np.sin(w * t)
        v = -self.x0 * w * np.sin(w * t) + self.v0 * np.cos(w * t)
        return x, v

    def invariants(self, q: FloatArray, v: FloatArray) -> dict[str, FloatArray]:
        energy = 0.5 * v[:, 0] ** 2 + 0.5 * self.omega**2 * q[:, 0] ** 2
        return {"energy": energy}

    def body_positions(self, q: FloatArray) -> FloatArray:
        pos = np.zeros((q.shape[0], 1, 3))
        pos[:, 0, 0] = q[:, 0]
        return pos

    def body_velocities(self, q: FloatArray, v: FloatArray) -> FloatArray:
        vel = np.zeros((v.shape[0], 1, 3))
        vel[:, 0, 0] = v[:, 0]
        return vel

    def columns(
        self, t: FloatArray, q: FloatArray, v: FloatArray
    ) -> dict[str, FloatArray]:
        x_exact, v_exact = self.exact(t)
        return {
            "t": t,
            "x": q[:, 0],
            "v": v[:, 0],
            "x_exact": x_exact,
            "v_exact": v_exact,
            "x_error": q[:, 0] - x_exact,
        }

    def phase_portraits(
        self, t: FloatArray, q: FloatArray, v: FloatArray
    ) -> list[PhasePortrait]:
        x_exact, _ = self.exact(t)
        return [
            PhasePortrait("Phase plane", "x", "v", q[:, 0], v[:, 0]),
            PhasePortrait(
                "Global error vs time", "t", "x - x_exact", t, q[:, 0] - x_exact
            ),
        ]

    def sections(self, events: EventTable) -> list[Section]:
        up = events.select(0, 1.0)
        if len(up) < 2:
            return []
        return [
            Section(
                "Measured period at each upward zero crossing",
                "t",
                "period",
                up.t[1:],
                np.diff(up.t),
                up.t[1:],
            )
        ]


# ==========================================================================
# Ekeland's symmetric three-body problem (Sitnikov-type)
# ==========================================================================


@dataclass(frozen=True)
class SymmetricThreeBody(DynamicalSystem):
    """Two equal stars on a Kepler orbit plus a comet on their symmetry axis.

    This is Ekeland's version of the Sitnikov problem.  The comet has mass
    ``mu * M`` and back-reacts on the stars, so the stellar plane recoils
    along the axis (coordinate ``y``) and the binary separation breathes.
    With ``mu -> 0`` and ``eccentricity > 0`` it is the classic (chaotic)
    Sitnikov problem; with ``eccentricity = 0`` the comet's motion is
    integrable.

    Default units are parsec, km/s and solar masses,
    so the time unit is ~0.978 Myr.

    Attributes:
        G: Gravitational constant.
        M: Mass of each star.
        r0: Initial distance of each star from the axis (half separation).
        mu: Comet mass in units of ``M``.
        v0: Initial comet speed relative to the stellar plane.
        z0: Initial comet height above the stellar plane.
        eccentricity: Binary eccentricity; stars start at pericentre.
    """

    G: float = 4.302e-3
    M: float = 1.0
    r0: float = 0.5
    mu: float = 1e-8
    v0: float = 0.17
    z0: float = 0.0
    eccentricity: float = 0.0

    name: ClassVar[str] = "symmetric"
    kernel: ClassVar[int] = _compiled.SYMMETRIC
    event_names: ClassVar[tuple[str, ...]] = (
        "comet crosses stellar plane",
        "binary phase = 0",
    )
    body_labels: ClassVar[tuple[str, ...]] = ("star A", "star B", "comet")

    def __post_init__(self) -> None:
        if not 0.0 <= self.eccentricity < 1.0:
            raise ValueError("eccentricity must be in [0, 1)")
        if self.mu < 0.0 or self.r0 <= 0.0 or self.M <= 0.0:
            raise ValueError("mu must be >= 0 and r0, M > 0")

    @property
    def params(self) -> FloatArray:
        return np.array([self.G, self.M, self.mu])

    # ---- analytic reference values ---------------------------------------

    @property
    def escape_velocity(self) -> float:
        """Comet escape speed from the plane (circular binary, mu -> 0)."""
        return math.sqrt(4.0 * self.G * self.M / self.r0)

    def binary_period(self) -> float:
        """Kepler period of the binary (exact for mu -> 0)."""
        a_rel = 2.0 * self.r0 / (1.0 - self.eccentricity)
        return 2.0 * math.pi * math.sqrt(a_rel**3 / (2.0 * self.G * self.M))

    def small_oscillation_period(self) -> float:
        """Comet period for small amplitude about a circular binary."""
        return 2.0 * math.pi * math.sqrt(self.r0**3 / (2.0 * self.G * self.M))

    def comet_period(self) -> float:
        """Exact comet period for a circular binary and mu -> 0 (quadrature).

        Returns ``inf`` at or above escape velocity.
        """
        return comet_period(self.G, self.M, self.r0, self.v0, self.z0)

    @classmethod
    def resonant(cls, ratio: float, **kwargs: Any) -> SymmetricThreeBody:
        """Model whose comet period equals ``ratio`` binary periods.

        Solved for ``v0`` with ``eccentricity = 0`` and ``z0 = 0`` using the
        quadrature period, so the orbit closes after ``lcm`` of the periods.
        """
        base = cls(**{**kwargs, "v0": 0.0, "z0": 0.0, "eccentricity": 0.0})
        target = ratio * base.binary_period()
        if target <= base.small_oscillation_period():
            raise ValueError(
                f"ratio {ratio} below the small-oscillation limit "
                f"{base.small_oscillation_period() / base.binary_period():.4f}"
            )
        v_esc = base.escape_velocity

        def f(v: float) -> float:
            return comet_period(base.G, base.M, base.r0, v, 0.0) - target

        v0 = brentq(f, 1e-9 * v_esc, v_esc * (1.0 - 1e-12), xtol=1e-15, rtol=1e-15)
        return cls(**{**kwargs, "v0": float(v0), "z0": 0.0, "eccentricity": 0.0})

    # ---- dynamics ---------------------------------------------------------

    def initial_state(self) -> tuple[FloatArray, FloatArray]:
        mu = self.mu
        # Centre-of-mass frame: mu*z + 2*y = 0 and mu*z' + 2*y' = 0.
        y0 = -mu * self.z0 / (2.0 + mu)
        z0 = y0 + self.z0
        vy = -mu * self.v0 / (2.0 + mu)
        vz = vy + self.v0
        d = math.hypot(self.r0, self.z0)
        radial = self.G * self.M * (0.25 / self.r0**2 + mu * self.r0 / d**3)
        omega = math.sqrt(radial / self.r0 * (1.0 + self.eccentricity))
        q = np.array([self.r0, 0.0, y0, z0])
        v = np.array([0.0, self.r0 * omega, vy, vz])
        return q, v

    def characteristic_time(self) -> float:
        return self.binary_period()

    def invariants(self, q: FloatArray, v: FloatArray) -> dict[str, FloatArray]:
        big_g, m, mu = self.G, self.M, self.mu
        rho = np.hypot(q[:, 0], q[:, 1])
        s = q[:, 3] - q[:, 2]
        d = np.sqrt(rho**2 + s**2)
        kinetic = (v[:, 0] ** 2 + v[:, 1] ** 2 + v[:, 2] ** 2) + 0.5 * mu * v[:, 3] ** 2
        potential = -big_g * m / (2.0 * rho) - 2.0 * big_g * m * mu / d
        out = {
            "energy (per unit M)": kinetic + potential,
            "angular momentum Lz": 2.0 * (q[:, 0] * v[:, 1] - q[:, 1] * v[:, 0]),
        }
        if self.eccentricity == 0.0 and mu < 1e-6:
            vrel = v[:, 3] - v[:, 2]
            out["comet energy (exact as mu -> 0)"] = 0.5 * vrel**2 - 2.0 * big_g * m / d
        return out

    def body_positions(self, q: FloatArray) -> FloatArray:
        n = q.shape[0]
        pos = np.zeros((n, 3, 3))
        pos[:, 0] = np.column_stack([q[:, 0], q[:, 1], q[:, 2]])
        pos[:, 1] = np.column_stack([-q[:, 0], -q[:, 1], q[:, 2]])
        pos[:, 2, 2] = q[:, 3]
        return pos

    def body_velocities(self, q: FloatArray, v: FloatArray) -> FloatArray:
        n = v.shape[0]
        vel = np.zeros((n, 3, 3))
        vel[:, 0] = np.column_stack([v[:, 0], v[:, 1], v[:, 2]])
        vel[:, 1] = np.column_stack([-v[:, 0], -v[:, 1], v[:, 2]])
        vel[:, 2, 2] = v[:, 3]
        return vel

    def body_masses(self) -> FloatArray:
        return np.array([1.0, 1.0, max(self.mu, 0.05)])

    @staticmethod
    def _polar(q: FloatArray, v: FloatArray) -> tuple[FloatArray, ...]:
        rho = np.hypot(q[:, 0], q[:, 1])
        vr = (q[:, 0] * v[:, 0] + q[:, 1] * v[:, 1]) / rho
        theta = np.mod(np.arctan2(q[:, 1], q[:, 0]), 2.0 * math.pi)
        w = (q[:, 0] * v[:, 1] - q[:, 1] * v[:, 0]) / rho**2
        return rho, vr, theta, w

    def columns(
        self, t: FloatArray, q: FloatArray, v: FloatArray
    ) -> dict[str, FloatArray]:
        rho, vr, theta, w = self._polar(q, v)
        return {
            "t": t,
            "r": rho,
            "vr": vr,
            "z": q[:, 3],
            "vz": v[:, 3],
            "y": q[:, 2],
            "vy": v[:, 2],
            "theta": theta,
            "w": w,
        }

    def distances(self, q: FloatArray) -> dict[str, FloatArray]:
        rho = np.hypot(q[:, 0], q[:, 1])
        s = q[:, 3] - q[:, 2]
        return {
            "star separation": 2.0 * rho,
            "comet height above plane": np.abs(s),
            "comet-star distance": np.sqrt(rho**2 + s**2),
        }

    def phase_portraits(
        self, t: FloatArray, q: FloatArray, v: FloatArray
    ) -> list[PhasePortrait]:
        rho, vr, theta, _ = self._polar(q, v)
        return [
            PhasePortrait("Comet (z, vz)", "z", "vz", q[:, 3], v[:, 3]),
            PhasePortrait("Binary radius (r, vr)", "r", "vr", rho, vr),
            PhasePortrait("Stellar plane recoil (y, vy)", "y", "vy", q[:, 2], v[:, 2]),
            PhasePortrait(
                "Comet height vs binary phase",
                "theta",
                "z - y",
                theta,
                q[:, 3] - q[:, 2],
                periodic_x=2.0 * math.pi,
            ),
        ]

    def sections(self, events: EventTable) -> list[Section]:
        out: list[Section] = []
        strobe = events.select(1, 1.0)
        if len(strobe) > 1:
            out.append(
                Section(
                    "Stroboscopic section at binary phase 0 (Sitnikov map)",
                    "z - y",
                    "vz - vy",
                    strobe.q[:, 3] - strobe.q[:, 2],
                    strobe.v[:, 3] - strobe.v[:, 2],
                    strobe.t,
                )
            )
        cross = events.select(0, 1.0)
        if len(cross) > 2:
            theta = np.mod(np.arctan2(cross.q[:, 1], cross.q[:, 0]), 2.0 * math.pi)
            out.append(
                Section(
                    "Return map: binary phase at successive upward plane crossings",
                    "theta_n",
                    "theta_n+1",
                    theta[:-1],
                    theta[1:],
                    cross.t[1:],
                )
            )
            out.append(
                Section(
                    "Comet period between upward plane crossings",
                    "t",
                    "period",
                    cross.t[1:],
                    np.diff(cross.t),
                    cross.t[1:],
                )
            )
        return out

    def extra_curves(
        self, t: FloatArray, q: FloatArray, v: FloatArray
    ) -> dict[str, FloatArray]:
        # The comet's height drawn on the cylinder swept by star A: a torus-
        # like embedding of (binary phase, comet height).  Rational period
        # ratios close, irrational ones fill a band, chaos scatters.
        rho, _, theta, _ = self._polar(q, v)
        r = float(np.mean(rho))
        curve = np.column_stack(
            [r * np.cos(theta), r * np.sin(theta), q[:, 3] - q[:, 2]]
        )
        return {"comet height on binary cylinder": curve}

    def spectrum_signal(self, q: FloatArray, v: FloatArray) -> tuple[str, FloatArray]:
        return "comet height z - y", q[:, 3] - q[:, 2]


def comet_period(big_g: float, m: float, r: float, v0: float, z0: float) -> float:
    """Comet period on the axis of a fixed circular binary (mu -> 0).

    With ``d = sqrt(r^2 + z^2)`` the energy equation gives
    ``z'^2 = 4 G M (1/d - 1/d_max)``.  Substituting ``z = z_max sin(phi)`` and
    using ``1/d - 1/d_max = z_max^2 cos^2(phi) / (d d_max (d + d_max))``
    removes both the turning-point singularity and the cancellation at small
    amplitude, leaving a smooth integrand:

        T = 4 * integral_0^(pi/2) sqrt(d d_max (d + d_max) / (4 G M)) dphi

    Returns ``inf`` at or above escape velocity.
    """
    gm = big_g * m
    d0 = math.hypot(r, z0)
    energy = 0.5 * v0**2 - 2.0 * gm / d0
    if energy >= 0.0:
        return math.inf
    d_max = -2.0 * gm / energy
    z_max = math.sqrt(max(d_max**2 - r**2, 0.0))

    def integrand(phi: float) -> float:
        d = math.hypot(r, z_max * math.sin(phi))
        return math.sqrt(d * d_max * (d + d_max) / (4.0 * gm))

    value, _ = quad(integrand, 0.0, 0.5 * math.pi, epsabs=0.0, epsrel=1e-13)
    return 4.0 * value


# ==========================================================================
# General gravitational N-body problem in 3D
# ==========================================================================


@dataclass(frozen=True)
class NBody(DynamicalSystem):
    """Newtonian point masses in 3D (optionally Plummer-softened).

    Attributes:
        masses: Body masses.
        positions: Initial positions, one ``(x, y, z)`` triple per body.
        velocities: Initial velocities, one triple per body.
        G: Gravitational constant.
        softening: Plummer softening length (0 for exact gravity).
        period: Known period of the orbit, if any (used as time unit).
        labels: Body labels.
    """

    masses: tuple[float, ...]
    positions: tuple[tuple[float, float, float], ...]
    velocities: tuple[tuple[float, float, float], ...]
    G: float = 1.0
    softening: float = 0.0
    period: float | None = None
    labels: tuple[str, ...] = field(default=())

    name: ClassVar[str] = "nbody"
    kernel: ClassVar[int] = _compiled.NBODY
    event_names: ClassVar[tuple[str, ...]] = ("body 1 crosses y = 0",)
    body_labels: ClassVar[tuple[str, ...]] = ()

    def __post_init__(self) -> None:
        n = len(self.masses)
        if len(self.positions) != n or len(self.velocities) != n:
            raise ValueError("masses, positions and velocities must match")
        if not self.labels:
            object.__setattr__(self, "labels", tuple(f"body {i + 1}" for i in range(n)))
        # Instance-level labels shadow the empty ClassVar used by the base.
        object.__setattr__(self, "body_labels", self.labels)

    @property
    def n_bodies(self) -> int:
        return len(self.masses)

    @property
    def params(self) -> FloatArray:
        return np.array([self.G, self.softening**2, *self.masses])

    def initial_state(self) -> tuple[FloatArray, FloatArray]:
        return (
            np.asarray(self.positions, dtype=np.float64).ravel(),
            np.asarray(self.velocities, dtype=np.float64).ravel(),
        )

    def body_masses(self) -> FloatArray:
        return np.asarray(self.masses, dtype=np.float64)

    def characteristic_time(self) -> float:
        if self.period is not None:
            return self.period
        q, v = self.initial_state()
        energy = float(self.invariants(q[None], v[None])["energy"][0])
        m_tot = float(np.sum(self.masses))
        if energy >= 0.0:
            return 1.0
        # Standard N-body crossing time.
        return float(self.G * m_tot**2.5 / (2.0 * abs(energy)) ** 1.5)

    def invariants(self, q: FloatArray, v: FloatArray) -> dict[str, FloatArray]:
        m = self.body_masses()
        pos = q.reshape(q.shape[0], -1, 3)
        vel = v.reshape(v.shape[0], -1, 3)
        kinetic = 0.5 * np.einsum("j,ijk,ijk->i", m, vel, vel)
        potential = np.zeros(q.shape[0])
        eps2 = self.softening**2
        for i in range(self.n_bodies):
            for j in range(i + 1, self.n_bodies):
                r = np.sqrt(np.sum((pos[:, i] - pos[:, j]) ** 2, axis=1) + eps2)
                potential -= self.G * m[i] * m[j] / r
        ang = np.einsum("j,ijk->ik", m, np.cross(pos, vel))
        mom = np.einsum("j,ijk->ik", m, vel)
        return {
            "energy": kinetic + potential,
            "angular momentum Lz": ang[:, 2],
            "|angular momentum|": np.linalg.norm(ang, axis=1),
            "|linear momentum|": np.linalg.norm(mom, axis=1),
        }

    def body_positions(self, q: FloatArray) -> FloatArray:
        return q.reshape(q.shape[0], -1, 3)

    def body_velocities(self, q: FloatArray, v: FloatArray) -> FloatArray:
        return v.reshape(v.shape[0], -1, 3)

    def columns(
        self, t: FloatArray, q: FloatArray, v: FloatArray
    ) -> dict[str, FloatArray]:
        cols: dict[str, FloatArray] = {"t": t}
        for i in range(self.n_bodies):
            for k, axis in enumerate("xyz"):
                cols[f"{axis}{i + 1}"] = q[:, 3 * i + k]
            for k, axis in enumerate("xyz"):
                cols[f"v{axis}{i + 1}"] = v[:, 3 * i + k]
        return cols

    def phase_portraits(
        self, t: FloatArray, q: FloatArray, v: FloatArray
    ) -> list[PhasePortrait]:
        out = []
        for i in range(min(self.n_bodies, 3)):
            label = self.body_labels[i]
            out.append(
                PhasePortrait(f"{label}: (x, vx)", "x", "vx", q[:, 3 * i], v[:, 3 * i])
            )
        # Shape-sphere style invariant view: separations of the first pairs.
        if self.n_bodies >= 3:
            pos = self.body_positions(q)
            r12 = np.linalg.norm(pos[:, 0] - pos[:, 1], axis=1)
            r13 = np.linalg.norm(pos[:, 0] - pos[:, 2], axis=1)
            out.append(
                PhasePortrait("Separation plane (r12, r13)", "r12", "r13", r12, r13)
            )
        return out

    def sections(self, events: EventTable) -> list[Section]:
        up = events.select(0, 1.0)
        if len(up) < 2:
            return []
        return [
            Section(
                f"{self.body_labels[0]} upward through y = 0: (x, vx)",
                "x",
                "vx",
                up.q[:, 0],
                up.v[:, 0],
                up.t,
            )
        ]

    def spectrum_signal(self, q: FloatArray, v: FloatArray) -> tuple[str, FloatArray]:
        return f"x of {self.body_labels[0]}", q[:, 0]


# ==========================================================================
# Elastic (spring) pendulum in 3D
# ==========================================================================


@dataclass(frozen=True)
class SpringPendulum(DynamicalSystem):
    """Elastic pendulum (the "swing spring") in 3D Cartesian coordinates.

    The spring constant is set from ``ratio``, the ratio of the spring to
    pendulum frequencies at equilibrium: ``k/m = (ratio^2 - 1) g / L0``.
    ``ratio = 2`` is the 2:1 autoparametric resonance in which energy
    sloshes between bouncing and swinging and the swing plane precesses in
    discrete steps.

    Attributes:
        g: Gravitational acceleration.
        L0: Unstretched spring length.
        ratio: Spring/pendulum frequency ratio.
        theta0: Initial angle from the downward vertical, in the x-z plane.
        omega0: Initial angular velocity in the x-z plane.
        stretch0: Initial extension beyond the equilibrium length.
        v_perp0: Initial velocity out of the x-z plane (along y).
    """

    g: float = 9.81
    L0: float = 1.0
    ratio: float = 2.0
    theta0: float = 0.0
    omega0: float = 1.0
    stretch0: float = 0.0
    v_perp0: float = 0.0

    name: ClassVar[str] = "pendulum"
    kernel: ClassVar[int] = _compiled.SPRING
    event_names: ClassVar[tuple[str, ...]] = ("bob passes under the pivot (x = 0)",)
    body_labels: ClassVar[tuple[str, ...]] = ("bob", "pivot")
    links: ClassVar[tuple[tuple[int, int], ...]] = ((1, 0),)

    def __post_init__(self) -> None:
        if self.ratio <= 1.0:
            raise ValueError("ratio must exceed 1 for a positive spring constant")

    @property
    def k_over_m(self) -> float:
        return (self.ratio**2 - 1.0) * self.g / self.L0

    @property
    def equilibrium_length(self) -> float:
        return self.L0 + self.g / self.k_over_m

    @property
    def params(self) -> FloatArray:
        return np.array([self.g, self.L0, self.k_over_m])

    def initial_state(self) -> tuple[FloatArray, FloatArray]:
        ell = self.equilibrium_length + self.stretch0
        s, c = math.sin(self.theta0), math.cos(self.theta0)
        q = np.array([ell * s, 0.0, -ell * c])
        v = np.array([ell * self.omega0 * c, self.v_perp0, ell * self.omega0 * s])
        return q, v

    def characteristic_time(self) -> float:
        return 2.0 * math.pi * math.sqrt(self.equilibrium_length / self.g)

    def invariants(self, q: FloatArray, v: FloatArray) -> dict[str, FloatArray]:
        ell = np.linalg.norm(q, axis=1)
        energy = (
            0.5 * np.sum(v * v, axis=1)
            + 0.5 * self.k_over_m * (ell - self.L0) ** 2
            + self.g * q[:, 2]
        )
        lz = q[:, 0] * v[:, 1] - q[:, 1] * v[:, 0]
        return {"energy (per unit mass)": energy, "vertical angular momentum Lz": lz}

    def body_positions(self, q: FloatArray) -> FloatArray:
        pos = np.zeros((q.shape[0], 2, 3))
        pos[:, 0] = q
        return pos

    def body_velocities(self, q: FloatArray, v: FloatArray) -> FloatArray:
        vel = np.zeros((v.shape[0], 2, 3))
        vel[:, 0] = v
        return vel

    def body_masses(self) -> FloatArray:
        return np.array([1.0, 0.3])

    def distances(self, q: FloatArray) -> dict[str, FloatArray]:
        return {"spring length": np.linalg.norm(q, axis=1)}

    @staticmethod
    def _spherical(q: FloatArray, v: FloatArray) -> dict[str, FloatArray]:
        ell = np.linalg.norm(q, axis=1)
        v_l = np.sum(q * v, axis=1) / ell
        theta = np.arctan2(q[:, 0], -q[:, 2])  # planar swing angle (x-z plane)
        omega = (-q[:, 2] * v[:, 0] + q[:, 0] * v[:, 2]) / (q[:, 0] ** 2 + q[:, 2] ** 2)
        polar = np.arccos(np.clip(-q[:, 2] / ell, -1.0, 1.0))
        azimuth = np.arctan2(q[:, 1], q[:, 0])
        return {
            "l": ell,
            "v_l": v_l,
            "theta": theta,
            "omega": omega,
            "polar": polar,
            "azimuth": azimuth,
        }

    def columns(
        self, t: FloatArray, q: FloatArray, v: FloatArray
    ) -> dict[str, FloatArray]:
        sph = self._spherical(q, v)
        return {
            "t": t,
            "x": q[:, 0],
            "y": q[:, 1],
            "z": q[:, 2],
            "vx": v[:, 0],
            "vy": v[:, 1],
            "vz": v[:, 2],
            **sph,
        }

    def phase_portraits(
        self, t: FloatArray, q: FloatArray, v: FloatArray
    ) -> list[PhasePortrait]:
        sph = self._spherical(q, v)
        return [
            PhasePortrait(
                "Swing (theta, omega)",
                "theta",
                "omega",
                sph["theta"],
                sph["omega"],
                periodic_x=2.0 * math.pi,
            ),
            PhasePortrait("Spring (l, v_l)", "l", "v_l", sph["l"], sph["v_l"]),
            PhasePortrait("Top view (x, y)", "x", "y", q[:, 0], q[:, 1]),
            PhasePortrait("Side view (x, z)", "x", "z", q[:, 0], q[:, 2]),
        ]

    def sections(self, events: EventTable) -> list[Section]:
        up = events.select(0, 1.0)
        if len(up) < 2:
            return []
        ell = np.linalg.norm(up.q, axis=1)
        v_l = np.sum(up.q * up.v, axis=1) / ell
        return [
            Section(
                "Section at x = 0 (upward): spring (l, v_l)",
                "l",
                "v_l",
                ell,
                v_l,
                up.t,
            ),
            Section(
                "Swing period between upward passes",
                "t",
                "period",
                up.t[1:],
                np.diff(up.t),
                up.t[1:],
            ),
        ]

    def spectrum_signal(self, q: FloatArray, v: FloatArray) -> tuple[str, FloatArray]:
        return "spring length", np.linalg.norm(q, axis=1)


MODELS: dict[str, type[DynamicalSystem]] = {
    "harmonic": HarmonicOscillator,
    "symmetric": SymmetricThreeBody,
    "nbody": NBody,
    "pendulum": SpringPendulum,
}
