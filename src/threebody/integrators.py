"""Integrators for separable second-order systems ``q'' = a(q)``.

Every model in this package is written in coordinates where the kinetic
energy is a diagonal quadratic form in the velocities, so Hamilton's
equations split into ``q' = v`` and ``v' = a(q)``.  That makes the
leapfrog (Stoermer-Verlet) map exactly symplectic and lets us build
higher-order symplectic schemes by composition (Yoshida 1990).

The hot loops are compiled with numba.  An acceleration function has the
signature ``accel(q, params, out) -> None`` and an event function has the
signature ``events(q, v, params, out) -> None``; both are ``@njit``
functions that write into ``out``.  Event functions return one scalar per
event type and an event is recorded whenever a scalar changes sign.

An adaptive Dormand-Prince 8(5,3) path (``"dop853"``) is provided for
problems with close encounters, where a fixed step is a poor fit.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import numpy as np
from scipy.integrate import solve_ivp

from threebody import _compiled

FloatArray = _compiled.FloatArray


class Method(StrEnum):
    """Available integration methods."""

    LEAPFROG = "leapfrog"
    YOSHIDA4 = "yoshida4"
    YOSHIDA6 = "yoshida6"
    DOP853 = "dop853"

    @property
    def order(self) -> int:
        """Global order of accuracy."""
        return {"leapfrog": 2, "yoshida4": 4, "yoshida6": 6, "dop853": 8}[self.value]

    @property
    def symplectic(self) -> bool:
        """Whether the method is a symplectic map."""
        return self is not Method.DOP853


_CBRT2 = 2.0 ** (1.0 / 3.0)
_Y4_OUTER = 1.0 / (2.0 - _CBRT2)
_Y4_INNER = -_CBRT2 / (2.0 - _CBRT2)
# Yoshida (1990) sixth-order "solution A".
_Y6 = (-1.17767998417887, 0.235573213359357, 0.784513610477560)
_Y6_CENTRE = 1.0 - 2.0 * sum(_Y6)

COMPOSITION_COEFFICIENTS: dict[Method, FloatArray] = {
    Method.LEAPFROG: np.array([1.0]),
    Method.YOSHIDA4: np.array([_Y4_OUTER, _Y4_INNER, _Y4_OUTER]),
    Method.YOSHIDA6: np.array(
        [_Y6[2], _Y6[1], _Y6[0], _Y6_CENTRE, _Y6[0], _Y6[1], _Y6[2]]
    ),
}


@dataclass(frozen=True)
class RawTrajectory:
    """Arrays produced by an integration run.

    Attributes:
        t: Sample times, shape ``(n_samples,)``.
        q: Sampled positions, shape ``(n_samples, dim)``.
        v: Sampled velocities, shape ``(n_samples, dim)``.
        events: One row per event: ``[event_id, direction, t, q..., v...]``.
        lyapunov_t: Times at which the Lyapunov estimate was updated.
        lyapunov: Running finite-time maximal Lyapunov exponent estimate.
        n_steps: Number of integration steps taken.
        n_force_evaluations: Number of acceleration evaluations.
    """

    t: FloatArray
    q: FloatArray
    v: FloatArray
    events: FloatArray
    lyapunov_t: FloatArray
    lyapunov: FloatArray
    n_steps: int
    n_force_evaluations: int


# --------------------------------------------------------------------------
# Fixed-step symplectic composition methods (numba)
# --------------------------------------------------------------------------


def _unit_direction(dim: int, seed: int) -> FloatArray:
    rng = np.random.default_rng(seed)
    d = rng.standard_normal(2 * dim)
    return d / np.linalg.norm(d)


def integrate_fixed(
    model: int,
    n_events: int,
    params: FloatArray,
    q0: FloatArray,
    v0: FloatArray,
    dt: float,
    n_steps: int,
    method: Method = Method.YOSHIDA4,
    sample_every: int = 1,
    lyapunov_every: int = 0,
    lyapunov_d0: float = 1e-9,
    seed: int = 0,
) -> RawTrajectory:
    """Integrate with a fixed-step symplectic composition method.

    Args:
        model: Kernel id from :mod:`threebody._compiled`.
        n_events: Number of event scalars the model's event kernel writes.
        params: Model parameter vector passed to the kernels.
        q0: Initial positions.
        v0: Initial velocities.
        dt: Time step.
        n_steps: Number of steps.
        method: One of the symplectic :class:`Method` members.
        sample_every: Store every ``sample_every``-th step.
        lyapunov_every: Renormalise the shadow orbit every this many steps
            (0 disables the Lyapunov estimate).
        lyapunov_d0: Phase-space separation of the shadow orbit.
        seed: Seed for the random initial separation direction.

    Returns:
        The sampled trajectory, events and Lyapunov estimate.
    """
    if method is Method.DOP853:
        raise ValueError("use integrate_adaptive for dop853")
    if dt <= 0 or n_steps <= 0 or sample_every <= 0:
        raise ValueError("dt, n_steps and sample_every must be positive")
    coeffs = COMPOSITION_COEFFICIENTS[method]
    q0 = np.ascontiguousarray(q0, dtype=np.float64)
    v0 = np.ascontiguousarray(v0, dtype=np.float64)
    direction = _unit_direction(q0.shape[0], seed)
    t, q, v, ev, lt, lv = _compiled.integrate_fixed(
        q0,
        v0,
        np.ascontiguousarray(params, dtype=np.float64),
        float(dt),
        int(n_steps),
        int(sample_every),
        coeffs,
        int(model),
        int(n_events),
        int(lyapunov_every),
        direction,
        float(lyapunov_d0),
    )
    n_evals = n_steps * len(coeffs) * (2 if lyapunov_every > 0 else 1)
    return RawTrajectory(t, q, v, ev, lt, lv, n_steps, n_evals)


# --------------------------------------------------------------------------
# Adaptive Dormand-Prince path (scipy)
# --------------------------------------------------------------------------


def integrate_adaptive(
    model: int,
    n_events: int,
    params: FloatArray,
    q0: FloatArray,
    v0: FloatArray,
    t_end: float,
    n_samples: int,
    lyapunov_interval: float = 0.0,
    lyapunov_d0: float = 1e-9,
    rtol: float = 1e-12,
    atol: float = 1e-12,
    seed: int = 0,
) -> RawTrajectory:
    """Integrate with scipy's adaptive DOP853.

    The run is split into chunks of length ``lyapunov_interval`` (or a single
    chunk when the Lyapunov estimate is disabled); the shadow orbit is
    integrated alongside the reference orbit and renormalised between chunks.
    """
    params = np.ascontiguousarray(params, dtype=np.float64)
    q0 = np.asarray(q0, dtype=np.float64)
    v0 = np.asarray(v0, dtype=np.float64)
    dim = q0.shape[0]
    use_lyap = lyapunov_interval > 0.0
    n_copies = 2 if use_lyap else 1
    acc = np.empty(dim)
    n_evals = 0

    def rhs(_t: float, y: FloatArray) -> FloatArray:
        nonlocal n_evals
        out = np.empty_like(y)
        for c in range(n_copies):
            base = 2 * dim * c
            q = np.ascontiguousarray(y[base : base + dim])
            _compiled.accel(model, q, params, acc)
            out[base : base + dim] = y[base + dim : base + 2 * dim]
            out[base + dim : base + 2 * dim] = acc
            n_evals += 1
        return out

    g = np.empty(max(n_events, 1))

    def make_event(e: int, direction: float) -> Any:
        def fn(_t: float, y: FloatArray) -> float:
            _compiled.events(
                model,
                np.ascontiguousarray(y[:dim]),
                np.ascontiguousarray(y[dim : 2 * dim]),
                params,
                g,
            )
            return float(g[e])

        fn.direction = direction  # type: ignore[attr-defined]
        return fn

    event_fns = [make_event(e, d) for e in range(n_events) for d in (1.0, -1.0)]

    t_samples = np.linspace(0.0, t_end, n_samples)
    boundaries = (
        np.append(np.arange(0.0, t_end, lyapunov_interval), t_end)
        if use_lyap
        else np.array([0.0, t_end])
    )
    direction = _unit_direction(dim, seed)
    y = np.concatenate([q0, v0])
    if use_lyap:
        y = np.concatenate([y, y + lyapunov_d0 * direction])

    t_list: list[float] = [0.0]
    y_list: list[FloatArray] = [y[: 2 * dim].copy()]
    event_rows: list[FloatArray] = []
    lyap_t: list[float] = []
    lyap_v: list[float] = []
    log_sum = 0.0
    n_steps = 0

    for t0, t1 in itertools.pairwise(boundaries):
        if t1 <= t0:
            continue
        sol = solve_ivp(
            rhs,
            (t0, t1),
            y,
            method="DOP853",
            dense_output=True,
            events=event_fns or None,
            rtol=rtol,
            atol=atol,
        )
        if not sol.success:
            raise RuntimeError(f"DOP853 failed at t={t0}: {sol.message}")
        n_steps += len(sol.t) - 1
        ts = t_samples[(t_samples > t0) & (t_samples <= t1)]
        if ts.size:
            assert sol.sol is not None  # dense_output=True
            t_list.extend(ts.tolist())
            y_list.extend(sol.sol(ts)[: 2 * dim].T)
        if sol.t_events is not None:
            for k, (te, ye) in enumerate(zip(sol.t_events, sol.y_events, strict=True)):
                e, sign = divmod(k, 2)
                for tt, yy in zip(te, ye, strict=True):
                    row = np.concatenate(
                        [[e, 1.0 if sign == 0 else -1.0, tt], yy[: 2 * dim]]
                    )
                    event_rows.append(row)
        y = sol.y[:, -1].copy()
        if use_lyap:
            delta = y[2 * dim :] - y[: 2 * dim]
            d = float(np.linalg.norm(delta))
            log_sum += math.log(d / lyapunov_d0)
            lyap_t.append(t1)
            lyap_v.append(log_sum / t1)
            y[2 * dim :] = y[: 2 * dim] + delta * (lyapunov_d0 / d)

    traj = np.array(y_list)
    events_arr = (
        np.array(sorted(event_rows, key=lambda r: r[2]))
        if event_rows
        else np.empty((0, 3 + 2 * dim))
    )
    return RawTrajectory(
        np.array(t_list),
        traj[:, :dim],
        traj[:, dim:],
        events_arr,
        np.array(lyap_t),
        np.array(lyap_v),
        n_steps,
        n_evals,
    )
