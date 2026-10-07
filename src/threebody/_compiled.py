"""Compiled (numba) kernels: equations of motion, events and the step loop.

Everything numba compiles lives in this one module and model kernels are
selected by an integer id rather than passed as function objects.  That
keeps every function cacheable on disk (numba cannot cache code
specialised on a function-valued argument), and because numba's cache is
invalidated per source file, editing a kernel correctly recompiles the
loop that inlines it.

Kernel signatures: ``accel(q, params, out)`` writes ``q''`` into ``out``;
``events(q, v, params, out)`` writes one scalar per event type, and an
event is recorded whenever a scalar changes sign.

To add a model: write its two kernels, give it an id below and add a branch
to :func:`accel` and :func:`events`.
"""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt
from numba import njit

FloatArray = npt.NDArray[np.float64]

HARMONIC = 0
SYMMETRIC = 1
NBODY = 2
SPRING = 3


@njit(cache=True)
def _sho_accel(q: FloatArray, p: FloatArray, out: FloatArray) -> None:
    out[0] = -p[0] * p[0] * q[0]


@njit(cache=True)
def _sho_events(q: FloatArray, v: FloatArray, p: FloatArray, out: FloatArray) -> None:
    out[0] = q[0]


@njit(cache=True)
def _sym_accel(q: FloatArray, p: FloatArray, out: FloatArray) -> None:
    # q = [X, Y, y, z]: star A at (X, Y, y), star B at (-X, -Y, y), comet at
    # (0, 0, z).  p = [G, M, mu].
    big_g, m, mu = p[0], p[1], p[2]
    rho2 = q[0] * q[0] + q[1] * q[1]
    rho = math.sqrt(rho2)
    s = q[3] - q[2]
    d2 = rho2 + s * s
    inv_d3 = 1.0 / (d2 * math.sqrt(d2))
    radial = big_g * m * (0.25 / (rho2 * rho) + mu * inv_d3)
    out[0] = -radial * q[0]
    out[1] = -radial * q[1]
    out[2] = big_g * m * mu * s * inv_d3
    out[3] = -2.0 * big_g * m * s * inv_d3


@njit(cache=True)
def _sym_events(q: FloatArray, v: FloatArray, p: FloatArray, out: FloatArray) -> None:
    out[0] = q[3] - q[2]  # comet crosses the stellar plane
    out[1] = q[1]  # star A crosses the +x half-plane: binary phase = 0 (upward)


@njit(cache=True)
def _nbody_accel(q: FloatArray, p: FloatArray, out: FloatArray) -> None:
    # p = [G, eps^2, m_0, ..., m_{n-1}]
    big_g, eps2 = p[0], p[1]
    n = q.shape[0] // 3
    for i in range(3 * n):
        out[i] = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            dx = q[3 * j] - q[3 * i]
            dy = q[3 * j + 1] - q[3 * i + 1]
            dz = q[3 * j + 2] - q[3 * i + 2]
            r2 = dx * dx + dy * dy + dz * dz + eps2
            inv_r3 = big_g / (r2 * math.sqrt(r2))
            fi = p[2 + j] * inv_r3
            fj = p[2 + i] * inv_r3
            out[3 * i] += fi * dx
            out[3 * i + 1] += fi * dy
            out[3 * i + 2] += fi * dz
            out[3 * j] -= fj * dx
            out[3 * j + 1] -= fj * dy
            out[3 * j + 2] -= fj * dz


@njit(cache=True)
def _nbody_events(q: FloatArray, v: FloatArray, p: FloatArray, out: FloatArray) -> None:
    out[0] = q[1]  # body 0 crosses the y = 0 plane


@njit(cache=True)
def _spring_accel(q: FloatArray, p: FloatArray, out: FloatArray) -> None:
    # p = [g, L0, k/m]; pivot at the origin, gravity along -z.
    g, l0, k = p[0], p[1], p[2]
    ell = math.sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2])
    f = -k * (ell - l0) / ell
    out[0] = f * q[0]
    out[1] = f * q[1]
    out[2] = f * q[2] - g


@njit(cache=True)
def _spring_events(
    q: FloatArray, v: FloatArray, p: FloatArray, out: FloatArray
) -> None:
    out[0] = q[0]  # bob crosses the vertical through the pivot (in x)


@njit(cache=True)
def accel(model: int, q: FloatArray, p: FloatArray, out: FloatArray) -> None:
    """Acceleration of ``model`` at ``q``."""
    if model == SYMMETRIC:
        _sym_accel(q, p, out)
    elif model == NBODY:
        _nbody_accel(q, p, out)
    elif model == SPRING:
        _spring_accel(q, p, out)
    else:
        _sho_accel(q, p, out)


@njit(cache=True)
def events(
    model: int, q: FloatArray, v: FloatArray, p: FloatArray, out: FloatArray
) -> None:
    """Event scalars of ``model`` at ``(q, v)``."""
    if model == SYMMETRIC:
        _sym_events(q, v, p, out)
    elif model == NBODY:
        _nbody_events(q, v, p, out)
    elif model == SPRING:
        _spring_events(q, v, p, out)
    else:
        _sho_events(q, v, p, out)


# --------------------------------------------------------------------------
# Fixed-step symplectic composition loop
# --------------------------------------------------------------------------


@njit(cache=True)
def _composed_step(
    q: FloatArray,
    v: FloatArray,
    acc: FloatArray,
    dt: float,
    coeffs: FloatArray,
    model: int,
    params: FloatArray,
) -> None:
    """Advance ``(q, v)`` in place by one composed kick-drift-kick step.

    ``acc`` must hold ``a(q)`` on entry and holds ``a(q)`` on exit, so each
    leapfrog stage costs exactly one force evaluation.
    """
    n = q.shape[0]
    for c in coeffs:
        h = c * dt
        for i in range(n):
            v[i] += 0.5 * h * acc[i]
            q[i] += h * v[i]
        accel(model, q, params, acc)
        for i in range(n):
            v[i] += 0.5 * h * acc[i]


@njit(cache=True)
def _grow(rows: FloatArray) -> FloatArray:
    bigger = np.empty((2 * rows.shape[0], rows.shape[1]))
    bigger[: rows.shape[0]] = rows
    return bigger


@njit(cache=True)
def _hermite(
    s: float,
    dt: float,
    q0: FloatArray,
    v0: FloatArray,
    a0: FloatArray,
    q1: FloatArray,
    v1: FloatArray,
    a1: FloatArray,
    q_out: FloatArray,
    v_out: FloatArray,
) -> None:
    """Cubic Hermite interpolation of a step at fraction ``s`` in [0, 1]."""
    s2 = s * s
    s3 = s2 * s
    h00 = 2.0 * s3 - 3.0 * s2 + 1.0
    h10 = s3 - 2.0 * s2 + s
    h01 = -2.0 * s3 + 3.0 * s2
    h11 = s3 - s2
    for i in range(q0.shape[0]):
        q_out[i] = h00 * q0[i] + h10 * dt * v0[i] + h01 * q1[i] + h11 * dt * v1[i]
        v_out[i] = h00 * v0[i] + h10 * dt * a0[i] + h01 * v1[i] + h11 * dt * a1[i]


@njit(cache=True)
def integrate_fixed(
    q_init: FloatArray,
    v_init: FloatArray,
    params: FloatArray,
    dt: float,
    n_steps: int,
    sample_every: int,
    coeffs: FloatArray,
    model: int,
    n_events: int,
    lyap_every: int,
    lyap_direction: FloatArray,
    lyap_d0: float,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray, FloatArray]:
    dim = q_init.shape[0]
    q = q_init.copy()
    v = v_init.copy()
    acc = np.empty(dim)
    accel(model, q, params, acc)

    n_samples = n_steps // sample_every + 1
    t_out = np.empty(n_samples)
    q_out = np.empty((n_samples, dim))
    v_out = np.empty((n_samples, dim))
    t_out[0] = 0.0
    q_out[0] = q
    v_out[0] = v
    k_sample = 1

    # Event bookkeeping.
    g_prev = np.empty(max(n_events, 1))
    g_new = np.empty(max(n_events, 1))
    g_mid = np.empty(max(n_events, 1))
    if n_events > 0:
        events(model, q, v, params, g_prev)
    records = np.empty((64, 3 + 2 * dim))
    n_rec = 0
    q_prev = q.copy()
    v_prev = v.copy()
    a_prev = acc.copy()
    q_int = np.empty(dim)
    v_int = np.empty(dim)

    # Shadow trajectory for the maximal Lyapunov exponent (Benettin et al.).
    use_lyap = lyap_every > 0
    n_lyap = n_steps // lyap_every if use_lyap else 0
    lyap_t = np.empty(n_lyap)
    lyap_val = np.empty(n_lyap)
    qs = q + lyap_d0 * lyap_direction[:dim]
    vs = v + lyap_d0 * lyap_direction[dim:]
    accs = np.empty(dim)
    if use_lyap:
        accel(model, qs, params, accs)
    log_sum = 0.0
    k_lyap = 0

    for step in range(1, n_steps + 1):
        if n_events > 0:
            q_prev[:] = q
            v_prev[:] = v
            a_prev[:] = acc
        _composed_step(q, v, acc, dt, coeffs, model, params)
        t = step * dt

        if n_events > 0:
            events(model, q, v, params, g_new)
            for e in range(n_events):
                up = g_prev[e] < 0.0 <= g_new[e]
                down = g_prev[e] >= 0.0 > g_new[e]
                if up or down:
                    # Regula falsi on the Hermite interpolant of the step.
                    lo, hi = 0.0, 1.0
                    g_lo, g_hi = g_prev[e], g_new[e]
                    s = lo
                    for _ in range(30):
                        s = lo + (hi - lo) * g_lo / (g_lo - g_hi)
                        _hermite(s, dt, q_prev, v_prev, a_prev, q, v, acc, q_int, v_int)
                        events(model, q_int, v_int, params, g_mid)
                        g_s = g_mid[e]
                        if abs(g_s) < 1e-15 * (abs(g_lo) + abs(g_hi)) or g_s == 0.0:
                            break
                        if (g_s < 0.0) == (g_lo < 0.0):
                            lo, g_lo = s, g_s
                        else:
                            hi, g_hi = s, g_s
                        if hi - lo < 1e-14:
                            break
                    if n_rec == records.shape[0]:
                        records = _grow(records)
                    records[n_rec, 0] = e
                    records[n_rec, 1] = 1.0 if up else -1.0
                    records[n_rec, 2] = t - dt + s * dt
                    records[n_rec, 3 : 3 + dim] = q_int
                    records[n_rec, 3 + dim :] = v_int
                    n_rec += 1
            g_prev[:] = g_new

        if use_lyap:
            _composed_step(qs, vs, accs, dt, coeffs, model, params)
            if step % lyap_every == 0:
                d2 = 0.0
                for i in range(dim):
                    d2 += (qs[i] - q[i]) ** 2 + (vs[i] - v[i]) ** 2
                d = math.sqrt(d2)
                log_sum += math.log(d / lyap_d0)
                lyap_t[k_lyap] = t
                lyap_val[k_lyap] = log_sum / t
                k_lyap += 1
                scale = lyap_d0 / d
                for i in range(dim):
                    qs[i] = q[i] + (qs[i] - q[i]) * scale
                    vs[i] = v[i] + (vs[i] - v[i]) * scale
                accel(model, qs, params, accs)

        if step % sample_every == 0:
            t_out[k_sample] = t
            q_out[k_sample] = q
            v_out[k_sample] = v
            k_sample += 1

    return t_out, q_out, v_out, records[:n_rec].copy(), lyap_t, lyap_val
