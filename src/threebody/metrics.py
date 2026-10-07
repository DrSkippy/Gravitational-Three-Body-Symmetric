"""Diagnostics computed from a finished run."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from threebody.integrators import FloatArray
from threebody.simulate import SimulationResult


@dataclass(frozen=True)
class InvariantDrift:
    """How well a conserved quantity was conserved."""

    name: str
    initial: float
    max_abs_error: float
    max_rel_error: float
    final_rel_error: float


def invariant_drift(name: str, series: FloatArray) -> InvariantDrift:
    """Absolute and relative deviation of ``series`` from its initial value.

    Quantities that start at (numerically) zero, such as total momentum in
    the centre-of-mass frame, are normalised by 1 instead.
    """
    err = series - series[0]
    scale = abs(float(series[0]))
    scale = scale if scale > 1e-12 else 1.0
    return InvariantDrift(
        name=name,
        initial=float(series[0]),
        max_abs_error=float(np.max(np.abs(err))),
        max_rel_error=float(np.max(np.abs(err)) / scale),
        final_rel_error=float(abs(err[-1]) / scale),
    )


def relative_error_series(series: FloatArray) -> FloatArray:
    """``|I(t) - I(0)| / |I(0)|`` (or absolute error when ``I(0) ~ 0``)."""
    scale = abs(float(series[0]))
    scale = scale if scale > 1e-12 else 1.0
    out: FloatArray = np.abs(series - series[0]) / scale
    return out


def power_spectrum(t: FloatArray, signal: FloatArray) -> tuple[FloatArray, FloatArray]:
    """One-sided Hann-windowed power spectrum of a uniformly sampled signal."""
    n = signal.shape[0]
    if n < 8:
        return np.zeros(0), np.zeros(0)
    dt = float(t[1] - t[0])
    x = (signal - np.mean(signal)) * np.hanning(n)
    power = np.abs(np.fft.rfft(x)) ** 2
    freq = np.fft.rfftfreq(n, dt)
    return freq[1:], (
        power[1:] / np.max(power[1:]) if np.max(power[1:]) > 0 else power[1:]
    )


def spectral_entropy(power: FloatArray) -> float:
    """Normalised Shannon entropy of a power spectrum, in [0, 1].

    Line spectra (periodic / quasi-periodic motion) score near 0; broadband
    (chaotic) spectra score higher.
    """
    p = power[power > 0]
    if p.size < 2:
        return 0.0
    p = p / p.sum()
    return float(-np.sum(p * np.log(p)) / math.log(p.size))


def recurrence(
    q: FloatArray, v: FloatArray, t: FloatArray, skip_fraction: float = 0.05
) -> tuple[float, float]:
    """Closest phase-space return to the initial state.

    Distances are normalised per coordinate block by the RMS spread of the
    orbit, so positions and velocities contribute comparably.

    Returns:
        ``(min_distance, time_of_min)`` after the first ``skip_fraction``.
    """
    x = np.hstack([q, v])
    scale = np.std(x, axis=0)
    scale[scale == 0] = 1.0
    d = np.linalg.norm((x - x[0]) / scale, axis=1) / math.sqrt(x.shape[1])
    start = max(1, int(skip_fraction * len(t)))
    k = start + int(np.argmin(d[start:]))
    return float(d[k]), float(t[k])


def classify(lyapunov_per_period: float, e_folds: float) -> str:
    """Heuristic regular/chaotic verdict from a finite-time Lyapunov estimate.

    For regular motion separations grow only linearly, so the finite-time
    estimate decays like ``ln(t) / t`` and the number of e-folds stays near
    ``ln(1/d0) ~ 10-20`` however long the run.  Chaotic motion accumulates
    e-folds in proportion to time.
    """
    if lyapunov_per_period > 0.1 and e_folds > 20.0:
        return "chaotic (positive Lyapunov exponent)"
    if lyapunov_per_period < 0.1:
        return "regular (periodic or quasi-periodic)"
    return "inconclusive (run longer)"


def summarize(result: SimulationResult) -> dict[str, Any]:
    """JSON-friendly summary of a run."""
    system = result.system
    period = system.characteristic_time()
    t, q, v = result.t, result.q, result.v

    invariants = [
        asdict(invariant_drift(k, s)) for k, s in system.invariants(q, v).items()
    ]

    vel = system.body_velocities(q, v)
    speeds = np.linalg.norm(vel, axis=2)
    speed_stats = {
        label: {
            "min": float(speeds[:, i].min()),
            "mean": float(speeds[:, i].mean()),
            "max": float(speeds[:, i].max()),
        }
        for i, label in enumerate(system.body_labels)
        if np.any(speeds[:, i] > 0)
    }
    distance_stats = {
        k: {"min": float(d.min()), "mean": float(d.mean()), "max": float(d.max())}
        for k, d in system.distances(q).items()
    }

    event_stats = {}
    for e, name in enumerate(system.event_names):
        up = result.events.select(e, 1.0)
        intervals = np.diff(up.t)
        event_stats[name] = {
            "count": len(result.events.select(e)),
            "upward": len(up),
            "mean_interval": float(intervals.mean()) if intervals.size else None,
            "std_interval": float(intervals.std()) if intervals.size else None,
        }

    lyap = float(result.lyapunov[-1]) if result.lyapunov.size else None
    sig_name, signal = system.spectrum_signal(q, v)
    _, power = power_spectrum(t, signal)
    min_return, t_return = recurrence(q, v, t)

    verdict = "n/a"
    if lyap is not None:
        verdict = classify(lyap * period, lyap * result.t_end)

    return {
        "model": system.describe(),
        "integration": {
            "method": result.config.method.value,
            "symplectic": result.config.method.symplectic,
            "dt": result.dt,
            "t_end": result.t_end,
            "characteristic_time": period,
            "periods": result.t_end / period,
            "n_steps": result.n_steps,
            "n_force_evaluations": result.n_force_evaluations,
            "n_samples": int(t.shape[0]),
            "wall_time_s": result.wall_time,
            "steps_per_second": (
                result.n_steps / result.wall_time if result.wall_time > 0 else None
            ),
        },
        "invariants": invariants,
        "speeds": speed_stats,
        "distances": distance_stats,
        "events": event_stats,
        "chaos": {
            "max_lyapunov_exponent": lyap,
            "lyapunov_per_period": lyap * period if lyap is not None else None,
            "lyapunov_time": (1.0 / lyap) if lyap and lyap > 0 else None,
            "spectral_entropy": spectral_entropy(power),
            "spectrum_signal": sig_name,
            "closest_return_distance": min_return,
            "closest_return_time": t_return,
            "verdict": verdict,
        },
    }
