"""Writing run outputs: CSV data files and a JSON summary."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from threebody.integrators import FloatArray
from threebody.simulate import SimulationResult


def write_csv(path: Path, columns: dict[str, FloatArray]) -> None:
    """Write equally long columns to ``path`` with a header row."""
    names = list(columns)
    data = np.column_stack([np.asarray(columns[n], dtype=np.float64) for n in names])
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(names)
        writer.writerows(data.tolist() if data.size else [])


def event_columns(result: SimulationResult) -> dict[str, FloatArray]:
    """Event rows with the model's own column names, plus intervals."""
    ev = result.events
    system = result.system
    if len(ev):
        cols = system.columns(ev.t, ev.q, ev.v)
    else:
        q0, v0 = system.initial_state()
        cols = system.columns(
            np.zeros(0), np.zeros((0, q0.size)), np.zeros((0, v0.size))
        )
    # Interval since the previous event of the same id and direction (the
    # "period" column of the original returnmap.csv).
    interval = np.full(len(ev), np.nan)
    for e in np.unique(ev.event_id):
        for d in (-1.0, 1.0):
            idx = np.flatnonzero((ev.event_id == e) & (ev.direction == d))
            interval[idx[1:]] = np.diff(ev.t[idx])
    return {
        "event": ev.event_id.astype(np.float64),
        "direction": ev.direction,
        **cols,
        "interval": interval,
    }


def write_run(result: SimulationResult, summary: dict[str, Any], out_dir: Path) -> None:
    """Write ``trajectory.csv``, ``events.csv`` and ``summary.json``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    system = result.system
    write_csv(out_dir / "trajectory.csv", system.columns(result.t, result.q, result.v))
    write_csv(out_dir / "events.csv", event_columns(result))
    if result.lyapunov.size:
        write_csv(
            out_dir / "lyapunov.csv",
            {"t": result.lyapunov_t, "lambda_max": result.lyapunov},
        )
    meta = {
        **summary,
        "event_names": dict(enumerate(system.event_names)),
    }
    (out_dir / "summary.json").write_text(json.dumps(meta, indent=2, default=str))
