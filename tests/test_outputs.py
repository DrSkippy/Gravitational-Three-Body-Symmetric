"""CLI, data files and dashboards."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from threebody.cli import main


def _header(path: Path) -> list[str]:
    with path.open() as f:
        return next(csv.reader(f))


def test_symmetric_run_writes_data_and_dashboard(tmp_path: Path) -> None:
    out = tmp_path / "sym"
    assert main(["symmetric", "--periods", "3", "--out", str(out)]) == 0
    assert _header(out / "trajectory.csv") == [
        "t",
        "r",
        "vr",
        "z",
        "vz",
        "y",
        "vy",
        "theta",
        "w",
    ]
    events = _header(out / "events.csv")
    assert events[:2] == ["event", "direction"] and events[-1] == "interval"
    summary = json.loads((out / "summary.json").read_text())
    assert summary["integration"]["method"] == "yoshida4"
    assert summary["invariants"][0]["max_rel_error"] < 1e-10
    html = (out / "dashboard.html").read_text()
    assert '<script src="plotly.min.js">' in html
    assert (out / "plotly.min.js").exists()
    assert "Stroboscopic" in html or "Return map" in html


@pytest.mark.parametrize(
    "argv",
    [
        ["sho", "--periods", "2"],
        ["pendulum", "--periods", "2"],
        ["nbody", "figure8", "--periods", "0.5"],
        ["nbody", "pythagorean", "--periods", "2", "--method", "dop853"],
        ["symmetric", "--ratio", "1.0", "--periods", "2", "--plotly", "cdn"],
    ],
)
def test_cli_commands(tmp_path: Path, argv: list[str]) -> None:
    out = tmp_path / "run"
    assert main([*argv, "--out", str(out)]) == 0
    assert (out / "dashboard.html").stat().st_size > 10_000
    assert (out / "trajectory.csv").exists()


def test_demo_gallery(tmp_path: Path) -> None:
    assert main(["demo", "sho", "ekeland-weave", "--out", str(tmp_path)]) == 0
    index = (tmp_path / "index.html").read_text()
    assert "sho/dashboard.html" in index and "ekeland-weave/dashboard.html" in index
    page = (tmp_path / "sho" / "dashboard.html").read_text()
    assert '<script src="../plotly.min.js">' in page


def test_unknown_demo(tmp_path: Path) -> None:
    assert main(["demo", "nope", "--out", str(tmp_path)]) == 2


def test_portrait(tmp_path: Path) -> None:
    from threebody.portrait import sitnikov_portrait
    from threebody.systems import SymmetricThreeBody

    orbits = sitnikov_portrait(
        SymmetricThreeBody(eccentricity=0.2), n_orbits=3, periods=20
    )
    assert len(orbits) == 3
    assert all(o.z.size > 5 for o in orbits[:2])
    assert (
        main(["portrait", "--orbits", "2", "--periods", "10", "--out", str(tmp_path)])
        == 0
    )
    assert (tmp_path / "portrait.html").exists()
