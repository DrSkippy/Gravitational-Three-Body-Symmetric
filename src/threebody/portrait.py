"""Poincare portraits of the Sitnikov map: many orbits on one section."""

from __future__ import annotations

import html
from dataclasses import dataclass, replace

import numpy as np
import plotly.graph_objects as go

from threebody.integrators import FloatArray
from threebody.metrics import classify
from threebody.simulate import SimulationConfig, simulate
from threebody.systems import SymmetricThreeBody
from threebody.viz import SERIES, page


@dataclass(frozen=True)
class PortraitOrbit:
    """Stroboscopic section points of one orbit."""

    v0: float
    z: FloatArray
    vz: FloatArray
    lyapunov_per_period: float
    verdict: str
    escaped: bool


def sitnikov_portrait(
    base: SymmetricThreeBody,
    n_orbits: int = 30,
    periods: float = 300.0,
    steps_per_period: int = 1000,
    max_speed_fraction: float = 0.98,
) -> list[PortraitOrbit]:
    """Sample the stroboscopic (binary phase 0) section for many launch speeds.

    Each orbit starts in the stellar plane with speed ``v0`` spread evenly
    over ``(0, max_speed_fraction * v_escape]``.
    """
    v_esc = base.escape_velocity
    config = SimulationConfig(
        periods=periods, steps_per_period=steps_per_period, max_samples=2
    )
    orbits = []
    for v0 in np.linspace(0.0, max_speed_fraction * v_esc, n_orbits + 1)[1:]:
        model = replace(base, v0=float(v0))
        result = simulate(model, config)
        strobe = result.events.select(1, 1.0)
        s = strobe.q[:, 3] - strobe.q[:, 2]
        vs = strobe.v[:, 3] - strobe.v[:, 2]
        lam_t = float(result.lyapunov[-1]) * model.characteristic_time()
        e_folds = float(result.lyapunov[-1]) * result.t_end
        escaped = bool(abs(result.q[-1, 3] - result.q[-1, 2]) > 50.0 * model.r0)
        orbits.append(
            PortraitOrbit(float(v0), s, vs, lam_t, classify(lam_t, e_folds), escaped)
        )
    return orbits


def portrait_html(
    base: SymmetricThreeBody,
    orbits: list[PortraitOrbit],
    plotly_src: str | None,
) -> str:
    """Render a portrait as an HTML page."""
    fig = go.Figure()
    v_esc = base.escape_velocity
    for orbit in orbits:
        chaotic = orbit.verdict.startswith("chaotic")
        fig.add_trace(
            go.Scatter(
                x=orbit.z,
                y=orbit.vz,
                mode="markers",
                marker={
                    "size": 3 if chaotic else 2,
                    "color": SERIES[1] if chaotic else SERIES[0],
                    "opacity": 0.6 if chaotic else 0.9,
                },
                name=f"v0 = {orbit.v0 / v_esc:.3f} v_esc",
                legendgroup="chaotic" if chaotic else "regular",
                hovertemplate=(
                    f"v0/v_esc = {orbit.v0 / v_esc:.3f}<br>"
                    f"λT = {orbit.lyapunov_per_period:.3f}<br>"
                    f"{orbit.verdict}{' / escaped' if orbit.escaped else ''}"
                    "<extra></extra>"
                ),
                showlegend=False,
            )
        )
    # Legend entries for the two classes (colour plus wording, never colour alone).
    for label, colour in (("regular orbit", SERIES[0]), ("chaotic orbit", SERIES[1])):
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker={"size": 8, "color": colour},
                name=label,
            )
        )
    # Frame the bound region: escaping orbits run off to large |z| and would
    # otherwise flatten the islands near the origin.
    bound = [o for o in orbits if not o.escaped and o.z.size] or orbits
    z_all = np.concatenate([o.z for o in bound]) if bound else np.zeros(1)
    v_all = np.concatenate([o.vz for o in bound]) if bound else np.zeros(1)
    z_lim = 1.1 * float(np.max(np.abs(z_all), initial=1e-9))
    v_lim = 1.1 * float(np.max(np.abs(v_all), initial=1e-9))
    fig.update_layout(
        title={"text": "Stroboscopic section at binary phase 0", "x": 0.0},
        xaxis={
            "title": {"text": "comet height z - y"},
            "zeroline": False,
            "range": [-z_lim, z_lim],
        },
        yaxis={
            "title": {"text": "comet velocity vz - vy"},
            "zeroline": False,
            "range": [-v_lim, v_lim],
        },
        height=720,
        margin={"l": 64, "r": 16, "t": 48, "b": 48},
        legend={"orientation": "h", "y": 1.02, "x": 1.0, "xanchor": "right"},
        font={"family": "system-ui, -apple-system, Segoe UI, sans-serif"},
    )
    n_chaotic = sum(o.verdict.startswith("chaotic") for o in orbits)
    rows = "".join(
        f"<tr><td>{o.v0 / v_esc:.3f}</td><td>{o.lyapunov_per_period:.3f}</td>"
        f"<td>{html.escape(o.verdict)}</td><td>{'yes' if o.escaped else 'no'}</td></tr>"
        for o in orbits
    )
    body = f"""
<h1>Sitnikov Poincaré portrait (e = {base.eccentricity}, mu = {base.mu:g})</h1>
<p class="lede">Each orbit starts in the stellar plane and is sampled once per
binary period, at pericentre. Closed invariant curves (KAM tori) are regular
orbits; scattered points are chaotic. {n_chaotic} of {len(orbits)} orbits are
chaotic by their Lyapunov exponent; hover a point for its launch speed. The axes
frame the bound orbits, so escaping orbits leave the frame.</p>
<div class="card">{fig.to_html(full_html=False, include_plotlyjs=False, div_id="portrait")}</div>
<h2>Orbits</h2>
<div class="card"><table><thead><tr><th>v0 / v_esc</th><th>λ_max per period</th>
<th>verdict</th><th>escaped</th></tr></thead><tbody>{rows}</tbody></table></div>
"""
    return page("Sitnikov portrait", body, plotly_src)
