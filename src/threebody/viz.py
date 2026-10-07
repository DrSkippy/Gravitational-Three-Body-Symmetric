"""Interactive Plotly dashboards: a 3D orbit animation plus diagnostics.

A dashboard is one self-contained HTML page.  ``plotly.min.js`` is either
inlined, loaded from a CDN, or (the default for the CLI) written once next
to the page so the dashboards work offline without bloating every file.
"""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs, get_plotlyjs_version

from threebody.integrators import FloatArray
from threebody.metrics import power_spectrum, relative_error_series
from threebody.simulate import SimulationResult

# Categorical slots of the validated reference palette, in fixed order.
SERIES = (
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#4a3aa7",
    "#e34948",
)
SEQUENTIAL = [[0.0, "#cde2fb"], [0.5, "#3987e5"], [1.0, "#0d366b"]]
MAX_POINTS = 4000
PLOTLY_CDN = (
    f"https://cdn.jsdelivr.net/npm/plotly.js-dist-min@{get_plotlyjs_version()}"
    "/plotly.min.js"
)


def _f32(a: FloatArray) -> npt.NDArray[np.float32]:
    """Single precision is plenty for plotting and halves the page size."""
    return np.asarray(a, dtype=np.float32)


def _stride(n: int, max_points: int = MAX_POINTS) -> slice:
    return slice(None, None, max(1, n // max_points))


def _base_layout(title: str, x_label: str, y_label: str, **kw: Any) -> dict[str, Any]:
    axis = {"showline": True, "zeroline": False, "ticks": "outside", "ticklen": 4}
    return {
        "title": {"text": title, "x": 0.0, "xanchor": "left", "font": {"size": 14}},
        "xaxis": {**axis, "title": {"text": x_label}},
        "yaxis": {**axis, "title": {"text": y_label}},
        "margin": {"l": 64, "r": 16, "t": 44, "b": 48},
        "height": 340,
        "hovermode": "closest",
        "legend": {"orientation": "h", "y": -0.25, "x": 0.0},
        "font": {"family": "system-ui, -apple-system, Segoe UI, sans-serif"},
        **kw,
    }


# --------------------------------------------------------------------------
# 3D orbit animation
# --------------------------------------------------------------------------


def orbit_figure(
    result: SimulationResult,
    animate_periods: float | None = 10.0,
    n_frames: int = 240,
    trail: int = 30,
) -> go.Figure:
    """Animated 3D view: full paths, moving bodies with trails, links.

    The animation covers the first ``animate_periods`` characteristic times
    (all of the run when ``None``); the static paths always show everything.
    """
    system = result.system
    pos = _f32(system.body_positions(result.q))
    n_samples, n_bodies, _ = pos.shape
    masses = system.body_masses()
    sizes = 6.0 + 10.0 * (masses / masses.max()) ** (1.0 / 3.0)
    period = system.characteristic_time()
    path_idx = np.arange(n_samples)[_stride(n_samples, 6000)]

    traces: list[go.Scatter3d] = []
    for b in range(n_bodies):
        p = pos[path_idx, b]
        if np.ptp(p, axis=0).max() == 0.0:
            continue  # fixed point (e.g. a pivot): no path
        traces.append(
            go.Scatter3d(
                x=p[:, 0],
                y=p[:, 1],
                z=p[:, 2],
                mode="lines",
                line={"color": SERIES[b % 8], "width": 2},
                opacity=0.35,
                name=f"{system.body_labels[b]} path",
                hoverinfo="skip",
            )
        )
    for name, curve in system.extra_curves(result.t, result.q, result.v).items():
        c = _f32(curve[path_idx])
        traces.append(
            go.Scatter3d(
                x=c[:, 0],
                y=c[:, 1],
                z=c[:, 2],
                mode="lines",
                line={
                    "color": _f32(result.t[path_idx] / period),
                    "colorscale": SEQUENTIAL,
                    "width": 3,
                },
                name=name,
                customdata=_f32(result.t[path_idx] / period),
                hovertemplate="t = %{customdata:.2f} periods<extra>"
                + name
                + "</extra>",
            )
        )
    n_static = len(traces)

    n_window = n_samples
    if animate_periods is not None:
        n_window = max(2, int(np.searchsorted(result.t, animate_periods * period)))
        n_window = min(n_window, n_samples)
    frame_idx = np.unique(np.linspace(0, n_window - 1, n_frames).astype(int))
    trail_len = max(2, int(trail * n_window / max(len(frame_idx), 1)))

    def animated(i: int) -> list[go.Scatter3d]:
        lo = max(0, i - trail_len)
        sl = slice(lo, i + 1, max(1, (i + 1 - lo) // 30))
        out: list[go.Scatter3d] = []
        for b in range(n_bodies):
            tr = pos[sl, b]
            out.append(
                go.Scatter3d(
                    x=tr[:, 0],
                    y=tr[:, 1],
                    z=tr[:, 2],
                    mode="lines",
                    line={"color": SERIES[b % 8], "width": 5},
                    showlegend=False,
                    hoverinfo="skip",
                )
            )
        for a, b in system.links:
            out.append(
                go.Scatter3d(
                    x=[pos[i, a, 0], pos[i, b, 0]],
                    y=[pos[i, a, 1], pos[i, b, 1]],
                    z=[pos[i, a, 2], pos[i, b, 2]],
                    mode="lines",
                    line={"color": "#898781", "width": 3},
                    showlegend=False,
                    hoverinfo="skip",
                )
            )
        out.append(
            go.Scatter3d(
                x=pos[i, :, 0],
                y=pos[i, :, 1],
                z=pos[i, :, 2],
                mode="markers",
                marker={
                    "size": sizes,
                    "color": [SERIES[b % 8] for b in range(n_bodies)],
                    "line": {"color": "#fcfcfb", "width": 2},
                },
                text=list(system.body_labels),
                hovertemplate="%{text}<br>(%{x:.3g}, %{y:.3g}, %{z:.3g})<extra></extra>",
                name="bodies",
                showlegend=False,
            )
        )
        return out

    first = animated(int(frame_idx[0]))
    animated_ids = list(range(n_static, n_static + len(first)))
    frames = [
        go.Frame(data=animated(int(i)), traces=animated_ids, name=str(k))
        for k, i in enumerate(frame_idx)
    ]

    lo = pos.reshape(-1, 3).min(axis=0)
    hi = pos.reshape(-1, 3).max(axis=0)
    centre = 0.5 * (lo + hi)
    half = 0.55 * float(np.max(hi - lo)) or 1.0
    axis = {"showbackground": False, "gridcolor": "#e1e0d9", "zeroline": False}
    ranges = [[float(c - half), float(c + half)] for c in centre]

    fig = go.Figure(data=traces + first, frames=frames)
    fig.update_layout(
        title={
            "text": "Orbits (drag to rotate, scroll to zoom)",
            "x": 0.0,
            "y": 0.98,
            "yanchor": "top",
        },
        height=660,
        margin={"l": 0, "r": 0, "t": 72, "b": 0},
        font={"family": "system-ui, -apple-system, Segoe UI, sans-serif"},
        legend={"orientation": "h", "y": 1.0, "x": 0.0, "yanchor": "top"},
        scene={
            "xaxis": {**axis, "range": ranges[0], "title": {"text": "x"}},
            "yaxis": {**axis, "range": ranges[1], "title": {"text": "y"}},
            "zaxis": {**axis, "range": ranges[2], "title": {"text": "z"}},
            "aspectmode": "cube",
        },
        updatemenus=[
            {
                "type": "buttons",
                "direction": "left",
                "x": 0.0,
                "y": 0.0,
                "xanchor": "left",
                "yanchor": "top",
                "pad": {"t": 8},
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "frame": {"duration": 40, "redraw": True},
                                "transition": {"duration": 0},
                                "fromcurrent": True,
                            },
                        ],
                    },
                    {
                        "label": "Pause",
                        "method": "animate",
                        "args": [
                            [None],
                            {"frame": {"duration": 0}, "mode": "immediate"},
                        ],
                    },
                ],
            }
        ],
        sliders=[
            {
                "x": 0.15,
                "len": 0.85,
                "y": 0.0,
                "yanchor": "top",
                "pad": {"t": 4},
                "currentvalue": {"prefix": "t = ", "suffix": " periods"},
                "steps": [
                    {
                        "label": f"{result.t[i] / period:.2f}",
                        "method": "animate",
                        "args": [
                            [str(k)],
                            {
                                "frame": {"duration": 0, "redraw": True},
                                "mode": "immediate",
                            },
                        ],
                    }
                    for k, i in enumerate(frame_idx)
                ],
            }
        ],
    )
    return fig


# --------------------------------------------------------------------------
# Diagnostic charts
# --------------------------------------------------------------------------


def _time_series(
    title: str,
    t: FloatArray,
    series: dict[str, FloatArray],
    y_label: str,
    log_y: bool = False,
    x_label: str = "time [periods]",
) -> go.Figure:
    sl = _stride(t.shape[0])
    fig = go.Figure()
    for k, (name, y) in enumerate(series.items()):
        yy = _f32(y[sl])
        if log_y:
            yy = np.where(yy > 0, yy, np.nan)
        fig.add_trace(
            go.Scatter(
                x=_f32(t[sl]),
                y=yy,
                mode="lines",
                name=name,
                line={"color": SERIES[k % 8], "width": 2},
                hovertemplate=f"{name}<br>t=%{{x:.3f}}<br>%{{y:.4g}}<extra></extra>",
            )
        )
    layout = _base_layout(title, x_label, y_label, showlegend=len(series) > 1)
    if log_y:
        layout["yaxis"]["type"] = "log"
        layout["yaxis"]["exponentformat"] = "power"
    fig.update_layout(**layout)
    return fig


def _scatter(
    title: str,
    x_label: str,
    y_label: str,
    x: FloatArray,
    y: FloatArray,
    color: FloatArray,
    color_label: str,
    lines: bool,
    periodic_x: float | None = None,
) -> go.Figure:
    sl = _stride(x.shape[0])
    xs, ys = _f32(x[sl]), _f32(y[sl])
    if lines and periodic_x is not None and xs.size > 1:
        # Break the line where an angle wraps instead of drawing across.
        jumps = np.flatnonzero(np.abs(np.diff(xs)) > 0.5 * periodic_x) + 1
        xs = np.insert(xs, jumps, np.nan)
        ys = np.insert(ys, jumps, np.nan)
    fig = go.Figure(
        go.Scatter(
            x=xs,
            y=ys,
            mode="lines" if lines else "markers",
            line={"color": SERIES[0], "width": 1},
            marker=(
                {}
                if lines
                else {
                    "size": 4,
                    # A list, not a typed array: plotly reads a length-3/4
                    # typed array as a single RGB(A) colour.
                    "color": color[sl].tolist(),
                    "colorscale": SEQUENTIAL,
                    "colorbar": {"title": {"text": color_label}, "thickness": 10},
                }
            ),
            hovertemplate=(
                f"{x_label}=%{{x:.4g}}<br>{y_label}=%{{y:.4g}}<extra></extra>"
            ),
            showlegend=False,
        )
    )
    fig.update_layout(**_base_layout(title, x_label, y_label))
    return fig


def diagnostic_figures(result: SimulationResult) -> list[go.Figure]:
    """All 2D diagnostic charts for a run."""
    system = result.system
    period = system.characteristic_time()
    t, q, v = result.t, result.q, result.v
    tp = t / period
    figs: list[go.Figure] = []

    inv = system.invariants(q, v)
    figs.append(
        _time_series(
            "Conservation: relative drift of invariants",
            tp,
            {k: relative_error_series(s) for k, s in inv.items()},
            "|ΔI / I₀|",
            log_y=True,
        )
    )
    speeds = np.linalg.norm(system.body_velocities(q, v), axis=2)
    figs.append(
        _time_series(
            "Speeds",
            tp,
            {
                label: speeds[:, i]
                for i, label in enumerate(system.body_labels)
                if np.any(speeds[:, i] > 0)
            },
            "speed",
        )
    )
    figs.append(_time_series("Separations", tp, system.distances(q), "distance"))
    if result.lyapunov.size:
        figs.append(
            _time_series(
                "Finite-time maximal Lyapunov exponent",
                result.lyapunov_t / period,
                {"λ_max × T": result.lyapunov * period},
                "λ_max per period",
            )
        )
    for pp in system.phase_portraits(t, q, v):
        figs.append(
            _scatter(
                pp.title,
                pp.x_label,
                pp.y_label,
                pp.x,
                pp.y,
                tp,
                "periods",
                True,
                pp.periodic_x,
            )
        )
    for sec in system.sections(result.events):
        figs.append(
            _scatter(
                sec.title,
                sec.x_label,
                sec.y_label,
                sec.x,
                sec.y,
                sec.t / period,
                "periods",
                False,
            )
        )
    name, signal = system.spectrum_signal(q, v)
    freq, power = power_spectrum(t, signal)
    if freq.size:
        fig = _time_series(
            f"Power spectrum of {name}",
            freq * period,
            {name: power},
            "relative power",
            log_y=True,
            x_label="frequency [cycles per period]",
        )
        fig.update_xaxes(range=[0, min(float(freq[-1] * period), 20.0)])
        figs.append(fig)
    return figs


# --------------------------------------------------------------------------
# HTML page
# --------------------------------------------------------------------------

_CSS = """
:root {
  color-scheme: light;
  --page: #f9f9f7; --surface: #fcfcfb; --text: #0b0b0b; --text-2: #52514e;
  --muted: #898781; --grid: #e1e0d9; --axis: #c3c2b7;
  --ring: rgba(11,11,11,0.10); --good: #006300; --warn: #b45309;
}
@media (prefers-color-scheme: dark) {
  :root:where(:not([data-theme="light"])) {
    color-scheme: dark;
    --page: #0d0d0d; --surface: #1a1a19; --text: #ffffff; --text-2: #c3c2b7;
    --muted: #898781; --grid: #2c2c2a; --axis: #383835;
    --ring: rgba(255,255,255,0.10); --good: #0ca30c; --warn: #fab219;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --page: #0d0d0d; --surface: #1a1a19; --text: #ffffff; --text-2: #c3c2b7;
  --muted: #898781; --grid: #2c2c2a; --axis: #383835;
  --ring: rgba(255,255,255,0.10); --good: #0ca30c; --warn: #fab219;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--page); color: var(--text);
  font-family: system-ui, -apple-system, "Segoe UI", sans-serif; line-height: 1.45; }
main { max-width: 1280px; margin: 0 auto; padding: 24px 16px 48px; }
h1 { font-size: 1.6rem; margin: 0 0 4px; }
h2 { font-size: 1.1rem; margin: 32px 0 12px; }
p.lede { color: var(--text-2); margin: 0 0 16px; max-width: 80ch; }
.card { background: var(--surface); border-radius: 12px;
  box-shadow: 0 0 0 1px var(--ring); padding: 8px; min-width: 0; }
.tiles { display: grid; gap: 12px;
  grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); margin: 16px 0; }
.tile { background: var(--surface); border-radius: 12px; padding: 12px 14px;
  box-shadow: 0 0 0 1px var(--ring); }
.tile .label { color: var(--text-2); font-size: 0.8rem; }
.tile .value { font-size: 1.35rem; font-weight: 600; margin-top: 2px; }
.tile .note { color: var(--muted); font-size: 0.75rem; }
.grid { display: grid; gap: 12px;
  grid-template-columns: repeat(auto-fit, minmax(min(100%, 420px), 1fr)); }
.badge { display: inline-block; padding: 2px 10px; border-radius: 999px;
  font-size: 0.85rem; box-shadow: 0 0 0 1px var(--ring); }
.badge.chaotic::before { content: "\\2731  "; color: var(--warn); }
.badge.regular::before { content: "\\25CB  "; color: var(--good); }
table { border-collapse: collapse; width: 100%; font-size: 0.9rem;
  font-variant-numeric: tabular-nums; }
th, td { text-align: left; padding: 6px 10px; border-bottom: 1px solid var(--grid); }
th { color: var(--text-2); font-weight: 500; }
.tables { display: grid; gap: 12px;
  grid-template-columns: repeat(auto-fit, minmax(min(100%, 380px), 1fr)); }
.files a { color: inherit; margin-right: 16px; }
.toggle { float: right; background: var(--surface); color: var(--text);
  border: 0; box-shadow: 0 0 0 1px var(--ring); border-radius: 8px;
  padding: 4px 10px; cursor: pointer; }
"""

_THEME_JS = """
(function () {
  const root = document.documentElement;
  function css(name) { return getComputedStyle(root).getPropertyValue(name).trim(); }
  function restyle() {
    const surface = css('--surface'), text = css('--text-2'), grid = css('--grid'),
          axis = css('--axis');
    const ax = {gridcolor: grid, linecolor: axis, zerolinecolor: grid,
                tickcolor: axis, color: text};
    document.querySelectorAll('.js-plotly-plot').forEach(function (div) {
      const upd = {paper_bgcolor: surface, plot_bgcolor: surface, 'font.color': text};
      for (const k of ['xaxis', 'yaxis']) {
        for (const [p, val] of Object.entries(ax)) upd[k + '.' + p] = val;
      }
      for (const k of ['scene.xaxis', 'scene.yaxis', 'scene.zaxis']) {
        for (const [p, val] of Object.entries(ax)) upd[k + '.' + p] = val;
      }
      Plotly.relayout(div, upd);
    });
  }
  window.addEventListener('load', restyle);
  window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', restyle);
  window.toggleTheme = function () {
    const dark = getComputedStyle(root).getPropertyValue('color-scheme').trim() === 'dark';
    root.setAttribute('data-theme', dark ? 'light' : 'dark');
    restyle();
  };
})();
"""


def _fmt(x: Any) -> str:
    if x is None:
        return "—"
    if isinstance(x, bool):
        return "yes" if x else "no"
    if isinstance(x, float):
        if x == 0.0:
            return "0"
        return f"{x:.4g}" if 1e-3 <= abs(x) < 1e5 else f"{x:.3e}"
    return html.escape(str(x))


def _table(rows: list[tuple[str, Any]], head: tuple[str, str]) -> str:
    body = "".join(
        f"<tr><td>{html.escape(k)}</td><td>{_fmt(v)}</td></tr>" for k, v in rows
    )
    return (
        f"<table><thead><tr><th>{head[0]}</th><th>{head[1]}</th></tr></thead>"
        f"<tbody>{body}</tbody></table>"
    )


def _tile(label: str, value: str, note: str = "") -> str:
    return (
        f'<div class="tile"><div class="label">{html.escape(label)}</div>'
        f'<div class="value">{value}</div><div class="note">{html.escape(note)}</div></div>'
    )


def _fig_div(fig: go.Figure, div_id: str) -> str:
    return str(
        fig.to_html(
            full_html=False,
            include_plotlyjs=False,
            div_id=div_id,
            config={"displaylogo": False, "responsive": True},
        )
    )


def plotly_script_tag(plotly_src: str | None) -> str:
    """``<script>`` tag loading plotly.js from ``plotly_src``.

    ``None`` inlines the library; an empty string omits it.
    """
    if plotly_src == "":
        return ""
    if plotly_src is None:
        return f"<script>{get_plotlyjs()}</script>"
    return f'<script src="{html.escape(plotly_src)}"></script>'


def write_plotly_js(directory: Path) -> Path:
    """Write ``plotly.min.js`` into ``directory`` (once) and return its path."""
    path = directory / "plotly.min.js"
    if not path.exists():
        directory.mkdir(parents=True, exist_ok=True)
        path.write_text(get_plotlyjs(), encoding="utf-8")
    return path


def page(title: str, body: str, plotly_src: str | None) -> str:
    """Wrap ``body`` in the dashboard page shell."""
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>{_CSS}</style>
{plotly_script_tag(plotly_src)}
</head><body><main>
<button class="toggle" onclick="toggleTheme()">Light / dark</button>
{body}
</main><script>{_THEME_JS}</script></body></html>
"""


def dashboard_html(
    result: SimulationResult,
    summary: dict[str, Any],
    title: str,
    description: str = "",
    plotly_src: str | None = PLOTLY_CDN,
    data_files: tuple[str, ...] = (),
    animate_periods: float | None = 10.0,
) -> str:
    """Render the full dashboard for one run as an HTML string."""
    integ = summary["integration"]
    chaos = summary["chaos"]
    energy = summary["invariants"][0]
    verdict = chaos["verdict"]
    badge_cls = "chaotic" if verdict.startswith("chaotic") else "regular"

    tiles = [
        _tile(
            "Max energy drift",
            _fmt(energy["max_rel_error"]),
            "relative, " + ("symplectic" if integ["symplectic"] else "adaptive"),
        ),
        _tile(
            "λ_max per period",
            _fmt(chaos["lyapunov_per_period"]),
            "finite-time estimate",
        ),
        _tile(
            "Spectral entropy",
            _fmt(chaos["spectral_entropy"]),
            "0 = line spectrum, 1 = white noise",
        ),
        _tile(
            "Closest return",
            _fmt(chaos["closest_return_distance"]),
            f"at t = {_fmt(chaos['closest_return_time'] / integ['characteristic_time'])}"
            " periods",
        ),
        _tile(
            "Run length",
            _fmt(integ["periods"]),
            f"periods of {_fmt(integ['characteristic_time'])}",
        ),
        _tile(
            "Steps / s",
            _fmt(integ["steps_per_second"]),
            f"{integ['n_steps']:,} steps in {integ['wall_time_s']:.2f} s",
        ),
    ]

    params = [(k, v) for k, v in summary["model"].items() if not isinstance(v, list)]
    params += [
        (k, json.dumps(v)) for k, v in summary["model"].items() if isinstance(v, list)
    ]
    integ_rows = [(k, v) for k, v in integ.items()]
    inv_rows = [
        (f"{i['name']} (max rel. drift)", i["max_rel_error"])
        for i in summary["invariants"]
    ]
    speed_rows = [
        (
            f"{b} speed min / mean / max",
            f"{_fmt(s['min'])} / {_fmt(s['mean'])} / {_fmt(s['max'])}",
        )
        for b, s in summary["speeds"].items()
    ]
    dist_rows = [
        (
            f"{k} min / mean / max",
            f"{_fmt(s['min'])} / {_fmt(s['mean'])} / {_fmt(s['max'])}",
        )
        for k, s in summary["distances"].items()
    ]
    event_rows = [
        (f"{k}: count, mean interval", f"{s['count']}, {_fmt(s['mean_interval'])}")
        for k, s in summary["events"].items()
    ]

    orbit = _fig_div(orbit_figure(result, animate_periods), "orbit")
    diag = "".join(
        f'<div class="card">{_fig_div(f, f"diag{k}")}</div>'
        for k, f in enumerate(diagnostic_figures(result))
    )
    files = "".join(
        f'<a href="{html.escape(f)}">{html.escape(f)}</a>' for f in data_files
    )

    body = f"""
<h1>{html.escape(title)}</h1>
<p class="lede">{html.escape(description)}</p>
<span class="badge {badge_cls}">{html.escape(verdict)}</span>
<div class="tiles">{''.join(tiles)}</div>
<div class="card">{orbit}</div>
<h2>Diagnostics</h2>
<div class="grid">{diag}</div>
<h2>Numbers</h2>
<div class="tables">
<div class="card">{_table(params, ("Parameter", "Value"))}</div>
<div class="card">{_table(integ_rows, ("Integration", "Value"))}</div>
<div class="card">{_table(inv_rows + speed_rows + dist_rows + event_rows, ("Metric", "Value"))}</div>
</div>
{f'<h2>Data files</h2><p class="files">{files}</p>' if files else ''}
"""
    return page(title, body, plotly_src)
