"""Command-line interface: ``threebody <command> [options]``."""

from __future__ import annotations

import argparse
import html
import os
import sys
import webbrowser
from collections.abc import Sequence
from pathlib import Path

from threebody.demos import DEMOS, PERIODIC_ORBITS, broken_symmetry, periodic_three_body
from threebody.demos import pythagorean as pythagorean_model
from threebody.integrators import Method
from threebody.io import write_run
from threebody.metrics import summarize
from threebody.portrait import portrait_html, sitnikov_portrait
from threebody.simulate import SimulationConfig, SimulationResult, simulate
from threebody.systems import (
    DynamicalSystem,
    HarmonicOscillator,
    SpringPendulum,
    SymmetricThreeBody,
)
from threebody.verification import report
from threebody.viz import PLOTLY_CDN, dashboard_html, page, write_plotly_js

DATA_FILES = ("trajectory.csv", "events.csv", "lyapunov.csv", "summary.json")


def _plotly_src(mode: str, out_dir: Path, root: Path) -> str | None:
    """Resolve how a page in ``out_dir`` should load plotly.js."""
    if mode == "cdn":
        return PLOTLY_CDN
    if mode == "inline":
        return None
    js = write_plotly_js(root)
    return Path(os.path.relpath(js, out_dir)).as_posix()


def run_and_report(
    system: DynamicalSystem,
    config: SimulationConfig,
    out_dir: Path,
    title: str,
    description: str = "",
    plotly: str = "local",
    plotly_root: Path | None = None,
    animate_periods: float | None = 10.0,
) -> tuple[SimulationResult, Path]:
    """Simulate, write data files and the dashboard; return the page path."""
    result = simulate(system, config)
    summary = summarize(result)
    write_run(result, summary, out_dir)
    root = plotly_root or out_dir
    files = tuple(f for f in DATA_FILES if (out_dir / f).exists())
    page_html = dashboard_html(
        result,
        summary,
        title,
        description,
        plotly_src=_plotly_src(plotly, out_dir, root),
        data_files=files,
        animate_periods=animate_periods,
    )
    path = out_dir / "dashboard.html"
    path.write_text(page_html, encoding="utf-8")
    integ, chaos = summary["integration"], summary["chaos"]
    print(
        f"{title}: {integ['n_steps']:,} steps in {integ['wall_time_s']:.2f} s, "
        f"max energy drift {summary['invariants'][0]['max_rel_error']:.2e}, "
        f"lambda*T = {chaos['lyapunov_per_period'] or 0:.3f} -> {chaos['verdict']}"
    )
    print(f"  wrote {out_dir}/{{{','.join(files)},dashboard.html}}")
    return result, path


def _add_common(p: argparse.ArgumentParser, periods: float, steps: int) -> None:
    p.add_argument("--periods", type=float, default=periods, help="run length")
    p.add_argument(
        "--steps-per-period", type=int, default=steps, help="fixed-step resolution"
    )
    p.add_argument(
        "--method",
        type=Method,
        choices=list(Method),
        default=Method.YOSHIDA4,
        help="integrator",
    )
    p.add_argument("--samples", type=int, default=20_000, help="max stored samples")
    p.add_argument(
        "--no-lyapunov", action="store_true", help="skip the Lyapunov estimate"
    )
    p.add_argument("--out", type=Path, default=None, help="output directory")
    p.add_argument(
        "--plotly",
        choices=("local", "cdn", "inline"),
        default="local",
        help="how dashboards load plotly.js (local: plotly.min.js beside the page)",
    )
    p.add_argument(
        "--animate-periods",
        type=float,
        default=10.0,
        help="periods covered by the 3D animation (0 = whole run)",
    )
    p.add_argument("--show", action="store_true", help="open the dashboard")


def _config(args: argparse.Namespace) -> SimulationConfig:
    return SimulationConfig(
        periods=args.periods,
        steps_per_period=args.steps_per_period,
        method=args.method,
        max_samples=args.samples,
        lyapunov=not args.no_lyapunov,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="threebody",
        description="Symplectic simulations of the symmetric three-body problem "
        "and friends, with interactive 3D dashboards.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("symmetric", help="Ekeland / Sitnikov symmetric problem")
    p.add_argument("--v0", type=float, default=0.17, help="comet launch speed")
    p.add_argument(
        "--ratio",
        type=float,
        default=None,
        help="instead of --v0: comet/binary period ratio to solve for",
    )
    p.add_argument("--mu", type=float, default=1e-8, help="comet mass / star mass")
    p.add_argument("--ecc", type=float, default=0.0, help="binary eccentricity")
    p.add_argument("--z0", type=float, default=0.0, help="initial comet height")
    p.add_argument("--r0", type=float, default=0.5, help="star distance from axis")
    p.add_argument("--G", type=float, default=4.302e-3, help="gravitational constant")
    p.add_argument("--M", type=float, default=1.0, help="star mass")
    _add_common(p, periods=40, steps=1500)

    p = sub.add_parser("nbody", help="general 3D N-body presets")
    p.add_argument(
        "preset",
        choices=sorted([*PERIODIC_ORBITS, "pythagorean", "broken-symmetry"]),
    )
    _add_common(p, periods=2, steps=4000)

    p = sub.add_parser("pendulum", help="3D elastic (spring) pendulum")
    p.add_argument("--ratio", type=float, default=2.0, help="spring/swing frequency")
    p.add_argument("--theta0", type=float, default=0.03, help="initial angle [rad]")
    p.add_argument("--omega0", type=float, default=0.0, help="initial rate [rad/s]")
    p.add_argument("--stretch0", type=float, default=0.1, help="initial stretch")
    p.add_argument("--v-perp0", type=float, default=0.02, help="out-of-plane speed")
    _add_common(p, periods=80, steps=1000)

    p = sub.add_parser("sho", help="simple harmonic oscillator run")
    p.add_argument("--omega", type=float, default=1.0)
    _add_common(p, periods=50, steps=64)

    p = sub.add_parser("demo", help="run curated demos and build a gallery")
    p.add_argument(
        "names", nargs="*", help=f"demo names (default all): {', '.join(DEMOS)}"
    )
    p.add_argument("--out", type=Path, default=Path("runs/demos"))
    p.add_argument("--plotly", choices=("local", "cdn", "inline"), default="local")
    p.add_argument("--show", action="store_true")

    p = sub.add_parser("portrait", help="Sitnikov Poincare portrait (many orbits)")
    p.add_argument("--ecc", type=float, default=0.2)
    p.add_argument("--mu", type=float, default=1e-8)
    p.add_argument("--orbits", type=int, default=30)
    p.add_argument("--periods", type=float, default=300)
    p.add_argument("--out", type=Path, default=Path("runs/portrait"))
    p.add_argument("--plotly", choices=("local", "cdn", "inline"), default="local")
    p.add_argument("--show", action="store_true")

    sub.add_parser("verify", help="integrator verification on the SHO")
    sub.add_parser("list", help="list demos")
    return parser


def _gallery(out: Path, entries: list[tuple[str, str, str]]) -> Path:
    cards = "".join(
        f'<div class="card" style="padding:16px"><h2 style="margin-top:0">'
        f'<a href="{name}/dashboard.html" style="color:inherit">{html.escape(title)}</a>'
        f'</h2><p class="lede">{html.escape(desc)}</p></div>'
        for name, title, desc in entries
    )
    body = (
        '<h1>Three-body gallery</h1><p class="lede">Periodic, quasi-periodic and '
        "chaotic solutions. Each page has a 3D animation, conservation checks, "
        f'phase portraits, sections and Lyapunov estimates.</p><div class="grid">{cards}</div>'
    )
    path = out / "index.html"
    path.write_text(page("Three-body gallery", body, plotly_src=""), encoding="utf-8")
    return path


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    shown: Path | None = None

    match args.command:
        case "verify":
            print(report())
            return 0
        case "list":
            width = max(map(len, DEMOS))
            for name, demo in DEMOS.items():
                print(f"{name:<{width}}  {demo.title}")
            return 0
        case "symmetric":
            common = {"G": args.G, "M": args.M, "r0": args.r0, "mu": args.mu}
            system: DynamicalSystem
            if args.ratio is not None:
                system = SymmetricThreeBody.resonant(args.ratio, **common)
            else:
                system = SymmetricThreeBody(
                    **common, v0=args.v0, z0=args.z0, eccentricity=args.ecc
                )
            out = args.out or Path("runs/symmetric")
            _, shown = run_and_report(
                system,
                _config(args),
                out,
                "Symmetric three-body problem",
                plotly=args.plotly,
                animate_periods=args.animate_periods or None,
            )
        case "nbody":
            if args.preset == "pythagorean":
                system = pythagorean_model()
            elif args.preset == "broken-symmetry":
                system = broken_symmetry()
            else:
                system = periodic_three_body(args.preset)
            out = args.out or Path(f"runs/{args.preset}")
            _, shown = run_and_report(
                system,
                _config(args),
                out,
                f"N-body: {args.preset}",
                plotly=args.plotly,
                animate_periods=args.animate_periods or None,
            )
        case "pendulum":
            system = SpringPendulum(
                ratio=args.ratio,
                theta0=args.theta0,
                omega0=args.omega0,
                stretch0=args.stretch0,
                v_perp0=args.v_perp0,
            )
            out = args.out or Path("runs/pendulum")
            _, shown = run_and_report(
                system,
                _config(args),
                out,
                "Elastic pendulum",
                plotly=args.plotly,
                animate_periods=args.animate_periods or None,
            )
        case "sho":
            out = args.out or Path("runs/sho")
            _, shown = run_and_report(
                HarmonicOscillator(omega=args.omega),
                _config(args),
                out,
                "Simple harmonic oscillator",
                plotly=args.plotly,
                animate_periods=args.animate_periods or None,
            )
        case "demo":
            names = args.names or list(DEMOS)
            unknown = [n for n in names if n not in DEMOS]
            if unknown:
                print(f"unknown demo(s): {', '.join(unknown)}", file=sys.stderr)
                return 2
            entries = []
            for name in names:
                demo = DEMOS[name]
                run_and_report(
                    demo.build(),
                    demo.config,
                    args.out / name,
                    demo.title,
                    demo.description,
                    plotly=args.plotly,
                    plotly_root=args.out,
                    animate_periods=demo.animate_periods,
                )
                entries.append((name, demo.title, demo.description))
            shown = _gallery(args.out, entries)
            print(f"gallery: {shown}")
        case "portrait":
            base = SymmetricThreeBody(eccentricity=args.ecc, mu=args.mu)
            orbits = sitnikov_portrait(base, args.orbits, args.periods)
            args.out.mkdir(parents=True, exist_ok=True)
            shown = args.out / "portrait.html"
            shown.write_text(
                portrait_html(
                    base, orbits, _plotly_src(args.plotly, args.out, args.out)
                ),
                encoding="utf-8",
            )
            print(f"portrait: {shown}")
        case _:  # pragma: no cover - argparse enforces the choices
            return 2

    if shown is not None and getattr(args, "show", False):
        webbrowser.open(shown.resolve().as_uri())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
