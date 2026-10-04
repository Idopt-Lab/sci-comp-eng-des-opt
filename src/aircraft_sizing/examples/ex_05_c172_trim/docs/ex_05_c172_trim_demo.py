"""Small, live demonstration of the C172 trim example (runs in a few seconds).

Verifies the documented M = 0.1 trim, then trims an N = 4 Mach sweep with every
*installed* backend (exact gradients) plus finite differences and the GA, printing a
side-by-side comparison, and saves a trim-schedule figure.  For the full scaling and
per-call benchmarks, run ``methods.study`` instead; this script also prints a summary
of those CSVs if they are present.

Run from the repository root:
    python src/aircraft_sizing/examples/ex_05_c172_trim/docs/ex_05_c172_trim_demo.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from aircraft_sizing.examples.ex_05_c172_trim.methods import backends as B
from aircraft_sizing.examples.ex_05_c172_trim.methods import solve as S
from aircraft_sizing.examples.ex_05_c172_trim.methods.aircraft import C172Data
from aircraft_sizing.examples.ex_05_c172_trim.methods.problem import TrimProblem
from aircraft_sizing.examples.ex_05_c172_trim.viz import plots

_DATA_DIR = Path(__file__).resolve().parent / "assets" / "data"
_IMAGE_DIR = Path(__file__).resolve().parent / "assets" / "images"


def verify_documented_trim() -> None:
    problem = TrimProblem(data=C172Data(), mach=np.array([0.1]))
    x = S.reference_solution(problem)
    theta, delta_e, omega = problem.unpack(x)
    print("Documented trim at M = 0.1 (target: 9.505408 deg, -9.581727 deg, 1823.38 RPM)")
    print(f"  theta   = {np.rad2deg(theta)[0]:.6f} deg")
    print(f"  delta_e = {np.rad2deg(delta_e)[0]:.6f} deg")
    print(f"  omega   = {omega[0]:.2f} RPM\n")


def compare_methods(n_nodes: int = 4) -> S.TrimSolution:
    problem = TrimProblem.with_sweep(n_nodes)
    reference = S.reference_solution(problem)
    stamp = S.platform_stamp()
    print(f"N = {n_nodes} trim sweep, Mach = {np.round(problem.mach, 3)}")
    print(f"{'method':18} {'device':8} {'iters':>5} {'nfev':>6} {'njev':>5} "
          f"{'f*':>9} {'err':>9} {'solve(ms)':>9}")

    exact_solution = None
    for name in B.available_backends(exact_grad_only=True):
        backend, setup = S.build_backend(name, problem)
        sol = S.solve_slsqp(problem, backend, deriv="exact", reference=reference,
                            setup_seconds=setup, stamp=stamp)
        exact_solution = exact_solution or sol
        _print_row(f"{name}:exact", sol)

    from aircraft_sizing.examples.ex_05_c172_trim.methods.backend_numpy import NumpyBackend

    for deriv in ("fd-central", "fd-forward"):
        sol = S.solve_slsqp(problem, NumpyBackend(problem), deriv=deriv, reference=reference)
        _print_row(f"numpy:{deriv}", sol)

    ga = S.solve_ga(problem, reference=reference, max_evals=60_000, stamp=stamp)
    _print_row("ga", ga)
    print()
    return exact_solution


def _print_row(label: str, sol: S.TrimSolution) -> None:
    print(f"{label:18} {sol.device:8} {sol.n_iter:>5} {sol.n_model_evals:>6} "
          f"{sol.n_grad_evals:>5} {sol.objective:>9.1e} {sol.error_vs_reference:>9.1e} "
          f"{sol.solve_seconds * 1e3:>9.1f}")


def summarize_study_csvs() -> None:
    if not (_DATA_DIR / "scaling.csv").exists():
        print("(No study CSVs yet — run methods.study to generate them.)")
        return
    import pandas as pd

    scaling = pd.read_csv(_DATA_DIR / "scaling.csv")
    stamp_cols = [c for c in scaling.columns if c.startswith("stamp_")]
    if stamp_cols:
        row = scaling.iloc[0]
        print("Study platform:", ", ".join(f"{c[6:]}={row[c]}" for c in stamp_cols))
    print(f"Scaling rows: {len(scaling)}, node counts: {sorted(scaling['n_nodes'].unique())}")


def main() -> None:
    verify_documented_trim()
    schedule = compare_methods()
    _IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    out = plots.plot_trim_schedule(schedule, _IMAGE_DIR / "trim_schedule.png")
    print(f"Trim-schedule figure: {out}\n")
    summarize_study_csvs()


if __name__ == "__main__":
    main()
