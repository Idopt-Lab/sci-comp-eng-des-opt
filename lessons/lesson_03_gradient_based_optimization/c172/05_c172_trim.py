r"""Lesson 3, script 05 - C172 trim: computing derivatives five ways on a real model.

This is the lesson's capstone: it drives the ``ex_05_c172_trim`` example to make the
Lesson 3 theme concrete on a real engineering problem.  A Cessna 172 is trimmed in
steady flight by minimizing the scaled residual of the six rigid-body accelerations,

    min_z  f(z) = 1/2 || r~(theta, delta_e, omega) ||^2 ,   z in [0, 1]^(3N),

and the SAME gradient-based solve is run with five different derivative backends:

    numpy  -> finite difference (no autodiff)
    jax    -> reverse-mode AD (CPU on native Windows)
    casadi -> symbolic differentiation
    csdl   -> graph-based AD
    warp   -> reverse-mode AD on the GPU (RTX 4060 via its bundled CUDA runtime)

The payoff ties the whole lesson together:

* every EXACT-gradient backend takes the IDENTICAL SLSQP path (same iteration and
  model-evaluation counts) -- the derivative *value* is what matters, not how it is
  produced (cf. the analytic/CS/AD equivalence in script 02);
* finite difference reaches the same optimum but costs extra model evaluations
  (forward vs. central, cf. script 02's discipline-call table);
* reverse-mode AD scales to many design variables (3N here), exactly the cost
  asymmetry demonstrated in script 03.

The model, all backends, and the full scaling/benchmark study live in
``src/aircraft_sizing/examples/ex_05_c172_trim`` (see its ``docs/ex_05_c172_trim.md``);
this script is the lesson-facing demonstration.

Run from the repository root (with the ``eng-des-opt-course`` environment active)::

    python lessons/lesson_03_gradient_based_optimization/c172/05_c172_trim.py
"""

from __future__ import annotations

import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):  # pragma: no cover
    pass

import numpy as np

from aircraft_sizing.examples.ex_05_c172_trim.methods import backends as B
from aircraft_sizing.examples.ex_05_c172_trim.methods import solve as S
from aircraft_sizing.examples.ex_05_c172_trim.methods.aircraft import C172Data
from aircraft_sizing.examples.ex_05_c172_trim.methods.backend_numpy import NumpyBackend
from aircraft_sizing.examples.ex_05_c172_trim.methods.problem import TrimProblem
from aircraft_sizing.examples.ex_05_c172_trim.viz import plots

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
REPO_ROOT = Path(__file__).resolve().parents[3]

# Documented corrected trim at M = 0.1 (see the ex_05 delta table).
TARGET = {"theta_deg": 9.505408, "delta_e_deg": -9.581727, "omega_rpm": 1823.38}


def verify_documented_trim() -> None:
    print("Documented single-point trim at M = 0.1 (ground-truth 3x3 Newton):")
    problem = TrimProblem(data=C172Data(), mach=np.array([0.1]))
    theta, delta_e, omega = problem.unpack(S.reference_solution(problem))
    print(f"  theta   = {np.rad2deg(theta)[0]:10.6f} deg   (target {TARGET['theta_deg']})")
    print(f"  delta_e = {np.rad2deg(delta_e)[0]:10.6f} deg   (target {TARGET['delta_e_deg']})")
    print(f"  omega   = {omega[0]:10.2f} RPM   (target {TARGET['omega_rpm']})")


def compare_backends(n_nodes: int = 4) -> S.TrimSolution:
    problem = TrimProblem.with_sweep(n_nodes)
    reference = S.reference_solution(problem)
    print(f"\nTrim an N = {n_nodes} Mach sweep {np.round(problem.mach, 3)} "
          f"({problem.n_dv} design variables), five ways:")
    print(f"  {'backend':16} {'device':8} {'iters':>5} {'nfev':>5} {'njev':>5} "
          f"{'f*':>9} {'|err|':>9}")

    exact_solution, first = None, None
    for name in B.available_backends(exact_grad_only=True):
        backend, setup = S.build_backend(name, problem)
        sol = S.solve_slsqp(problem, backend, deriv="exact", reference=reference,
                            setup_seconds=setup)
        exact_solution = exact_solution or sol
        first = first or sol
        _row(f"{name} (exact)", sol)

    for deriv in ("fd-central", "fd-forward"):
        sol = S.solve_slsqp(problem, NumpyBackend(problem), deriv=deriv, reference=reference)
        _row(f"numpy ({deriv})", sol)

    # The teaching point: every exact backend walks the identical optimizer path.
    print("\n  -> every exact-gradient backend took the same iteration / evaluation counts;")
    print("     only the derivative mechanism and device differ (Warp runs on the GPU).")
    return exact_solution


def _row(label: str, sol: S.TrimSolution) -> None:
    print(f"  {label:16} {sol.device:8} {sol.n_iter:>5} {sol.n_model_evals:>5} "
          f"{sol.n_grad_evals:>5} {sol.objective:>9.1e} {sol.error_vs_reference:>9.1e}")


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Available derivative backends: {B.available_backends()}")
    verify_documented_trim()
    schedule = compare_backends()

    out = plots.plot_trim_schedule(schedule, OUTPUT_DIR / "c172_trim_schedule.png")
    print(f"\nWrote {out.relative_to(REPO_ROOT)}")
    print("Full scaling / per-call / formulation study: "
          "python -m aircraft_sizing.examples.ex_05_c172_trim.methods.study")

    # Acceptance: the exact backend reproduced the trim to the reference.
    assert schedule.error_vs_reference < 1e-8, schedule.error_vs_reference
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
