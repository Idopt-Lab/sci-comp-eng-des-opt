"""Benchmark study CLI: scaling, per-call gradient cost, and formulation comparison.

Run as a module::

    python -m aircraft_sizing.examples.ex_05_c172_trim.methods.study            # standard
    python -m aircraft_sizing.examples.ex_05_c172_trim.methods.study --quick     # fast smoke run

It writes CSVs to ``docs/assets/data/`` and figures to ``docs/assets/images/``.  Every
row carries the platform/device stamp.  Only installed backends are exercised.

Timing rules (see solve.py): a backend's setup (graph build / JIT / CUDA context) is
reported separately and never counted in ``solve_s``; each solve time is the median of
a few repeats; the Warp source always runs on its preferred device (GPU if present)
and the device is recorded per row -- there is no CPU-vs-GPU axis.
"""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import backends as B
from .problem import TrimProblem
from .solve import build_backend, platform_stamp, reference_solution, solve_ga, solve_slsqp

_EXAMPLE_DIR = Path(__file__).resolve().parents[1]
_DATA_DIR = _EXAMPLE_DIR / "docs" / "assets" / "data"
_IMAGE_DIR = _EXAMPLE_DIR / "docs" / "assets" / "images"

_SCALING_NODES = (1, 2, 4, 8, 16, 32, 64, 128)
_GRADIENT_NODES = (1, 4, 16, 64, 256, 1024, 4096)
_QUICK_SCALING_NODES = (1, 2, 4, 8)
_QUICK_GRADIENT_NODES = (1, 4, 16, 64)
_GA_MAX_NODES = 4
#: Finite differences cost O(n) model calls per gradient, so cap them well below the
#: exact backends to keep the full study tractable (the trend is clear by here).
_FD_MAX_NODES = 16


def _flatten_stamp(stamp: dict[str, str]) -> dict[str, str]:
    return {f"stamp_{k}": v for k, v in stamp.items()}


def _solve_row(solution, stamp) -> dict:
    return {
        "n_nodes": solution.n_nodes,
        "n_dv": 3 * solution.n_nodes,
        "source": solution.source,
        "deriv": solution.deriv,
        "device": solution.device,
        "objective": solution.objective,
        "n_iter": solution.n_iter,
        "n_model_evals": solution.n_model_evals,
        "n_grad_evals": solution.n_grad_evals,
        "setup_s": solution.setup_seconds,
        "solve_s": solution.solve_seconds,
        "model_s": solution.model_seconds,
        "optimizer_s": solution.optimizer_seconds,
        "error_vs_reference": solution.error_vs_reference,
        **_flatten_stamp(stamp),
    }


def run_scaling_study(node_counts=_SCALING_NODES, *, repeats: int = 3, ga_max_nodes: int = _GA_MAX_NODES,
                      fd_max_nodes: int = _FD_MAX_NODES) -> pd.DataFrame:
    """SLSQP (every backend+deriv) and GA cost versus problem size."""
    stamp = platform_stamp()
    exact = B.available_backends(exact_grad_only=True)

    rows: list[dict] = []
    for n in node_counts:
        problem = TrimProblem.with_sweep(n)
        reference = reference_solution(problem)
        combos = [(name, "exact") for name in exact]
        if n <= fd_max_nodes:  # finite differences get prohibitive at large N
            combos += [("numpy", "fd-central"), ("numpy", "fd-forward")]
        for name, deriv in combos:
            backend, setup = build_backend(name, problem)
            # Median solve time over repeats (solution is deterministic).
            samples = [
                solve_slsqp(problem, backend, deriv=deriv, reference=reference,
                            setup_seconds=setup, stamp=stamp)
                for _ in range(repeats)
            ]
            best = min(samples, key=lambda s: s.solve_seconds)
            best.solve_seconds = statistics.median(s.solve_seconds for s in samples)
            rows.append(_solve_row(best, stamp))
            print(f"  N={n:<4} {name:>7}:{deriv:<10} iter={best.n_iter:<3} "
                  f"solve={best.solve_seconds * 1e3:7.1f}ms dev={best.device}")
        if n <= ga_max_nodes:
            ga = solve_ga(problem, reference=reference, stamp=stamp)
            rows.append(_solve_row(ga, stamp))
            print(f"  N={n:<4} {'ga':>7}:{'none':<10} eval={ga.n_model_evals} err={ga.error_vs_reference:.1e}")
    return pd.DataFrame(rows)


def run_gradient_benchmark(node_counts=_GRADIENT_NODES, *, repeats: int = 10) -> pd.DataFrame:
    """Per-call analytic gradient time versus problem size, per exact backend."""
    stamp = platform_stamp()
    rows: list[dict] = []
    for n in node_counts:
        problem = TrimProblem.with_sweep(n)
        x = problem.unscale(problem.z0())
        for name in B.available_backends(exact_grad_only=True):
            backend, _ = build_backend(name, problem)  # warm-up happens in the constructor
            times = []
            for _ in range(repeats):
                start = time.perf_counter()
                backend.grad(x)
                times.append(time.perf_counter() - start)
            rows.append({
                "n_nodes": n,
                "n_dv": 3 * n,
                "backend": name,
                "device": getattr(backend, "device", "cpu"),
                "per_call_ms": statistics.median(times) * 1e3,
                **_flatten_stamp(stamp),
            })
            print(f"  N={n:<5} {name:>7} grad {statistics.median(times) * 1e3:8.3f}ms "
                  f"dev={getattr(backend, 'device', 'cpu')}")
    return pd.DataFrame(rows)


def run_formulation_study(n_nodes: int = 4) -> pd.DataFrame:
    """Compare three JAX formulations of the same trim: half-sq, norm, equality-constrained.

    All find the same optimum; the point is how the formulation changes the SLSQP path.
    """
    import jax
    import jax.numpy as jnp
    import scipy.optimize

    from . import model

    problem = TrimProblem.with_sweep(n_nodes)
    data, mach = problem.data, jnp.asarray(problem.mach)
    reference = reference_solution(problem)
    active = jnp.array([0, 2, 4])  # only u_dot, w_dot, q_dot are non-trivially zero

    def residual_scaled(z):
        # The 3N active, scaled residuals -- a square system (3N eqs, 3N DVs).
        x = problem.lower_physical + z * (problem.upper_physical - problem.lower_physical)
        theta, delta_e, omega = problem.unpack(x)
        r = model.trim_residual(data, mach, theta, delta_e, omega, jnp)
        scale = model.residual_scale(data, jnp)
        return (r[:, active] / scale[active]).reshape(-1)

    half_sq = jax.jit(lambda z: 0.5 * jnp.sum(residual_scaled(z) ** 2))
    norm = jax.jit(lambda z: jnp.linalg.norm(residual_scaled(z)))
    g_half = jax.jit(jax.grad(half_sq))
    g_norm = jax.jit(jax.grad(norm))
    jac_res = jax.jit(jax.jacobian(residual_scaled))
    z0 = problem.z0()
    rows = []

    for label, fun, jac in (("half_sq", half_sq, g_half), ("norm", norm, g_norm)):
        res = scipy.optimize.minimize(
            lambda z: float(fun(z)), z0, jac=lambda z: np.asarray(jac(z)),
            method="SLSQP", bounds=problem.bounds_z(), options={"ftol": 1e-20, "maxiter": 5000},
        )
        err = float(np.max(np.abs(problem.scale(problem.unscale(res.x)) - problem.scale(reference))))
        rows.append({"formulation": label, "n_iter": int(res.nit), "n_fev": int(res.nfev),
                     "objective": float(res.fun), "error_vs_reference": err})

    # Equality-constrained feasibility: find the feasible point (residual = 0) nearest
    # z0.  A constant objective makes SLSQP's convergence test fire immediately, so we
    # use the well-posed minimum-change objective 1/2||z - z0||^2 instead; the square
    # trim system has a unique feasible point regardless.
    z_start = np.asarray(z0, dtype=float)
    constraints = [{"type": "eq", "fun": lambda z: np.asarray(residual_scaled(z)),
                    "jac": lambda z: np.asarray(jac_res(z))}]
    res = scipy.optimize.minimize(
        lambda z: 0.5 * float(np.sum((z - z_start) ** 2)), z_start,
        jac=lambda z: np.asarray(z - z_start), method="SLSQP",
        bounds=problem.bounds_z(), constraints=constraints, options={"ftol": 1e-12, "maxiter": 5000},
    )
    err = float(np.max(np.abs(problem.scale(problem.unscale(res.x)) - problem.scale(reference))))
    rows.append({"formulation": "equality", "n_iter": int(res.nit), "n_fev": int(res.nfev),
                 "objective": 0.0, "error_vs_reference": err})
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="C172 trim benchmark study.")
    parser.add_argument("--quick", action="store_true", help="small node counts for a fast run")
    parser.add_argument("--no-figures", action="store_true", help="write CSVs only")
    args = parser.parse_args(argv)

    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    _IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    scaling_nodes = _QUICK_SCALING_NODES if args.quick else _SCALING_NODES
    gradient_nodes = _QUICK_GRADIENT_NODES if args.quick else _GRADIENT_NODES

    print(f"Available backends: {B.available_backends()}")
    print("Scaling study ...")
    scaling = run_scaling_study(scaling_nodes)
    scaling.to_csv(_DATA_DIR / "scaling.csv", index=False)

    print("Gradient per-call benchmark ...")
    gradient = run_gradient_benchmark(gradient_nodes)
    gradient.to_csv(_DATA_DIR / "gradient_per_call.csv", index=False)

    print("Formulation comparison ...")
    formulation = run_formulation_study()
    formulation.to_csv(_DATA_DIR / "formulation.csv", index=False)
    print(formulation.to_string(index=False))

    if not args.no_figures:
        from ..viz import plots

        reference_problem = TrimProblem.with_sweep(max(scaling_nodes))
        ref_backend, setup = build_backend("jax", reference_problem)
        schedule = solve_slsqp(reference_problem, ref_backend, deriv="exact",
                               reference=reference_solution(reference_problem), setup_seconds=setup)
        plots.plot_trim_schedule(schedule, _IMAGE_DIR / "trim_schedule.png")
        plots.plot_scaling(scaling, _IMAGE_DIR / "scaling_solve_time.png")
        plots.plot_scaling(scaling, _IMAGE_DIR / "scaling_model_evals.png",
                           y="n_model_evals", ylabel="model evaluations")
        plots.plot_gradient_per_call(gradient, _IMAGE_DIR / "gradient_per_call.png")
        print(f"Figures written to {_IMAGE_DIR}")

    print(f"CSVs written to {_DATA_DIR}")


if __name__ == "__main__":
    main()
