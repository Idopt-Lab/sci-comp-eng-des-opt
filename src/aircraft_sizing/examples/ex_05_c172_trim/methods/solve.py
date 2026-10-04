"""Solvers for the C172 trim problem and the result record they return.

``reference_solution`` is the ground truth: because a symmetric trim has only three
active residuals (Fx = Fz = pitching-moment = 0) per node, each node is a small 3x3
root-find, solved independently with :func:`scipy.optimize.root`.  Every
optimization backend is checked against it.

``solve_slsqp`` runs the *identical* gradient-based solve for every source backend
and derivative method, so their iteration counts and solutions can be compared
apples-to-apples; ``solve_ga`` runs a gradient-free pymoo GA on the same objective.
Both return a :class:`TrimSolution` carrying counts, split timings, the error versus
the reference, and a platform/device stamp.

Timing rule: a backend's setup (graph build / JIT / CUDA context) is measured by
``build_backend`` and reported separately; ``solve_seconds`` is the optimizer call
only, and ``model_seconds`` (time spent inside the backend) is split out of it.
"""

from __future__ import annotations

import platform
import subprocess
import sys
import time
from dataclasses import dataclass, field
from math import radians

import numpy as np
import scipy.optimize

from . import model
from .backends import make_backend
from .objective import CountingObjective
from .problem import TrimProblem

__all__ = [
    "reference_solution",
    "build_backend",
    "solve_slsqp",
    "solve_ga",
    "TrimSolution",
    "platform_stamp",
]

# Indices of the three active (non-trivially-zero) residuals: u_dot, w_dot, q_dot.
_ACTIVE = (0, 2, 4)

# SLSQP settings shared by every backend so the comparison is fair.
_SLSQP_FTOL = 1e-20
_SLSQP_MAXITER = 5000


def reference_solution(problem: TrimProblem) -> np.ndarray:
    """Return the physical block vector ``[theta | delta_e | omega]`` that trims.

    Each node is solved on its own with a dense 3x3 Newton (``scipy.optimize.root``,
    which finite-differences its own Jacobian), independent of any backend.
    """
    data = problem.data
    theta = np.empty(problem.n_nodes)
    delta_e = np.empty(problem.n_nodes)
    omega = np.empty(problem.n_nodes)
    guess = np.array([radians(5.0), radians(-5.0), 2000.0])

    for node, mach in enumerate(np.asarray(problem.mach, dtype=float)):
        mach_1d = np.array([mach])

        def active_residual(xk: np.ndarray) -> np.ndarray:
            res = model.trim_residual(
                data, mach_1d, xk[0:1], xk[1:2], xk[2:3], np
            )[0]  # shape (6,)
            return np.array([res[i] for i in _ACTIVE])

        sol = scipy.optimize.root(active_residual, guess, method="hybr", tol=1e-13)
        if not sol.success:
            raise RuntimeError(f"reference trim failed at M={mach}: {sol.message}")
        theta[node], delta_e[node], omega[node] = sol.x

    return problem.pack(theta, delta_e, omega)


# --------------------------------------------------------------------------- #
# Result record + platform stamp
# --------------------------------------------------------------------------- #
def platform_stamp() -> dict[str, str]:
    """OS / CPU / Python / GPU identification for reproducibility of study numbers."""
    stamp = {
        "os": f"{platform.system()} {platform.release()}",
        "arch": platform.machine(),
        "cpu": platform.processor() or "unknown",
        "python": sys.version.split()[0],
        "gpu": "none",
        "gpu_driver": "n/a",
    }
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout.strip().splitlines()
        if out:
            name, _, driver = out[0].partition(",")
            stamp["gpu"] = name.strip()
            stamp["gpu_driver"] = driver.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return stamp


@dataclass
class TrimSolution:
    """Everything one solve produced: solution, counts, split timings, and a stamp."""

    source: str  # backend name, "reference", or "ga"
    deriv: str
    n_nodes: int
    mach: np.ndarray
    x_physical: np.ndarray
    objective: float
    n_iter: int
    n_model_evals: int
    n_grad_evals: int
    setup_seconds: float
    solve_seconds: float
    model_seconds: float
    device: str
    error_vs_reference: float  # max abs error vs the reference, in scaled (dimensionless) space
    stamp: dict[str, str] = field(default_factory=dict)

    @property
    def theta_deg(self) -> np.ndarray:
        return np.rad2deg(self.x_physical[: self.n_nodes])

    @property
    def delta_e_deg(self) -> np.ndarray:
        return np.rad2deg(self.x_physical[self.n_nodes : 2 * self.n_nodes])

    @property
    def omega_rpm(self) -> np.ndarray:
        return self.x_physical[2 * self.n_nodes :]

    @property
    def optimizer_seconds(self) -> float:
        """Solve time spent outside the model (optimizer / framework overhead)."""
        return max(self.solve_seconds - self.model_seconds, 0.0)

    def within_bounds(self, problem: TrimProblem, tol: float = 1e-9) -> bool:
        lo, hi = problem.lower_physical, problem.upper_physical
        return bool(np.all(self.x_physical >= lo - tol) and np.all(self.x_physical <= hi + tol))


# --------------------------------------------------------------------------- #
# Gradient-based solve
# --------------------------------------------------------------------------- #
def build_backend(name: str, problem: TrimProblem):
    """Construct a backend, returning ``(backend, setup_seconds)``.

    Setup includes any graph build, JIT, or CUDA-context creation -- all of which a
    backend front-loads in its constructor (JAX/Warp warm up there).
    """
    start = time.perf_counter()
    backend = make_backend(name, problem)
    return backend, time.perf_counter() - start


def solve_slsqp(
    problem: TrimProblem,
    backend,
    *,
    deriv: str = "exact",
    reference: np.ndarray | None = None,
    setup_seconds: float = 0.0,
    stamp: dict[str, str] | None = None,
) -> TrimSolution:
    """Run SLSQP in scaled space with one backend + derivative method."""
    objective = CountingObjective(problem, backend, deriv)
    z0 = problem.z0()

    start = time.perf_counter()
    result = scipy.optimize.minimize(
        objective.objective,
        z0,
        jac=objective.gradient,
        method="SLSQP",
        bounds=problem.bounds_z(),
        options={"ftol": _SLSQP_FTOL, "maxiter": _SLSQP_MAXITER},
    )
    solve_seconds = time.perf_counter() - start

    x_physical = problem.unscale(result.x)
    error = _error_vs_reference(problem, x_physical, reference)
    return TrimSolution(
        source=backend.name,
        deriv=deriv,
        n_nodes=problem.n_nodes,
        mach=np.asarray(problem.mach),
        x_physical=x_physical,
        objective=float(result.fun),
        n_iter=int(result.nit),
        n_model_evals=objective.n_value,
        n_grad_evals=objective.n_grad,
        setup_seconds=setup_seconds,
        solve_seconds=solve_seconds,
        model_seconds=objective.model_seconds,
        device=getattr(backend, "device", "cpu"),
        error_vs_reference=error,
        stamp=stamp or {},
    )


# --------------------------------------------------------------------------- #
# Gradient-free solve (pymoo GA)
# --------------------------------------------------------------------------- #
def solve_ga(
    problem: TrimProblem,
    *,
    pop_size: int = 100,
    seed: int = 1,
    max_evals: int = 300_000,
    ftol_abs: float = 1e-14,
    reference: np.ndarray | None = None,
    stamp: dict[str, str] | None = None,
) -> TrimSolution:
    """Gradient-free GA (pymoo) on the same scaled objective, using the numpy backend."""
    from pymoo.algorithms.soo.nonconvex.ga import GA
    from pymoo.core.problem import Problem
    from pymoo.core.termination import Termination
    from pymoo.operators.crossover.sbx import SBX
    from pymoo.operators.mutation.pm import PM
    from pymoo.operators.sampling.rnd import FloatRandomSampling
    from pymoo.optimize import minimize as pymoo_minimize

    from .backend_numpy import NumpyBackend

    backend = NumpyBackend(problem)

    class _TrimProblem(Problem):
        def __init__(self) -> None:
            super().__init__(n_var=problem.n_dv, n_obj=1, xl=0.0, xu=1.0)

        def _evaluate(self, z, out, *args, **kwargs):
            out["F"] = backend.value_batch(problem.unscale(z)).reshape(-1, 1)

    class _FtolOrEvals(Termination):
        def _update(self, algorithm) -> float:
            best = float("inf") if algorithm.opt is None else float(algorithm.opt.get("F")[0, 0])
            if best <= ftol_abs:
                return 1.0
            return min(1.0, algorithm.evaluator.n_eval / max_evals)

    algorithm = GA(
        pop_size=pop_size,
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(eta=20),
        eliminate_duplicates=True,
    )

    start = time.perf_counter()
    result = pymoo_minimize(
        _TrimProblem(), algorithm, _FtolOrEvals(), seed=seed, verbose=False
    )
    solve_seconds = time.perf_counter() - start

    x_physical = problem.unscale(np.asarray(result.X, dtype=float))
    error = _error_vs_reference(problem, x_physical, reference)
    return TrimSolution(
        source="ga",
        deriv="none",
        n_nodes=problem.n_nodes,
        mach=np.asarray(problem.mach),
        x_physical=x_physical,
        objective=float(result.F[0]),
        n_iter=int(result.algorithm.n_iter),
        n_model_evals=int(result.algorithm.evaluator.n_eval),
        n_grad_evals=0,
        setup_seconds=0.0,
        solve_seconds=solve_seconds,
        model_seconds=solve_seconds,  # the model dominates a gradient-free run
        device=backend.device,
        error_vs_reference=error,
        stamp=stamp or {},
    )


def _error_vs_reference(
    problem: TrimProblem, x_physical: np.ndarray, reference: np.ndarray | None
) -> float:
    """Max absolute error in *scaled* space, so radians and RPM compare fairly."""
    if reference is None:
        return float("nan")
    return float(np.max(np.abs(problem.scale(x_physical) - problem.scale(reference))))
