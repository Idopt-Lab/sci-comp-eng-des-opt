"""The *uncertain* Sellar problem - the model Lesson 5 propagates and optimizes.

This is the **same canonical Sellar problem** from Lessons 2-4, with four
coefficients turned into random variables. Each uncertain parameter ``a0..a3`` is a
*multiplicative* factor with **mean 1**, so at the mean values the equations reduce
exactly to the classic Sellar MDA and its deterministic optimum is the familiar
``f* = 3.183394`` at ``z = (1.9776, 0)``, ``x = 0`` (OpenMDAO's canonical result).
Adding the scatter is what makes it a UQ problem; the distributions follow the
style of NASA's UQPCE Sellar example.

    Discipline 1:  y1 = a0*z1^2 + z2 + x - 0.2*y2
    Discipline 2:  y2 = a1*sqrt(y1) + a2*z1 + z2
    Objective:     obj = x^2 + z2 + a3*y1 + exp(-y2)
    Constraints:   g1 = 3.16 - y1 <= 0,   g2 = y2 - 24 <= 0     (classic Sellar)

with ``a0 ~ N(1, 0.06)``, ``a1 ~ U(0.9, 1.1)``, ``a2 ~ N(1, 0.05)``,
``a3 ~ N(1, 0.10)``. The factors are attached to terms that are
*non-zero at the optimum* (``z1``, ``sqrt(y1)``, ``y1``) so every one of them
actually influences the response there.

At the deterministic optimum ``g1`` is active (``y1 = 3.16``), so once ``a0`` scatters
``y1`` the constraint is violated about half the time - the motivation for the robust
reformulation in ``04_robust_optimization.py``. ``g2`` keeps a huge margin and never
binds.

This module provides two views of the model:

* a **pure-numpy, vectorized** MDA (``solve_mda``) that solves many samples at once by
  fixed-point iteration - used by the Monte Carlo and PCE scripts;
* the **OpenMDAO components** (``SellarMDA`` + objective/constraints, analytic
  partials) used by the gradient-based robust optimization.

Run from the repository root (with the ``eng-des-opt-course`` environment active)::

    python lessons/lesson_05_uncertainty_quantification/sellar_uncertain.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import openmdao.api as om

INPUT_YAML = Path(__file__).resolve().parent / "input.yaml"

# Distribution specs, kept in lock-step with input.yaml. All means are 1, so the
# nominal (mean) problem is the classic Sellar from Lessons 2-4.
UNCERTAIN_VARS = [
    {"name": "a0", "distribution": "normal", "mean": 1.0, "stdev": 0.06},
    {"name": "a1", "distribution": "uniform", "low": 0.9, "high": 1.1},
    {"name": "a2", "distribution": "normal", "mean": 1.0, "stdev": 0.05},
    {"name": "a3", "distribution": "normal", "mean": 1.0, "stdev": 0.10},
]
VAR_NAMES = [v["name"] for v in UNCERTAIN_VARS]

# Classic Sellar design-variable bounds and start point (as in Lessons 2-4).
DESIGN_BOUNDS = {"z1": (-10.0, 10.0), "z2": (0.0, 10.0), "x": (0.0, 10.0)}
DESIGN_START = {"z": [5.0, 2.0], "x": 1.0}


def mean_values() -> dict[str, float]:
    """Nominal (mean) value of each uncertain parameter - all 1.0 here."""
    out = {}
    for v in UNCERTAIN_VARS:
        out[v["name"]] = v["mean"] if v["distribution"] == "normal" else 0.5 * (v["low"] + v["high"])
    return out


def sample_uncertain(n: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Draw ``n`` i.i.d. samples of each uncertain parameter (for Monte Carlo)."""
    return {
        "a0": rng.normal(1.0, 0.06, n),
        "a1": rng.uniform(0.9, 1.1, n),
        "a2": rng.normal(1.0, 0.05, n),
        "a3": rng.normal(1.0, 0.10, n),
    }


def sample_matrix(n: int, rng: np.random.Generator) -> np.ndarray:
    """``(n, 4)`` matrix of samples with columns ordered as ``VAR_NAMES``."""
    d = sample_uncertain(n, rng)
    return np.column_stack([d[name] for name in VAR_NAMES])


def solve_mda(z, x, a0, a1, a2, a3, iters: int = 200) -> dict[str, np.ndarray]:
    """Vectorized fixed-point MDA solve. Inputs ``a*`` are arrays of equal length."""
    z1, z2 = float(z[0]), float(z[1])
    a0, a1, a2, a3 = (np.asarray(v, dtype=float) for v in (a0, a1, a2, a3))
    y1 = np.full(a0.shape, 3.0)
    for _ in range(iters):
        y2 = a1 * np.sqrt(np.abs(y1)) + a2 * z1 + z2
        y1 = a0 * z1**2 + z2 + x - 0.2 * y2
    y2 = a1 * np.sqrt(np.abs(y1)) + a2 * z1 + z2
    obj = x**2 + z2 + a3 * y1 + np.exp(-y2)
    return {"y1": y1, "y2": y2, "obj": obj, "con1": 3.16 - y1, "con2": y2 - 24.0}


def deterministic_optimum() -> dict:
    """Optimize with every uncertain parameter fixed at its mean (= classic Sellar).

    This is the "ignore the uncertainty" baseline. It recovers the canonical Sellar
    optimum f* = 3.183394 at z = (1.9776, 0), x = 0 (constraints in <= 0 form, so the
    SciPy ``ineq`` functions return ``-con``).
    """
    from scipy.optimize import minimize

    m = mean_values()
    a = ([m["a0"]], [m["a1"]], [m["a2"]], [m["a3"]])

    def solve(v):
        return solve_mda([v[0], v[1]], v[2], *a)

    cons = [
        {"type": "ineq", "fun": lambda v: -solve(v)["con1"][0]},
        {"type": "ineq", "fun": lambda v: -solve(v)["con2"][0]},
    ]
    bounds = [DESIGN_BOUNDS["z1"], DESIGN_BOUNDS["z2"], DESIGN_BOUNDS["x"]]
    res = minimize(
        lambda v: solve(v)["obj"][0],
        x0=[DESIGN_START["z"][0], DESIGN_START["z"][1], DESIGN_START["x"]],
        bounds=bounds, constraints=cons, method="SLSQP",
        options={"ftol": 1e-9, "maxiter": 300},
    )
    return {"z": [float(res.x[0]), float(res.x[1])], "x": float(res.x[2]),
            "obj": float(res.fun), "success": bool(res.success)}


# --------------------------------------------------------------------------- #
# OpenMDAO components (vectorized over `vec_size` collocation points), analytic
# partials throughout. a0 enters Discipline 1; a1, a2 enter Discipline 2; a3 the
# objective. All are mean-1 factors on terms that are non-zero at the optimum.
# --------------------------------------------------------------------------- #
class SellarDis1(om.ExplicitComponent):
    """y1 = a0*z1**2 + z2 + x - 0.2*y2."""

    def initialize(self):
        self.options.declare("vec_size", types=int)

    def setup(self):
        n = self.options["vec_size"]
        arange = np.arange(n)
        self.add_input("z", shape=(2,))
        self.add_input("x")
        self.add_input("y2", shape=(n,))
        self.add_input("a0", shape=(n,))
        self.add_output("y1", shape=(n,))
        self.declare_partials("y1", ["z", "x"])
        self.declare_partials("y1", ["a0", "y2"], rows=arange, cols=arange)

    def compute(self, inputs, outputs):
        z1, z2 = inputs["z"]
        outputs["y1"] = inputs["a0"] * z1**2 + z2 + inputs["x"] - 0.2 * inputs["y2"]

    def compute_partials(self, inputs, partials):
        n = self.options["vec_size"]
        z1, z2 = inputs["z"]
        partials["y1", "z"] = np.column_stack([2 * z1 * inputs["a0"], np.ones(n)])
        partials["y1", "x"] = np.ones(n)
        partials["y1", "y2"] = -0.2
        partials["y1", "a0"] = z1**2


class SellarDis2(om.ExplicitComponent):
    """y2 = a1*sqrt(y1) + a2*z1 + z2."""

    def initialize(self):
        self.options.declare("vec_size", types=int)

    def setup(self):
        n = self.options["vec_size"]
        arange = np.arange(n)
        self.add_input("z", shape=(2,))
        self.add_input("y1", shape=(n,))
        self.add_input("a1", shape=(n,))
        self.add_input("a2", shape=(n,))
        self.add_output("y2", shape=(n,))
        self.declare_partials("y2", ["z"])
        self.declare_partials("y2", ["a1", "a2", "y1"], rows=arange, cols=arange)

    def compute(self, inputs, outputs):
        z1, z2 = inputs["z"]
        outputs["y2"] = inputs["a1"] * np.sqrt(np.abs(inputs["y1"])) + inputs["a2"] * z1 + z2

    def compute_partials(self, inputs, partials):
        n = self.options["vec_size"]
        z1, z2 = inputs["z"]
        root = np.sqrt(np.abs(inputs["y1"]))
        partials["y2", "z"] = np.column_stack([inputs["a2"], np.ones(n)])
        partials["y2", "y1"] = inputs["a1"] * 0.5 / root
        partials["y2", "a1"] = root
        partials["y2", "a2"] = z1


class SellarObj(om.ExplicitComponent):
    """obj = x**2 + z2 + a3*y1 + exp(-y2)."""

    def initialize(self):
        self.options.declare("vec_size", types=int)

    def setup(self):
        n = self.options["vec_size"]
        arange = np.arange(n)
        self.add_input("x")
        self.add_input("z", shape=(2,))
        self.add_input("y1", shape=(n,))
        self.add_input("y2", shape=(n,))
        self.add_input("a3", shape=(n,))
        self.add_output("obj", shape=(n,))
        self.declare_partials("obj", ["x", "z"])
        self.declare_partials("obj", ["y1", "y2", "a3"], rows=arange, cols=arange)

    def compute(self, inputs, outputs):
        z2 = inputs["z"][1]
        outputs["obj"] = (inputs["x"] ** 2 + z2 + inputs["a3"] * inputs["y1"]
                          + np.exp(-inputs["y2"]))

    def compute_partials(self, inputs, partials):
        n = self.options["vec_size"]
        z2 = inputs["z"][1]
        partials["obj", "x"] = 2 * inputs["x"] * np.ones(n)
        partials["obj", "z"] = np.column_stack([np.zeros(n), np.ones(n)])
        partials["obj", "y1"] = inputs["a3"]
        partials["obj", "y2"] = -np.exp(-inputs["y2"])
        partials["obj", "a3"] = inputs["y1"]


class SellarConst1(om.ExplicitComponent):
    """con1 = 3.16 - y1  (classic Sellar; feasible when <= 0)."""

    def initialize(self):
        self.options.declare("vec_size", types=int)

    def setup(self):
        n = self.options["vec_size"]
        arange = np.arange(n)
        self.add_input("y1", shape=(n,))
        self.add_output("con1", shape=(n,))
        self.declare_partials("con1", ["y1"], rows=arange, cols=arange)

    def compute(self, inputs, outputs):
        outputs["con1"] = 3.16 - inputs["y1"]

    def compute_partials(self, inputs, partials):
        partials["con1", "y1"] = -1.0


class SellarConst2(om.ExplicitComponent):
    """con2 = y2 - 24  (classic Sellar; feasible when <= 0)."""

    def initialize(self):
        self.options.declare("vec_size", types=int)

    def setup(self):
        n = self.options["vec_size"]
        arange = np.arange(n)
        self.add_input("y2", shape=(n,))
        self.add_output("con2", shape=(n,))
        self.declare_partials("con2", ["y2"], rows=arange, cols=arange)

    def compute(self, inputs, outputs):
        outputs["con2"] = inputs["y2"] - 24.0

    def compute_partials(self, inputs, partials):
        partials["con2", "y2"] = 1.0


class SellarMDA(om.Group):
    """Coupled uncertain Sellar MDA, vectorized over the collocation points."""

    def initialize(self):
        self.options.declare("vec_size", types=int)

    def setup(self):
        n = self.options["vec_size"]
        self.add_subsystem("d1", SellarDis1(vec_size=n),
                           promotes_inputs=["x", "z", "y2", "a0"], promotes_outputs=["y1"])
        self.add_subsystem("d2", SellarDis2(vec_size=n),
                           promotes_inputs=["z", "y1", "a1", "a2"], promotes_outputs=["y2"])
        self.nonlinear_solver = om.NewtonSolver(
            maxiter=50, rtol=1e-12, atol=1e-12, iprint=0,
            err_on_non_converge=False, solve_subsystems=True, stall_limit=4,
        )
        self.linear_solver = om.DirectSolver()

    def guess_nonlinear(self, inputs, outputs, residuals):
        n = self.options["vec_size"]
        outputs["y1"] = 3.16 * np.ones(n)
        outputs["y2"] = 3.76 * np.ones(n)
        self.run_apply_nonlinear()


# A fixed, reproducible UQPCE run matrix (collocation points) backs both the PCE
# comparison and the robust optimization, so they share one designed experiment.
RUN_MATRIX_PATH = Path(__file__).resolve().parent / "run_matrix.dat"
RUN_MATRIX_COUNT = 60  # ~4x the 15 order-2 PCE basis terms for a stable regression
RUN_MATRIX_SEED = 7


def build_pce(order: int | None = None):
    """A fresh UQPCE ``PCE`` built from ``input.yaml`` (variables added, not fit)."""
    import warnings

    from uqpce.pce.io import read_input_file
    from uqpce.pce.pce import PCE

    var_dict, settings = read_input_file(str(INPUT_YAML))
    for key in ("plot", "verbose"):
        settings.pop(key, None)
    if order is not None:
        settings["order"] = order
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        pce = PCE(outputs=False, plot=False, verbose=False, **settings)
        for value in var_dict.values():
            pce.add_variable(**value)
    return pce


def ensure_run_matrix(count: int = RUN_MATRIX_COUNT, seed: int = RUN_MATRIX_SEED,
                      path: Path = RUN_MATRIX_PATH) -> Path:
    """Generate (once) and return the path to a reproducible run matrix file."""
    if not path.exists():
        samples = build_pce().sample(count=count, seed=seed)
        np.savetxt(path, samples)
    return path


def main() -> None:
    det = deterministic_optimum()
    print("Deterministic (mean-value) optimum = classic Sellar:")
    print(f"  z = [{det['z'][0]:.5f}, {det['z'][1]:.5f}], x = {det['x']:.5f}, "
          f"obj = {det['obj']:.6f}  (expect f* = 3.183394)")
    rng = np.random.default_rng(0)
    res = solve_mda(det["z"], det["x"], **sample_uncertain(50_000, rng))
    print(f"  under uncertainty: obj mean={res['obj'].mean():.4f} std={res['obj'].std():.4f}; "
          f"P(g1 violated) = {(res['con1'] > 0).mean():.1%}")


if __name__ == "__main__":
    main()
