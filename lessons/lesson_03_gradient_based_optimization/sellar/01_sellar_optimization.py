r"""Lesson 3 - Step 1: wrap the Sellar MDA in an optimizer (the blue OPT pill).

This is the "week 3" payoff promised at the end of the Lesson 2 Sellar walkthrough:
take the converged MDA and drive it with a gradient-based optimizer.

General nonlinear program (the ``fmincon`` form)::

    min_x   f(x)
    s.t.    c(x)   <= 0        (nonlinear inequality)
            c_eq(x) = 0        (nonlinear equality)
            A x    <= b        (linear inequality)
            A_eq x  = b_eq     (linear equality)
            lb <= x <= ub      (bounds)

Sellar MDF statement::

    min   x^2 + z2 + y1 + exp(-y2)
    wrt   z1 in [-10, 10],  z2 in [0, 10],  x in [0, 10]
    s.t.  g1 = 3.16 - y1 <= 0
          g2 = y2 - 24   <= 0
    with  y1, y2 from the MDA:  y1 = z1^2 + z2 + x - 0.2 y2,  y2 = sqrt(y1) + z1 + z2

Terminology cheat sheet: *design variables* (z, x), *bounds* (lb/ub), *objective*
(f), *nonlinear inequality constraints* (g1, g2), *feasible set* (points satisfying
all constraints), *active constraint* (holds with equality at the optimum),
*Lagrangian* L = f + sum(lambda_i g_i), *KKT conditions* (stationarity +
feasibility + complementarity, lambda_i >= 0), *Lagrange multipliers* (lambda),
*iteration* (an optimizer step) vs *function evaluation* (one model solve),
*tolerance* (convergence target), *exit flag* (why the solver stopped).

Run from the repository root (with the ``eng-des-opt-course`` environment active)::

    python lessons/lesson_03_gradient_based_optimization/sellar/01_sellar_optimization.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import openmdao.api as om  # noqa: E402

from aircraft_sizing.examples.ex_01_asw.methods.components import CallCounter  # noqa: E402

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"

# Canonical Sellar optimum (OpenMDAO docs).
EXPECTED = {"obj": 3.183394, "z1": 1.977639, "z2": 0.0, "x": 0.0, "y1": 3.16, "y2": 3.755278}


class SellarDis1(om.ExplicitComponent):
    """Discipline 1: ``y1 = z1**2 + z2 + x - 0.2 * y2`` (finite-difference partials)."""

    def initialize(self):
        self.options.declare("counter", default=None, recordable=False)

    def setup(self):
        self.add_input("z", val=np.zeros(2))
        self.add_input("x", val=0.0)
        self.add_input("y2", val=1.0)
        self.add_output("y1", val=1.0)

    def setup_partials(self):
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        counter = self.options["counter"]
        if counter is not None:
            counter.tick("Dis1", "compute")
        z1, z2 = inputs["z"]
        outputs["y1"] = z1**2 + z2 + inputs["x"] - 0.2 * inputs["y2"]


class SellarDis2(om.ExplicitComponent):
    """Discipline 2: ``y2 = sqrt(y1) + z1 + z2`` (finite-difference partials)."""

    def initialize(self):
        self.options.declare("counter", default=None, recordable=False)

    def setup(self):
        self.add_input("z", val=np.zeros(2))
        self.add_input("y1", val=1.0)
        self.add_output("y2", val=1.0)

    def setup_partials(self):
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        counter = self.options["counter"]
        if counter is not None:
            counter.tick("Dis2", "compute")
        z1, z2 = inputs["z"]
        y1 = inputs["y1"]
        if y1.real < 0.0:
            y1 *= -1
        outputs["y2"] = y1**0.5 + z1 + z2


def build_problem(counter: CallCounter | None = None) -> om.Problem:
    """Sellar MDA wrapped in an SLSQP driver."""
    prob = om.Problem(reports=False)
    model = prob.model

    cycle = model.add_subsystem("cycle", om.Group(), promotes=["*"])
    cycle.add_subsystem("d1", SellarDis1(counter=counter),
                        promotes_inputs=["x", "z", "y2"], promotes_outputs=["y1"])
    cycle.add_subsystem("d2", SellarDis2(counter=counter),
                        promotes_inputs=["z", "y1"], promotes_outputs=["y2"])
    # NonlinearBlockGS converges the coupling; DirectSolver is REQUIRED for correct
    # *coupled total derivatives* (it factorizes the full coupled linear system).
    # Lesson 2 used the default LinearRunOnce, which is fine for analysis but gives
    # wrong totals across a feedback loop -- see 02_sellar_derivatives.py.
    cycle.nonlinear_solver = om.NonlinearBlockGS()
    cycle.linear_solver = om.DirectSolver()

    model.add_subsystem(
        "obj_cmp", om.ExecComp("obj = x**2 + z[1] + y1 + exp(-y2)", z=np.zeros(2), x=0.0),
        promotes=["x", "z", "y1", "y2", "obj"],
    )
    model.add_subsystem("con_cmp1", om.ExecComp("con1 = 3.16 - y1"), promotes=["con1", "y1"])
    model.add_subsystem("con_cmp2", om.ExecComp("con2 = y2 - 24.0"), promotes=["con2", "y2"])

    model.set_input_defaults("x", val=1.0)
    model.set_input_defaults("z", val=np.array([5.0, 2.0]))

    prob.driver = om.ScipyOptimizeDriver(optimizer="SLSQP", tol=1e-9)
    prob.model.add_design_var("z", lower=np.array([-10.0, 0.0]), upper=np.array([10.0, 10.0]))
    prob.model.add_design_var("x", lower=0.0, upper=10.0)
    prob.model.add_objective("obj")
    prob.model.add_constraint("con1", upper=0.0)
    prob.model.add_constraint("con2", upper=0.0)
    return prob


def _history(sql_path: Path) -> dict[str, list[float]]:
    cases = om.CaseReader(str(sql_path)).get_cases("driver")
    keys = ("obj", "con1", "con2", "y1", "y2", "x")
    history: dict[str, list[float]] = {k: [] for k in keys}
    history["z1"], history["z2"] = [], []
    for case in cases:
        for key in keys:
            history[key].append(float(case.get_val(key)[0]))
        z = case.get_val("z")
        history["z1"].append(float(z[0]))
        history["z2"].append(float(z[1]))
    return history


def _plot_history(history: dict[str, list[float]], outfile: Path) -> None:
    iterations = range(len(history["obj"]))
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    axes[0, 0].plot(iterations, history["obj"], "o-")
    axes[0, 0].set_title("objective")
    axes[0, 1].plot(iterations, history["con1"], "o-", label="g1 = 3.16 - y1")
    axes[0, 1].plot(iterations, history["con2"], "s-", label="g2 = y2 - 24")
    axes[0, 1].axhline(0.0, color="k", lw=0.8)
    axes[0, 1].set_title("constraints")
    axes[0, 1].legend(fontsize=8)
    axes[1, 0].plot(iterations, history["y1"], "o-", label="y1")
    axes[1, 0].plot(iterations, history["y2"], "s-", label="y2")
    axes[1, 0].set_title("coupling variables")
    axes[1, 0].legend(fontsize=8)
    axes[1, 1].plot(iterations, history["z1"], "o-", label="z1")
    axes[1, 1].plot(iterations, history["z2"], "s-", label="z2")
    axes[1, 1].plot(iterations, history["x"], "^-", label="x")
    axes[1, 1].set_title("design variables")
    axes[1, 1].legend(fontsize=8)
    for ax in axes.flat:
        ax.set_xlabel("driver iteration")
        ax.grid(True, alpha=0.3)
    fig.suptitle("Sellar SLSQP optimization history")
    fig.tight_layout()
    fig.savefig(outfile, dpi=130)
    plt.close(fig)


def _kkt_check(prob: om.Problem) -> dict[str, float]:
    """Qualitative KKT check for the one interior design variable (z1)."""
    totals = prob.compute_totals(of=["obj", "con1", "con2"], wrt=["z"])
    df_dz1 = float(totals["obj", "z"][0, 0])
    dg1_dz1 = float(totals["con1", "z"][0, 0])
    # Stationarity along z1 with the active constraint g1:  df/dz1 + lambda1 dg1/dz1 = 0.
    lambda1 = -df_dz1 / dg1_dz1
    return {
        "g1": float(prob.get_val("con1")[0]),
        "g2": float(prob.get_val("con2")[0]),
        "lambda1": lambda1,
        "stationarity_residual": abs(df_dz1 + lambda1 * dg1_dz1),
    }


def _write_counts(prob: om.Problem, counter: CallCounter, kkt: dict, outfile: Path) -> dict:
    result = prob.driver.result
    counts = {
        "slsqp_iterations": int(getattr(result, "iter_count", 0)),
        "model_evals": int(getattr(result, "model_evals", 0)),
        "deriv_evals": int(getattr(result, "deriv_evals", 0)),
        "dis1_compute": counter.counts.get(("Dis1", "compute"), 0),
        "dis2_compute": counter.counts.get(("Dis2", "compute"), 0),
    }
    lines = ["# Lesson 3 - Sellar optimization (generated)", ""]
    lines.append(f"- Optimum: obj = {prob.get_val('obj')[0]:.6f}, "
                 f"z = [{prob.get_val('z')[0]:.6f}, {prob.get_val('z')[1]:.6f}], "
                 f"x = {prob.get_val('x')[0]:.6f}")
    lines.append(f"- Coupling: y1 = {prob.get_val('y1')[0]:.6f}, y2 = {prob.get_val('y2')[0]:.6f}")
    lines.append("")
    lines.append("| Count | Value |")
    lines.append("|---|---|")
    for key, value in counts.items():
        lines.append(f"| {key} | {value} |")
    lines.append("")
    lines.append("## KKT at the optimum")
    lines.append(f"- g1 = 3.16 - y1 = {kkt['g1']:.3e}  (**active**)")
    lines.append(f"- g2 = y2 - 24  = {kkt['g2']:.3e}  (**inactive**)")
    lines.append(f"- lambda_1 = {kkt['lambda1']:.4f}  (> 0, as required)")
    lines.append(f"- z1-stationarity residual = {kkt['stationarity_residual']:.2e}")
    outfile.write_text("\n".join(lines), encoding="utf-8")
    return counts


def _plot_counts(counts: dict, outfile: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    labels = list(counts)
    ax.bar(labels, [counts[k] for k in labels], color="C0")
    ax.set_ylabel("count")
    ax.set_title("SLSQP iterations vs. model/discipline evaluations")
    ax.tick_params(axis="x", rotation=25)
    for i, k in enumerate(labels):
        ax.text(i, counts[k], str(counts[k]), ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(outfile, dpi=130)
    plt.close(fig)


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    counter = CallCounter()
    prob = build_problem(counter=counter)

    with tempfile.TemporaryDirectory() as tmp:
        sql = Path(tmp) / "sellar_opt.sql"
        recorder = om.SqliteRecorder(str(sql))
        prob.driver.add_recorder(recorder)
        prob.driver.recording_options["record_desvars"] = True
        prob.driver.recording_options["record_objectives"] = True
        prob.driver.recording_options["record_constraints"] = True
        prob.driver.recording_options["includes"] = ["y1", "y2"]

        prob.setup()
        prob.set_val("x", 1.0)
        prob.set_val("z", np.array([5.0, 2.0]))
        prob.run_driver()
        prob.cleanup()

        history = _history(sql)

    obj = float(prob.get_val("obj")[0])
    z = prob.get_val("z")
    x = float(prob.get_val("x")[0])
    y1 = float(prob.get_val("y1")[0])
    y2 = float(prob.get_val("y2")[0])
    print(f"Optimum: obj={obj:.6f}  z=[{z[0]:.6f}, {z[1]:.6f}]  x={x:.6f}  y1={y1:.6f}  y2={y2:.6f}")

    kkt = _kkt_check(prob)
    print(f"KKT: g1={kkt['g1']:.2e} (active), g2={kkt['g2']:.2e} (inactive), "
          f"lambda1={kkt['lambda1']:.4f} (>0)")

    _plot_history(history, OUTPUT_DIR / "sellar_opt_history.png")
    counts = _write_counts(prob, counter, kkt, OUTPUT_DIR / "sellar_opt_counts.md")
    _plot_counts(counts, OUTPUT_DIR / "sellar_opt_counts.png")
    print(f"Counts: SLSQP iters={counts['slsqp_iterations']}, model_evals={counts['model_evals']}, "
          f"deriv_evals={counts['deriv_evals']}, Dis1.compute={counts['dis1_compute']}, "
          f"Dis2.compute={counts['dis2_compute']}")

    # --- acceptance asserts -------------------------------------------------- #
    assert abs(obj - EXPECTED["obj"]) < 1e-4, obj
    assert abs(z[0] - EXPECTED["z1"]) < 1e-3, z[0]
    assert abs(z[1] - EXPECTED["z2"]) < 1e-3, z[1]
    assert abs(x - EXPECTED["x"]) < 1e-3, x
    assert abs(y1 - EXPECTED["y1"]) < 1e-3, y1
    assert abs(y2 - EXPECTED["y2"]) < 1e-3, y2
    assert kkt["lambda1"] > 0.0 and kkt["g2"] < 0.0
    print(f"\nWrote outputs to {OUTPUT_DIR.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
