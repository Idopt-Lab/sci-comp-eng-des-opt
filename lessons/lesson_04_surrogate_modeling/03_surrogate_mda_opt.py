r"""Lesson 4 - Step 3: optimize the Sellar problem through the surrogates.

This is the payoff. We train a Kriging surrogate for each discipline, wrap each in
an ``SmtSurrogateComp`` (identical interface to the analytic ``SellarDis1/2``), and
drop them into the *exact same* machinery as Lesson 3: the ``cycle`` group with
``NonlinearBlockGS`` + ``DirectSolver``, the objective/constraint ``ExecComp``s, and
the SLSQP driver. The optimizer never knows it is driving surrogates.

We then compare the surrogate-driven optimum against the canonical analytic optimum
from Lesson 3 (``f* = 3.183394`` at ``z = (1.9776, 0)``, ``x = 0``). Because the
surrogates are near-perfect (held-out ``R^2 ~ 1`` from ``02_fit_surrogates.py``),
the optimizer lands essentially on top of the true solution - demonstrating that a
trained surrogate can *replace* the analysis inside an MDO loop.

Caveat worth stating in class: this only works because the whole optimization
trajectory stays **inside the box the surrogates were trained on**
(``z1 in [-10, 10]``, ``z2, x in [0, 10]``, ``y2 in [0, 24]``, ``y1 in [0, 40]``).
Push a design variable outside that box and the surrogate is extrapolating - the
predictions, and the "optimum", become meaningless.

Run from the repository root (with the ``eng-des-opt-course`` environment active)::

    python lessons/lesson_04_surrogate_modeling/03_surrogate_mda_opt.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
for p in (SRC_ROOT, Path(__file__).resolve().parent):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import openmdao.api as om  # noqa: E402
from smt.sampling_methods import LHS  # noqa: E402
from smt.surrogate_models import KRG  # noqa: E402

from aircraft_sizing.examples.ex_01_asw.methods.components import CallCounter  # noqa: E402
from sellar_functions import DISCIPLINES  # noqa: E402
from surrogate_component import SmtSurrogateComp  # noqa: E402

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"

# Canonical analytic Sellar optimum (Lesson 3), our reference to beat.
ANALYTIC = {"obj": 3.183394, "z1": 1.977639, "z2": 0.0, "x": 0.0, "y1": 3.16, "y2": 3.755278}
N_TRAIN = {"d1": 80, "d2": 50}


def train_surrogate(disc_key: str, n: int, seed: int = 1):
    """Fit a Kriging surrogate on a normalized LHS DoE of one discipline."""
    disc = DISCIPLINES[disc_key]
    lo, hi = disc["xlimits"][:, 0], disc["xlimits"][:, 1]
    x_raw = LHS(xlimits=disc["xlimits"], criterion="ese", seed=seed)(n)
    model = KRG(print_global=False)
    model.set_training_values((x_raw - lo) / (hi - lo), disc["func"](x_raw))
    model.train()
    return model


def build_surrogate_problem(models: dict, counter: CallCounter | None = None) -> om.Problem:
    """The Lesson-3 Sellar-MDF problem with surrogate disciplines in the cycle."""
    prob = om.Problem(reports=False)
    model = prob.model

    cycle = model.add_subsystem("cycle", om.Group(), promotes=["*"])
    cycle.add_subsystem(
        "d1",
        SmtSurrogateComp(model=models["d1"], kind="d1",
                         xlimits=DISCIPLINES["d1"]["xlimits"], counter=counter),
        promotes_inputs=["x", "z", "y2"], promotes_outputs=["y1"],
    )
    cycle.add_subsystem(
        "d2",
        SmtSurrogateComp(model=models["d2"], kind="d2",
                         xlimits=DISCIPLINES["d2"]["xlimits"], counter=counter),
        promotes_inputs=["z", "y1"], promotes_outputs=["y2"],
    )
    # A surrogate is only an approximation, so the coupled system can be made
    # self-consistent only to about the surrogate's own accuracy (~1e-6 here), not to
    # the 1e-10 that the *exact* analytic model reaches (Lesson 3). We therefore match
    # the solver tolerance to the surrogate fidelity -- otherwise NLBGS chases a
    # residual floor it can never reach and prints spurious "failed to converge"
    # warnings. DirectSolver still gives correct coupled totals for the optimizer.
    cycle.nonlinear_solver = om.NonlinearBlockGS(maxiter=50, atol=1e-6, rtol=1e-6)
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
    model.add_design_var("z", lower=np.array([-10.0, 0.0]), upper=np.array([10.0, 10.0]))
    model.add_design_var("x", lower=0.0, upper=10.0)
    model.add_objective("obj")
    model.add_constraint("con1", upper=0.0)
    model.add_constraint("con2", upper=0.0)
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


def _plot_history(history: dict, outfile: Path) -> None:
    iters = range(len(history["obj"]))
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    axes[0, 0].plot(iters, history["obj"], "o-")
    axes[0, 0].axhline(ANALYTIC["obj"], color="C3", ls="--", lw=1, label="analytic $f^*$")
    axes[0, 0].set_title("objective")
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].plot(iters, history["con1"], "o-", label="g1 = 3.16 - y1")
    axes[0, 1].plot(iters, history["con2"], "s-", label="g2 = y2 - 24")
    axes[0, 1].axhline(0.0, color="k", lw=0.8)
    axes[0, 1].set_title("constraints")
    axes[0, 1].legend(fontsize=8)
    axes[1, 0].plot(iters, history["y1"], "o-", label="y1")
    axes[1, 0].plot(iters, history["y2"], "s-", label="y2")
    axes[1, 0].set_title("coupling variables (through surrogates)")
    axes[1, 0].legend(fontsize=8)
    axes[1, 1].plot(iters, history["z1"], "o-", label="z1")
    axes[1, 1].plot(iters, history["z2"], "s-", label="z2")
    axes[1, 1].plot(iters, history["x"], "^-", label="x")
    axes[1, 1].set_title("design variables")
    axes[1, 1].legend(fontsize=8)
    for ax in axes.flat:
        ax.set_xlabel("driver iteration")
        ax.grid(True, alpha=0.3)
    fig.suptitle("Sellar SLSQP optimization through Kriging surrogates")
    fig.tight_layout()
    fig.savefig(outfile, dpi=130)
    plt.close(fig)


def _write_compare(opt: dict, counter: CallCounter, outfile: Path) -> None:
    lines = ["# Lesson 4 - Surrogate vs. analytic Sellar optimum (generated)", ""]
    lines.append("| Quantity | Surrogate | Analytic | abs. error |")
    lines.append("|---|---|---|---|")
    for key in ("obj", "z1", "z2", "x", "y1", "y2"):
        err = abs(opt[key] - ANALYTIC[key])
        lines.append(f"| {key} | {opt[key]:.6f} | {ANALYTIC[key]:.6f} | {err:.2e} |")
    lines.append("")
    lines.append(f"- Discipline 1 surrogate evaluations: {counter.counts.get(('Dis1', 'compute'), 0)}")
    lines.append(f"- Discipline 2 surrogate evaluations: {counter.counts.get(('Dis2', 'compute'), 0)}")
    lines.append("")
    lines.append(
        "The optimizer runs the identical Lesson-3 MDF setup; only the discipline "
        "internals changed\n(analytic -> Kriging). The optimum matches the analytic "
        "solution because the surrogates are\naccurate across the trajectory, which "
        "stays inside the training box."
    )
    outfile.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    models = {k: train_surrogate(k, N_TRAIN[k]) for k in ("d1", "d2")}
    counter = CallCounter()
    prob = build_surrogate_problem(models, counter=counter)

    with tempfile.TemporaryDirectory() as tmp:
        sql = Path(tmp) / "surrogate_opt.sql"
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

    opt = {
        "obj": float(prob.get_val("obj")[0]),
        "z1": float(prob.get_val("z")[0]),
        "z2": float(prob.get_val("z")[1]),
        "x": float(prob.get_val("x")[0]),
        "y1": float(prob.get_val("y1")[0]),
        "y2": float(prob.get_val("y2")[0]),
    }
    print("Surrogate-driven optimum:")
    print(f"  obj={opt['obj']:.6f}  z=[{opt['z1']:.6f}, {opt['z2']:.6f}]  x={opt['x']:.6f}  "
          f"y1={opt['y1']:.6f}  y2={opt['y2']:.6f}")
    print(f"  |obj - analytic f*| = {abs(opt['obj'] - ANALYTIC['obj']):.2e}")

    _plot_history(history, OUTPUT_DIR / "surrogate_opt_history.png")
    _write_compare(opt, counter, OUTPUT_DIR / "surrogate_opt_compare.md")
    print(f"\nWrote outputs to {OUTPUT_DIR.relative_to(REPO_ROOT)}")

    # --- acceptance asserts -------------------------------------------------- #
    assert abs(opt["obj"] - ANALYTIC["obj"]) < 0.05, opt["obj"]
    assert abs(opt["z1"] - ANALYTIC["z1"]) < 0.1, opt["z1"]
    assert abs(opt["y1"] - ANALYTIC["y1"]) < 0.1, opt["y1"]
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
