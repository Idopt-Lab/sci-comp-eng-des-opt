"""Lesson 3, script 04 - the ASW sizing loop solved as an *optimization*.

Lesson 2 closed the ASW sizing loop with a nonlinear *solver* (NLBGS / Aitken /
Newton / Broyden) driving the takeoff-gross-weight residual to zero.  Here we instead
hand the takeoff gross weight to an *optimizer* as a design variable, in two ways:

    opt:min_weight   ->  min  W_TO            s.t.  R(W_TO) >= 0
    opt:residual_ls  ->  min  (R(W_TO)/ref)^2        (unconstrained least squares)

where ``R(W_TO) = W_TO*(1 - Wf/WTO - We/WTO) - W_fixed`` is the sizing residual.
The two formulations differ in what the optimizer *minimizes*:

* **min_weight** treats closure as a constraint and the weight as the thing to
  minimize -- it finds the *lightest* aircraft that still closes.  Because R is
  monotone in W_TO over the feasible range, the constraint is active (R = 0) at the
  optimum, so it lands on the same closing weight.
* **residual_ls** treats closure itself as the objective -- it drives the squared
  residual to zero with no explicit constraint (a feasibility / least-squares view).

Both recover the same closing weight as every solver -- 57,615.87 lb -- illustrating
"optimizer as solver" (the SAND / IDF idea).

Fair cost accounting: EVERY method here computes any derivatives it needs by finite
difference (``deriv="fd"``), and we report the *total* discipline evaluations,
including the extra model calls finite differencing spends on the Jacobian.  That way
the gradient-using methods (Newton, Broyden, both optimizers) are not flattered by
hiding their derivative cost.

Run:  python lessons/lesson_03_gradient_based_optimization/asw/04_asw_as_optimization.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import openmdao.api as om  # noqa: E402

from aircraft_sizing.examples.ex_01_asw.methods.components import CallCounter  # noqa: E402
from aircraft_sizing.examples.ex_01_asw.methods.config import ASWSizingInputs  # noqa: E402
from aircraft_sizing.examples.ex_01_asw.methods.group import (  # noqa: E402
    OPT_FORMULATIONS,
    SOLVER_CHOICES,
    build_asw_optimization_problem,
    build_asw_problem,
)

BASELINE_TOGW_LB = 57_615.87
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
# All methods finite-difference any derivatives they need, so the model-call count is
# a fair, apples-to-apples total (it includes the FD Jacobian/gradient perturbations).
DERIV = "fd"
# The implicit Sizing state is declared with ref=1e4, so a *solver* recorder stores
# W_TO in scaled units; the *driver* recorder stores the physical design variable.
WTO_REF = 1.0e4


def _counts(counter: CallCounter) -> tuple[int, int]:
    """(model evaluations, derivative-assembly calls) from a CallCounter.

    Model evaluations count every discipline ``compute`` / residual evaluation,
    *including* the finite-difference perturbations spent forming derivatives.
    """
    model = sum(n for (_, k), n in counter.counts.items()
                if k in ("compute", "apply_nonlinear", "solve_nonlinear"))
    linearize = sum(n for (_, k), n in counter.counts.items() if k in ("partials", "linearize"))
    return model, linearize


def _history(sql: Path, source: str, var: str, initial: float, scale: float = 1.0) -> list[float]:
    """W_TO iterate history, starting from the initial guess then each recorded iterate.

    ``scale`` un-scales recorder values that are stored in a solver's scaled space.
    """
    history = [float(initial)]
    try:
        for case in om.CaseReader(str(sql)).get_cases(source, recurse=False):
            history.append(scale * float(np.asarray(case.get_val(var)).ravel()[0]))
    except Exception:
        pass
    return history


def run_solver(solver: str, work_dir: Path) -> dict:
    """Close the loop with a nonlinear solver; FD and analytic cost, plus the history."""
    inputs = ASWSizingInputs.baseline()

    def builder(deriv, counter):
        return build_asw_problem(inputs.params(), inputs.input_values(), solver=solver,
                                 deriv=deriv, counter=counter)

    # Analytic (JAX) cost: no finite-difference perturbations.
    counter = CallCounter()
    prob = builder("jax", counter)
    with np.errstate(divide="ignore", invalid="ignore"):
        prob.run_model()
    evals_analytic, _ = _counts(counter)

    # Finite-difference cost + recorded history.
    counter = CallCounter()
    prob = builder("fd", counter)
    initial = float(prob.get_val("sizing.takeoff_gross_weight")[0])
    sql = work_dir / f"solver_{solver}.sql"
    nl = prob.model.nonlinear_solver
    nl.add_recorder(om.SqliteRecorder(str(sql)))
    nl.recording_options["record_solver_residuals"] = True
    with np.errstate(divide="ignore", invalid="ignore"):
        prob.run_model()
    w_to = float(prob.get_val("sizing.takeoff_gross_weight")[0])
    iters = int(nl._iter_count)
    prob.cleanup()
    evals_fd, _ = _counts(counter)

    return {
        "label": f"solver:{solver}",
        "iters": iters,
        "evals_analytic": evals_analytic,
        "evals_fd": evals_fd,
        "w_to": w_to,
        "history": _history(sql, "root.nonlinear_solver", "sizing.takeoff_gross_weight",
                            initial, scale=WTO_REF),
    }


def run_optimizer(formulation: str, work_dir: Path) -> dict:
    """Close the loop with SLSQP; FD and analytic cost, plus the history."""
    inputs = ASWSizingInputs.baseline()

    def builder(deriv, counter):
        return build_asw_optimization_problem(inputs.params(), inputs.input_values(),
                                              formulation=formulation, deriv=deriv, counter=counter)

    counter = CallCounter()
    prob = builder("jax", counter)
    with np.errstate(divide="ignore", invalid="ignore"):
        prob.run_driver()
    evals_analytic, _ = _counts(counter)

    counter = CallCounter()
    prob = builder("fd", counter)
    initial = float(prob.get_val("dv.takeoff_gross_weight")[0])
    sql = work_dir / f"opt_{formulation}.sql"
    prob.driver.add_recorder(om.SqliteRecorder(str(sql)))
    prob.driver.recording_options["includes"] = ["dv.takeoff_gross_weight"]
    with np.errstate(divide="ignore", invalid="ignore"):
        prob.run_driver()
    w_to = float(prob.get_val("dv.takeoff_gross_weight")[0])
    iters = int(getattr(getattr(prob.driver, "result", None), "iter_count", 0))
    prob.cleanup()
    evals_fd, _ = _counts(counter)
    history = _history(sql, "driver", "dv.takeoff_gross_weight", initial)  # driver stores physical

    return {
        "label": f"opt:{formulation}",
        "iters": iters or max(len(history) - 1, 0),
        "evals_analytic": evals_analytic,
        "evals_fd": evals_fd,
        "w_to": w_to,
        "history": history,
    }


def _write_counts_table(rows: list[dict]) -> Path:
    header = (
        "`analytic evals` uses exact JAX partials (no finite-difference perturbations); "
        "`FD evals` finite-differences every required derivative, so the extra calls "
        "(`FD evals` − `analytic evals`) are exactly the model calls spent building "
        "derivatives by finite difference. The optimizers stay cheap even with FD because "
        "there is a **single design variable** (W_TO) and OpenMDAO's relevance analysis "
        "finite-differences only the two components on the W_TO → residual path "
        "(Structures, SizingResidual); the upstream disciplines do not depend on W_TO and "
        "are never perturbed. The nonlinear solvers re-linearize the full coupled model "
        "every iteration, so their FD cost is far higher.\n"
    )
    lines = [header,
             "| method | iterations | analytic evals | FD evals | FD overhead | W_TO (lb) |",
             "|---|---|---|---|---|---|"]
    for r in rows:
        overhead = r["evals_fd"] - r["evals_analytic"]
        lines.append(f"| `{r['label']}` | {r['iters']} | {r['evals_analytic']} | "
                     f"{r['evals_fd']} | +{overhead} | {r['w_to']:,.2f} |")
    path = OUTPUT_DIR / "asw_opt_counts.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _plot_histories(rows: list[dict]) -> Path:
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for r in rows:
        hist = r["history"]
        if len(hist) > 1:
            ax.plot(range(len(hist)), hist, "o-", markersize=4, label=r["label"])
    ax.axhline(BASELINE_TOGW_LB, color="k", ls="--", lw=1, label=f"closing weight ({BASELINE_TOGW_LB:,.0f} lb)")
    ax.set_xlabel("iteration")
    ax.set_ylabel(r"$W_{TO}$ (lb)")
    ax.set_title("ASW takeoff weight: solver iterates vs. optimizer iterates")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = OUTPUT_DIR / "asw_opt_vs_solver.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    with tempfile.TemporaryDirectory() as tmp:
        work_dir = Path(tmp)
        for solver in SOLVER_CHOICES:
            rows.append(run_solver(solver, work_dir))
        for formulation in OPT_FORMULATIONS:
            rows.append(run_optimizer(formulation, work_dir))

    print("ASW sizing: solver vs. optimizer  (model-call cost, analytic vs. finite difference)")
    print(f"  {'method':22} {'iters':>5} {'analytic':>9} {'FD':>6} {'FD over':>8} {'W_TO (lb)':>12}")
    for r in rows:
        overhead = r["evals_fd"] - r["evals_analytic"]
        print(f"  {r['label']:22} {r['iters']:>5} {r['evals_analytic']:>9} {r['evals_fd']:>6} "
              f"{'+' + str(overhead):>8} {r['w_to']:>12,.2f}")

    print("\nThe two optimizer formulations:")
    print("  opt:min_weight   minimizes W_TO subject to the sizing residual R >= 0")
    print("                   (lightest closing aircraft; the constraint is active at R = 0).")
    print("  opt:residual_ls  minimizes (R/ref)^2 with no constraint (feasibility / least squares).")
    print("  Both land on the same closing weight as every nonlinear solver.")

    print("\nWhy the optimizers stay cheap even with finite differences:")
    print("  'FD evals' > 'analytic evals', so the FD derivative calls ARE counted. But with a")
    print("  single design variable (W_TO), OpenMDAO's relevance analysis finite-differences only")
    print("  the two components on the W_TO -> residual path (Structures, SizingResidual); the")
    print("  upstream disciplines do not depend on W_TO and are never perturbed. The nonlinear")
    print("  solvers re-linearize the full coupled model every iteration, so they cost far more.")

    # Every method closes on the same takeoff weight.
    for r in rows:
        assert abs(r["w_to"] - BASELINE_TOGW_LB) < 1e-2, f"{r['label']} -> {r['w_to']}"
    # Each history must start at the initial guess and reach the closing weight.
    for r in rows:
        assert len(r["history"]) >= 2, r["label"]
        assert abs(r["history"][-1] - BASELINE_TOGW_LB) < 1.0, (r["label"], r["history"][-1])

    counts = _write_counts_table(rows)
    plot = _plot_histories(rows)
    print(f"\nWrote {counts}\n      {plot}")
    print(
        "\nDiscussion: an optimizer can stand in for a solver (SAND/IDF), but its "
        "intermediate iterates need not be feasible/consistent the way a fixed-point "
        "solver's are, and its progress is sensitive to design-variable and residual "
        "scaling (here W_TO and R are referenced to 1e4)."
    )


if __name__ == "__main__":
    main()
