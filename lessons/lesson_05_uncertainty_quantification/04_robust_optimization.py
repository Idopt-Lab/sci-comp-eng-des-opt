r"""Lesson 5 - Step 4: design under uncertainty (robust optimization with UQPCE).

The deterministic optimum (Step 2) is the canonical Sellar design, sitting right on
the ``g1: y1 >= 3.16`` constraint - so once the inputs scatter it is infeasible ~50%
of the time. Robust optimization fixes that by **reformulating the objective and
constraints in terms of statistics** instead of single values:

    deterministic:   min  obj                 s.t.  g <= 0
    robust:          min  E[obj]              s.t.  CI_upper(g) <= 0

Minimizing the *mean* objective while requiring the *upper confidence bound* of each
constraint to stay feasible forces the optimizer to buy margin against the
uncertainty. We build this with NASA's **UQPCE** OpenMDAO group:

* the uncertain Sellar model is vectorized over the 60 collocation points;
* ``MultiUQPCEGroup`` turns the 60-vector of each response into PCE statistics
  (``obj:mean``, ``con1:ci_upper``, ``con2:ci_upper``, ...);
* SLSQP drives the design, with ``obj:mean`` as objective and the ``ci_upper``
  constraints (<= 0) as robust feasibility.

We then Monte-Carlo both designs to show the payoff: the robust design pushes the
probability of violating ``g1`` from ~50% down to a few percent, at a modest increase
in mean objective - the price of reliability.

Run from the repository root (with the ``eng-des-opt-course`` environment active)::

    python lessons/lesson_05_uncertainty_quantification/04_robust_optimization.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import openmdao.api as om  # noqa: E402
from uqpce.mdao import interface  # noqa: E402
from uqpce.mdao.uqpcegroup import MultiUQPCEGroup  # noqa: E402

from sellar_uncertain import (  # noqa: E402
    DESIGN_BOUNDS, DESIGN_START, INPUT_YAML, SellarConst1, SellarConst2, SellarMDA,
    SellarObj, deterministic_optimum, ensure_run_matrix, sample_uncertain, solve_mda,
)

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"


def build_robust_problem():
    """Uncertain Sellar + MultiUQPCEGroup, driven by SLSQP on the statistics."""
    matrix_path = ensure_run_matrix()
    (var_basis, norm_sq, resampled_var_basis, aleatory_cnt, epistemic_cnt,
     resp_cnt, order, variables, sig, run_matrix) = interface.initialize(
        str(INPUT_YAML), str(matrix_path))

    prob = om.Problem(reports=False)
    model = prob.model
    model.add_subsystem("MDA", SellarMDA(vec_size=resp_cnt),
                        promotes_inputs=["z", "x", "a0", "a1", "a2"],
                        promotes_outputs=["y1", "y2"])
    model.add_subsystem("con_cmp1", SellarConst1(vec_size=resp_cnt),
                        promotes_inputs=["y1"], promotes_outputs=["con1"])
    model.add_subsystem("con_cmp2", SellarConst2(vec_size=resp_cnt),
                        promotes_inputs=["y2"], promotes_outputs=["con2"])
    model.add_subsystem("obj_func", SellarObj(vec_size=resp_cnt),
                        promotes_inputs=["z", "x", "y1", "y2", "a3"],
                        promotes_outputs=["obj"])
    model.add_subsystem(
        "UQPCE",
        MultiUQPCEGroup(
            significance=sig, var_basis=var_basis, norm_sq=norm_sq,
            resampled_var_basis=resampled_var_basis, tail="both",
            epistemic_cnt=epistemic_cnt, aleatory_cnt=aleatory_cnt,
            uncert_list=["con1", "con2", "obj"],
        ),
        promotes_inputs=["con1", "con2", "obj"],
        promotes_outputs=["con1:ci_upper", "con2:ci_upper",
                          "obj:mean", "obj:ci_lower", "obj:ci_upper"],
    )

    prob.driver = om.ScipyOptimizeDriver(optimizer="SLSQP", tol=1e-7)
    prob.driver.options["maxiter"] = 100

    model.set_input_defaults("x", DESIGN_START["x"])
    model.set_input_defaults("z", np.array(DESIGN_START["z"]))
    model.add_design_var("z", lower=[DESIGN_BOUNDS["z1"][0], DESIGN_BOUNDS["z2"][0]],
                         upper=[DESIGN_BOUNDS["z1"][1], DESIGN_BOUNDS["z2"][1]])
    model.add_design_var("x", lower=DESIGN_BOUNDS["x"][0], upper=DESIGN_BOUNDS["x"][1])

    # Robust reformulation: expected objective, confidence-bounded (<= 0) constraints.
    model.add_objective("obj:mean")
    model.add_constraint("con1:ci_upper", upper=0.0)
    model.add_constraint("con2:ci_upper", upper=0.0)

    prob.setup(force_alloc_complex=True)
    interface.set_vals(prob, variables, run_matrix)
    return prob


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # UQPCE's confidence-interval resampling draws from NumPy's legacy global RNG;
    # seed it so the robust optimum (and the committed outputs) are reproducible.
    np.random.seed(0)

    det = deterministic_optimum()
    print(f"Deterministic optimum: z=[{det['z'][0]:.4f}, {det['z'][1]:.4f}], "
          f"x={det['x']:.4f}, obj={det['obj']:.6f}  (canonical Sellar f*)")

    prob = build_robust_problem()
    prob.run_driver()
    rob = {
        "z": [float(prob.get_val("z")[0]), float(prob.get_val("z")[1])],
        "x": float(prob.get_val("x")[0]),
        "obj_mean": float(prob.get_val("obj:mean")[0]),
        "c1_ci_upper": float(prob.get_val("con1:ci_upper")[0]),
        "c2_ci_upper": float(prob.get_val("con2:ci_upper")[0]),
    }
    print(f"Robust optimum:        z=[{rob['z'][0]:.4f}, {rob['z'][1]:.4f}], "
          f"x={rob['x']:.4f}, obj:mean={rob['obj_mean']:.4f}")

    # Monte Carlo both designs to compare reliability (g1 violated when con1 > 0).
    rng = np.random.default_rng(5)
    a = sample_uncertain(100_000, rng)
    det_mc = solve_mda(det["z"], det["x"], **a)
    rob_mc = solve_mda(rob["z"], rob["x"], **a)
    det_viol = float((det_mc["con1"] > 0).mean())
    rob_viol = float((rob_mc["con1"] > 0).mean())
    print(f"P(g1 violated): deterministic={det_viol:.1%}  robust={rob_viol:.1%}")
    print(f"E[obj]: deterministic={det_mc['obj'].mean():.4f}  robust={rob_mc['obj'].mean():.4f}")

    _plot(det, rob, det_mc, rob_mc, det_viol, rob_viol, OUTPUT_DIR / "robust_vs_deterministic.png")
    _write_md(det, rob, det_mc, rob_mc, det_viol, rob_viol, OUTPUT_DIR / "robust_vs_deterministic.md")
    print(f"\nWrote outputs to {OUTPUT_DIR.relative_to(REPO_ROOT)}")

    # --- acceptance asserts -------------------------------------------------- #
    assert abs(det["obj"] - 3.183394) < 1e-3, det["obj"]      # canonical Sellar baseline
    assert rob_viol < det_viol - 0.2, (det_viol, rob_viol)    # robust is far more reliable
    assert rob_viol < 0.1, rob_viol                            # ~within the CI target
    assert rob_mc["obj"].mean() >= det_mc["obj"].mean() - 1e-6  # reliability costs objective
    return 0


def _plot(det, rob, det_mc, rob_mc, det_viol, rob_viol, outfile: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    ax = axes[0]
    bins = 80
    ax.hist(det_mc["con1"], bins=bins, density=True, alpha=0.5, color="C3",
            label=f"deterministic (P[viol]={det_viol:.0%})")
    ax.hist(rob_mc["con1"], bins=bins, density=True, alpha=0.5, color="C0",
            label=f"robust (P[viol]={rob_viol:.1%})")
    ax.axvline(0.0, color="k", lw=1.5)
    ax.set_title("const1 = 3.16 - y1 under uncertainty")
    ax.set_xlabel("const1  (feasible <= 0)")
    ax.set_ylabel("density")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    labels = ["z1", "z2", "x", "E[obj]"]
    det_vals = [det["z"][0], det["z"][1], det["x"], det_mc["obj"].mean()]
    rob_vals = [rob["z"][0], rob["z"][1], rob["x"], rob_mc["obj"].mean()]
    xpos = np.arange(len(labels))
    ax.bar(xpos - 0.2, det_vals, 0.4, color="C3", label="deterministic")
    ax.bar(xpos + 0.2, rob_vals, 0.4, color="C0", label="robust")
    ax.set_xticks(xpos)
    ax.set_xticklabels(labels)
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_title("design variables & mean objective")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle("Robust vs. deterministic Sellar design")
    fig.tight_layout()
    fig.savefig(outfile, dpi=130)
    plt.close(fig)


def _write_md(det, rob, det_mc, rob_mc, det_viol, rob_viol, outfile: Path) -> None:
    lines = ["# Lesson 5 - Robust vs. deterministic optimum (generated)", ""]
    lines.append("| Quantity | Deterministic | Robust |")
    lines.append("|---|---|---|")
    lines.append(f"| z1 | {det['z'][0]:.4f} | {rob['z'][0]:.4f} |")
    lines.append(f"| z2 | {det['z'][1]:.4f} | {rob['z'][1]:.4f} |")
    lines.append(f"| x | {det['x']:.4f} | {rob['x']:.4f} |")
    lines.append(f"| E[obj] | {det_mc['obj'].mean():.4f} | {rob_mc['obj'].mean():.4f} |")
    lines.append(f"| mean const1 | {det_mc['con1'].mean():.4f} | {rob_mc['con1'].mean():.4f} |")
    lines.append(f"| **P(g1 violated)** | **{det_viol:.1%}** | **{rob_viol:.1%}** |")
    lines.append("")
    lines.append(
        f"- Baseline is the canonical Sellar optimum (`obj = {det['obj']:.4f}`), which "
        f"holds `g1` exactly\n  and so is infeasible ~{det_viol:.0%} of the time under "
        f"uncertainty.\n"
        f"- The robust reformulation (`min E[obj]` s.t. `CI_upper(g) <= 0`) raises `y1` "
        f"(mostly via `z1`)\n  to hold `g1` across the scatter: the violation probability "
        f"drops from **{det_viol:.0%}** to **{rob_viol:.1%}**.\n"
        f"- The price is a higher expected objective "
        f"({det_mc['obj'].mean():.2f} -> {rob_mc['obj'].mean():.2f}): "
        f"**reliability costs performance**.\n"
        f"- Statistics come from UQPCE's order-2 polynomial chaos on 60 collocation "
        f"points, evaluated\n  inside every optimizer iteration."
    )
    outfile.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
