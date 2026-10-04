r"""Lesson 3 - Step 2: derivatives for the Sellar MDA, five ways.

Covers, in order:

1. partials vs totals -- a component's local Jacobian vs the model's end-to-end one;
2. coupled totals by hand -- the 2x2 direct and adjoint linear systems for df/dz1
   through the MDA, matched against ``compute_totals``; and why ``LinearRunOnce``
   gives *wrong* totals across a feedback loop while ``DirectSolver`` gives right ones;
3. analytic partials -- ``compute_partials`` with the hand-derived Jacobian, checked
   with ``check_partials``;
4. finite differences -- forward (O(h)) and central (O(h^2)) error vs step size, and
   the error *floor* set by the MDA solver tolerance;
5. complex step -- no subtractive cancellation, so the error is flat down to h = 1e-200;
6. algorithmic differentiation -- a JAX discipline whose partials come from
   ``jax.jacfwd`` / ``jax.jacrev``;

and finishes with a table optimizing Sellar using each derivative source.

The full math -- the analytic-partial derivations, the direct/adjoint coupled-total
equations, how OpenMDAO assembles totals from partials, and why LinearRunOnce is wrong
across a feedback loop -- is in the companion file ``derivations.md`` next to this script.

Analytic partials:  dy1/d(z1,z2,x,y2) = (2*z1, 1, 1, -0.2);
                    dy2/d(z1,z2,y1)   = (1, 1, 0.5*y1**-0.5).

Run from the repository root (with the ``eng-des-opt-course`` environment active)::

    python lessons/lesson_03_gradient_based_optimization/sellar/02_sellar_derivatives.py
"""

from __future__ import annotations

import sys
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

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from aircraft_sizing.examples.ex_01_asw.methods.components import CallCounter  # noqa: E402

jax.config.update("jax_enable_x64", True)

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
EXPECTED_OBJ = 3.183394


# --------------------------------------------------------------------------- #
# Pure discipline functions (used by the analytic, JAX, and hand-derivative work)
# --------------------------------------------------------------------------- #
def d1(z1, z2, x, y2):
    return z1**2 + z2 + x - 0.2 * y2


def d2(z1, z2, y1):
    return y1**0.5 + z1 + z2


# --------------------------------------------------------------------------- #
# Disciplines with selectable derivative source
# --------------------------------------------------------------------------- #
class SellarDis1(om.ExplicitComponent):
    def initialize(self):
        self.options.declare("deriv", default="fd-forward")
        self.options.declare("counter", default=None, recordable=False)

    def setup(self):
        self.add_input("z", val=np.zeros(2))
        self.add_input("x", val=0.0)
        self.add_input("y2", val=1.0)
        self.add_output("y1", val=1.0)

    def setup_partials(self):
        deriv = self.options["deriv"]
        if deriv in ("analytic", "jax"):
            self.declare_partials("y1", ["z", "x", "y2"])
        elif deriv == "cs":
            self.declare_partials("*", "*", method="cs")
        else:
            form = "central" if deriv == "fd-central" else "forward"
            self.declare_partials("*", "*", method="fd", form=form)

    def compute(self, inputs, outputs):
        counter = self.options["counter"]
        if counter is not None:
            counter.tick("d1", "compute")
        z1, z2 = inputs["z"]
        outputs["y1"] = d1(z1, z2, inputs["x"], inputs["y2"])

    def compute_partials(self, inputs, partials):
        deriv = self.options["deriv"]
        z1 = inputs["z"][0]
        if deriv == "analytic":
            partials["y1", "z"] = np.array([[2.0 * z1, 1.0]])
            partials["y1", "x"] = 1.0
            partials["y1", "y2"] = -0.2
        elif deriv == "jax":
            z2, x, y2 = inputs["z"][1], inputs["x"][0], inputs["y2"][0]
            g = jax.jacfwd(lambda a: d1(a[0], a[1], a[2], a[3]))(jnp.array([z1, z2, x, y2]))
            partials["y1", "z"] = np.array([[float(g[0]), float(g[1])]])
            partials["y1", "x"] = float(g[2])
            partials["y1", "y2"] = float(g[3])


class SellarDis2(om.ExplicitComponent):
    def initialize(self):
        self.options.declare("deriv", default="fd-forward")
        self.options.declare("counter", default=None, recordable=False)

    def setup(self):
        self.add_input("z", val=np.zeros(2))
        self.add_input("y1", val=1.0)
        self.add_output("y2", val=1.0)

    def setup_partials(self):
        deriv = self.options["deriv"]
        if deriv in ("analytic", "jax"):
            self.declare_partials("y2", ["z", "y1"])
        elif deriv == "cs":
            self.declare_partials("*", "*", method="cs")
        else:
            form = "central" if deriv == "fd-central" else "forward"
            self.declare_partials("*", "*", method="fd", form=form)

    def compute(self, inputs, outputs):
        counter = self.options["counter"]
        if counter is not None:
            counter.tick("d2", "compute")
        z1, z2 = inputs["z"]
        y1 = inputs["y1"]
        if np.real(y1) < 0.0:
            y1 = -y1
        outputs["y2"] = d2(z1, z2, y1)

    def compute_partials(self, inputs, partials):
        deriv = self.options["deriv"]
        z1, z2 = inputs["z"]
        y1 = inputs["y1"][0]
        if deriv == "analytic":
            partials["y2", "z"] = np.array([[1.0, 1.0]])
            partials["y2", "y1"] = 0.5 * y1**-0.5
        elif deriv == "jax":
            g = jax.jacrev(lambda a: d2(a[0], a[1], a[2]))(jnp.array([z1, z2, y1]))
            partials["y2", "z"] = np.array([[float(g[0]), float(g[1])]])
            partials["y2", "y1"] = float(g[2])


def build_problem(deriv="analytic", *, linear_solver="direct", force_alloc_complex=False,
                  nlbgs_atol=1e-12, counter=None) -> om.Problem:
    prob = om.Problem(reports=False)
    model = prob.model
    cycle = model.add_subsystem("cycle", om.Group(), promotes=["*"])
    cycle.add_subsystem("d1", SellarDis1(deriv=deriv, counter=counter),
                        promotes_inputs=["x", "z", "y2"], promotes_outputs=["y1"])
    cycle.add_subsystem("d2", SellarDis2(deriv=deriv, counter=counter),
                        promotes_inputs=["z", "y1"], promotes_outputs=["y2"])
    cycle.nonlinear_solver = om.NonlinearBlockGS(atol=nlbgs_atol, rtol=1e-99, maxiter=200)
    cycle.linear_solver = om.DirectSolver() if linear_solver == "direct" else om.LinearRunOnce()

    model.add_subsystem("obj_cmp", om.ExecComp("obj = x**2 + z[1] + y1 + exp(-y2)",
                                               z=np.zeros(2), x=0.0),
                        promotes=["x", "z", "y1", "y2", "obj"])
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
    prob.setup(force_alloc_complex=force_alloc_complex)
    return prob


# --------------------------------------------------------------------------- #
# (2) coupled totals by hand: direct and adjoint
# --------------------------------------------------------------------------- #
def hand_coupled_totals(z1, z2, x, y1, y2) -> dict[str, float]:
    """df/dz1 through the converged MDA via the direct and adjoint methods."""
    # Residuals R = y - f(y): dR/dy and dR/dz1 at the converged point.
    dR_dy = np.array([[1.0, 0.2], [-0.5 * y1**-0.5, 1.0]])
    dR_dz1 = np.array([-2.0 * z1, -1.0])
    df_dy = np.array([1.0, -np.exp(-y2)])  # d f / d (y1, y2); f has z2, not z1
    df_dz1_partial = 0.0

    # Direct:  dy/dz1 = -(dR/dy)^-1 dR/dz1 ;  total = df/dz1 + df/dy . dy/dz1
    dy_dz1 = -np.linalg.solve(dR_dy, dR_dz1)
    direct = df_dz1_partial + df_dy @ dy_dz1
    # Adjoint: (dR/dy)^T psi = (df/dy)^T ;  total = df/dz1 - psi . dR/dz1
    psi = np.linalg.solve(dR_dy.T, df_dy)
    adjoint = df_dz1_partial - psi @ dR_dz1
    return {"direct": float(direct), "adjoint": float(adjoint)}


def part2_coupled_totals() -> dict:
    prob = build_problem("analytic", linear_solver="direct")
    prob.run_model()
    z1, z2 = prob.get_val("z")
    x = prob.get_val("x")[0]
    y1, y2 = prob.get_val("y1")[0], prob.get_val("y2")[0]
    hand = hand_coupled_totals(z1, z2, x, y1, y2)
    om_df_dz1 = float(prob.compute_totals(of=["obj"], wrt=["z"])["obj", "z"][0, 0])

    # LinearRunOnce makes a single linear pass and ignores the feedback loop, so its
    # coupled total is wrong; DirectSolver factorizes the full coupled system.
    wrong = build_problem("analytic", linear_solver="runonce")
    wrong.run_model()
    runonce_df_dz1 = float(wrong.compute_totals(of=["obj"], wrt=["z"])["obj", "z"][0, 0])

    # Independent central-FD reference through the whole MDA (the ground truth).
    h = 1e-6
    fd_ref = (_total_f_of_z1(z1 + h, 1e-12) - _total_f_of_z1(z1 - h, 1e-12)) / (2 * h)
    return {
        "hand_direct": hand["direct"], "hand_adjoint": hand["adjoint"],
        "openmdao_direct": om_df_dz1, "runonce": runonce_df_dz1, "fd_reference": fd_ref,
    }


# --------------------------------------------------------------------------- #
# (3) analytic partials: check_partials
# --------------------------------------------------------------------------- #
def part3_check_partials() -> float:
    prob = build_problem("analytic", linear_solver="direct")
    prob.run_model()
    data = prob.check_partials(compact_print=True, out_stream=None)
    max_error = 0.0
    for comp in data.values():
        for pair in comp.values():
            err = pair["rel error"].forward
            if err is not None and np.isfinite(err):
                max_error = max(max_error, float(err))
    return max_error


# --------------------------------------------------------------------------- #
# (4) finite-difference error vs step; solver-tolerance floor
# --------------------------------------------------------------------------- #
def _total_f_of_z1(z1_value: float, atol: float) -> float:
    """Objective as a function of z1, re-solving the MDA to tolerance ``atol``."""
    prob = build_problem("analytic", linear_solver="direct", nlbgs_atol=atol)
    prob.set_val("z", np.array([z1_value, 2.0]))
    prob.set_val("x", 1.0)
    prob.run_model()
    return float(prob.get_val("obj")[0])


def part4_fd_error(outfile: Path) -> None:
    steps = np.logspace(-1, -14, 40)
    # (a) a pure partial: d2's dy2/dy1 = 0.5 * y1**-0.5, exact at y1 = 3.0.
    y1 = 3.0
    exact_partial = 0.5 * y1**-0.5
    fwd = np.abs((np.sqrt(y1 + steps) - np.sqrt(y1)) / steps - exact_partial)
    cen = np.abs((np.sqrt(y1 + steps) - np.sqrt(y1 - steps)) / (2 * steps) - exact_partial)

    # (b) a total through the MDA at two solver tolerances (error floor).
    reference = build_problem("analytic", linear_solver="direct")
    reference.run_model()
    exact_total = float(reference.compute_totals(of=["obj"], wrt=["z"])["obj", "z"][0, 0])
    z1_0 = 5.0
    coarse_steps = np.logspace(-1, -10, 20)
    floors = {}
    for atol in (1e-6, 1e-12):
        errs = []
        base = _total_f_of_z1(z1_0, atol)
        for h in coarse_steps:
            fd = (_total_f_of_z1(z1_0 + h, atol) - base) / h
            errs.append(abs(fd - exact_total))
        floors[atol] = np.array(errs)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].loglog(steps, fwd, "o-", label="forward  O(h)")
    axes[0].loglog(steps, cen, "s-", label="central  O(h^2)")
    axes[0].set_title(r"Partial $\partial y_2/\partial y_1 = \frac{1}{2} y_1^{-1/2}$")
    axes[0].legend()
    for atol, errs in floors.items():
        axes[1].loglog(coarse_steps, np.maximum(errs, 1e-20), "o-", label=f"NLBGS atol={atol:g}")
    axes[1].set_title("Total df/dz1 through the MDA (forward FD)")
    axes[1].legend()
    for ax in axes:
        ax.set_xlabel("step size h")
        ax.set_ylabel("absolute error")
        ax.grid(True, which="both", alpha=0.3)
    fig.suptitle("Finite-difference error: truncation, roundoff, and the solver-tolerance floor")
    fig.tight_layout()
    fig.savefig(outfile, dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# (5) complex step: flat to h = 1e-200
# --------------------------------------------------------------------------- #
def part5_complex_step() -> dict:
    y1 = 3.0
    exact = 0.5 * y1**-0.5
    results = {}
    for h in (1e-8, 1e-30, 1e-100, 1e-200):
        cs = np.imag(np.sqrt(y1 + 1j * h)) / h
        results[h] = abs(cs - exact)
    return results


# --------------------------------------------------------------------------- #
# (7) optimize with each derivative source
# --------------------------------------------------------------------------- #
def part7_comparison(outfile: Path) -> list[dict]:
    rows = []
    analytic_x = None
    for deriv in ("analytic", "jax", "cs", "fd-forward", "fd-central"):
        counter = CallCounter()
        prob = build_problem(deriv, linear_solver="direct",
                             force_alloc_complex=(deriv == "cs"), counter=counter)
        prob.set_val("z", np.array([5.0, 2.0]))
        prob.set_val("x", 1.0)
        prob.run_driver()
        x_star = np.concatenate([prob.get_val("z"), prob.get_val("x")])
        if deriv == "analytic":
            analytic_x = x_star
        result = prob.driver.result
        rows.append({
            "deriv": deriv,
            "obj": float(prob.get_val("obj")[0]),
            "dx": float(np.linalg.norm(x_star - analytic_x)),
            "iters": int(getattr(result, "iter_count", 0)),
            "grad_evals": int(getattr(result, "deriv_evals", 0)),
            # Actual discipline evaluations: analytic/jax compute no extra for partials,
            # complex-step and forward-FD perturb each input once, central-FD twice.
            "disc_calls": counter.total("compute"),
        })

    note = (
        "Discipline calls = total d1+d2 `compute` evaluations over the whole optimization. "
        "Analytic and JAX partials add *no* extra model calls (the Jacobian is formed "
        "in closed form); complex-step and forward finite difference perturb each input "
        "once per gradient, and central finite difference perturbs it twice -- so the "
        "model-call cost climbs even though the SLSQP path (iters, gradient evaluations) "
        "is identical."
    )
    lines = ["# Lesson 3 - Sellar optimization by derivative source (generated)", "", note, ""]
    lines.append("| Derivative source | f* | |dx*| vs analytic | iters | gradient evals | discipline calls |")
    lines.append("|---|---|---|---|---|---|")
    for r in rows:
        lines.append(f"| {r['deriv']} | {r['obj']:.6f} | {r['dx']:.2e} | {r['iters']} | "
                     f"{r['grad_evals']} | {r['disc_calls']} |")
    outfile.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return rows


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("== (2) coupled totals ==")
    totals = part2_coupled_totals()
    print(f"  hand direct   df/dz1 = {totals['hand_direct']:.6f}")
    print(f"  hand adjoint  df/dz1 = {totals['hand_adjoint']:.6f}")
    print(f"  OpenMDAO      df/dz1 = {totals['openmdao_direct']:.6f}")
    print(f"  LinearRunOnce df/dz1 = {totals['runonce']:.6f}  (WRONG; FD ref={totals['fd_reference']:.6f})")
    assert abs(totals["hand_direct"] - totals["openmdao_direct"]) < 1e-6
    assert abs(totals["hand_adjoint"] - totals["openmdao_direct"]) < 1e-6
    assert abs(totals["runonce"] - totals["fd_reference"]) > 1e-3  # runonce really is wrong

    print("== (3) analytic partials check_partials ==")
    max_err = part3_check_partials()
    print(f"  max relative partial error = {max_err:.2e}")
    assert max_err < 1e-6

    print("== (4) finite-difference error study ==")
    part4_fd_error(OUTPUT_DIR / "fd_error.png")
    print(f"  wrote {OUTPUT_DIR / 'fd_error.png'}")

    print("== (5) complex step ==")
    for h, err in part5_complex_step().items():
        print(f"  h={h:g}: error={err:.2e}")
    cs_errors = part5_complex_step()
    assert cs_errors[1e-200] < 1e-10  # flat, no cancellation

    print("== (7) derivative-source comparison ==")
    rows = part7_comparison(OUTPUT_DIR / "deriv_comparison.md")
    print(f"  {'source':10s} {'f*':>10s} {'|dx*|':>9s} {'iters':>5s} {'grad evals':>10s} "
          f"{'disc calls':>10s}")
    for r in rows:
        print(f"  {r['deriv']:10s} {r['obj']:10.6f} {r['dx']:9.1e} {r['iters']:5d} "
              f"{r['grad_evals']:10d} {r['disc_calls']:10d}")
    for r in rows:
        assert abs(r["obj"] - EXPECTED_OBJ) < 1e-4, (r["deriv"], r["obj"])
    # Finite differences must cost more discipline calls than analytic, and central
    # must cost more than forward -- the whole point of the comparison.
    calls = {r["deriv"]: r["disc_calls"] for r in rows}
    assert calls["fd-forward"] > calls["analytic"], calls
    assert calls["fd-central"] > calls["fd-forward"], calls

    print(f"\nWrote outputs to {OUTPUT_DIR.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
