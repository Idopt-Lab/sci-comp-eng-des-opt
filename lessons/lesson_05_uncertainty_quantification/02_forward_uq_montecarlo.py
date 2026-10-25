r"""Lesson 5 - Step 2: forward uncertainty propagation by Monte Carlo (the baseline).

"Forward UQ" pushes the *input* distributions through the model to get the *output*
distributions. Monte Carlo is the simplest, most trusted way to do it: draw many
samples of the uncertain parameters, run the model at each, and look at the
resulting spread. It is the reference every cheaper method (next script: polynomial
chaos) must reproduce.

We first find the **deterministic optimum** - the design you get if you ignore the
uncertainty and fix every parameter at its mean. Because every factor has mean 1,
that is exactly the canonical Sellar optimum (``f* = 3.183394``). Then we propagate
uncertainty *at that design* and hit the catch: the deterministic optimum sits right
on the ``g1: y1 >= 3.16`` constraint boundary, so once the inputs scatter the
constraint is violated about half the time. That failure is the whole motivation for
the robust reformulation in ``04_robust_optimization.py``.

Outputs:

* histograms of ``obj``, ``const1`` (the active constraint) and ``const2`` with their
  mean / 95% CI and the probability of constraint violation;
* first-order and total **Sobol indices** for ``obj`` (variance-based sensitivity,
  via the pick-and-freeze estimator) - which uncertain input drives the scatter.

Run from the repository root (with the ``eng-des-opt-course`` environment active)::

    python lessons/lesson_05_uncertainty_quantification/02_forward_uq_montecarlo.py
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

from sellar_uncertain import (  # noqa: E402
    VAR_NAMES, deterministic_optimum, sample_matrix, sample_uncertain, solve_mda,
)

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
N_MC = 100_000
N_SOBOL = 30_000


def _sobol_indices(z, x, n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """First-order (S1) and total (ST) Sobol indices of obj via pick-and-freeze."""
    A = sample_matrix(n, rng)
    B = sample_matrix(n, rng)

    def obj_of(M):
        return solve_mda(z, x, M[:, 0], M[:, 1], M[:, 2], M[:, 3])["obj"]

    yA, yB = obj_of(A), obj_of(B)
    var_y = np.var(np.concatenate([yA, yB]))
    s1, st = np.zeros(4), np.zeros(4)
    for i in range(4):
        Ci = A.copy()
        Ci[:, i] = B[:, i]
        yCi = obj_of(Ci)
        s1[i] = np.mean(yB * (yCi - yA)) / var_y          # Saltelli 2010
        st[i] = 0.5 * np.mean((yA - yCi) ** 2) / var_y    # Jansen 1999
    return s1, st


def _plot(resp: dict, viol1: float, viol2: float, s1: np.ndarray, st: np.ndarray,
          outfile: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    obj = resp["obj"]
    ax = axes[0, 0]
    ax.hist(obj, bins=80, density=True, color="C0", alpha=0.7)
    ax.axvline(obj.mean(), color="k", ls="--", lw=1.2, label=f"mean={obj.mean():.3f}")
    lo, hi = np.percentile(obj, [2.5, 97.5])
    ax.axvspan(lo, hi, color="C1", alpha=0.15, label="95% CI")
    ax.set_title("objective distribution")
    ax.set_xlabel("obj")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    c1 = resp["con1"]
    ax.hist(c1, bins=80, density=True, color="C3", alpha=0.7)
    ax.axvline(0.0, color="k", lw=1.5)
    ax.set_title(f"const1 = 3.16 - y1  (active; P[violation] = {viol1:.1%})")
    ax.set_xlabel("const1  (feasible <= 0)")

    ax = axes[1, 0]
    c2 = resp["con2"]
    ax.hist(c2, bins=80, density=True, color="C2", alpha=0.7)
    ax.axvline(0.0, color="k", lw=1.5)
    ax.set_title(f"const2 = y2 - 24  (inactive; P[violation] = {viol2:.1%})")
    ax.set_xlabel("const2  (feasible <= 0)")

    ax = axes[1, 1]
    xpos = np.arange(4)
    ax.bar(xpos - 0.2, s1, 0.4, label="first-order $S_i$")
    ax.bar(xpos + 0.2, st, 0.4, label="total $S_{Ti}$")
    ax.set_xticks(xpos)
    ax.set_xticklabels(VAR_NAMES)
    ax.set_title("Sobol sensitivity of obj")
    ax.set_ylabel("index")
    ax.legend(fontsize=8)

    for ax in axes.flat:
        ax.grid(True, alpha=0.3)
    fig.suptitle("Monte Carlo forward UQ at the deterministic optimum")
    fig.tight_layout()
    fig.savefig(outfile, dpi=130)
    plt.close(fig)


def _write_stats(det: dict, resp: dict, viol1: float, viol2: float,
                 s1: np.ndarray, st: np.ndarray, outfile: Path) -> None:
    obj = resp["obj"]
    lo, hi = np.percentile(obj, [2.5, 97.5])
    lines = ["# Lesson 5 - Monte Carlo forward UQ (generated)", ""]
    lines.append(f"Deterministic optimum (uncertain params fixed at their means = classic "
                 f"Sellar): z = [{det['z'][0]:.4f}, {det['z'][1]:.4f}], x = {det['x']:.4f}, "
                 f"obj = {det['obj']:.6f}.")
    lines.append(f"\nPropagated with {N_MC:,} Monte Carlo samples at that design:")
    lines.append("")
    lines.append("| Output | mean | std | 95% CI | P(violation) |")
    lines.append("|---|---|---|---|---|")
    lines.append(f"| obj | {obj.mean():.4f} | {obj.std():.4f} | [{lo:.3f}, {hi:.3f}] | - |")
    lines.append(f"| const1 = 3.16 - y1 | {resp['con1'].mean():.4f} | {resp['con1'].std():.4f} "
                 f"| - | {viol1:.1%} |")
    lines.append(f"| const2 = y2 - 24 | {resp['con2'].mean():.4f} | {resp['con2'].std():.4f} "
                 f"| - | {viol2:.1%} |")
    lines.append("")
    lines.append("## Sobol sensitivity of obj")
    lines.append("")
    lines.append("| input | first-order S_i | total S_Ti |")
    lines.append("|---|---|---|")
    for i, name in enumerate(VAR_NAMES):
        lines.append(f"| {name} | {s1[i]:.3f} | {st[i]:.3f} |")
    lines.append("")
    lines.append(
        f"The deterministic optimum puts the active constraint `const1` right on its "
        f"boundary, so under\nuncertainty it is violated **{viol1:.0%}** of the time - a "
        f"design that is optimal on paper but\nunreliable in practice. `const2` keeps a "
        f"wide margin and never binds. This motivates the\nrobust reformulation in "
        f"`04_robust_optimization.py`."
    )
    outfile.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    det = deterministic_optimum()
    print(f"Deterministic optimum: z=[{det['z'][0]:.4f}, {det['z'][1]:.4f}], "
          f"x={det['x']:.4f}, obj={det['obj']:.6f}")

    rng = np.random.default_rng(1)
    a = sample_uncertain(N_MC, rng)
    resp = solve_mda(det["z"], det["x"], **a)
    viol1 = float((resp["con1"] > 0).mean())   # g1 violated when y1 < 3.16, i.e. con1 > 0
    viol2 = float((resp["con2"] > 0).mean())
    print(f"obj: mean={resp['obj'].mean():.4f} std={resp['obj'].std():.4f}")
    print(f"P(const1>0)={viol1:.1%} (active), P(const2>0)={viol2:.1%} (inactive)")

    s1, st = _sobol_indices(det["z"], det["x"], N_SOBOL, np.random.default_rng(2))
    print("Sobol (first-order) of obj: " + ", ".join(f"{n}={v:.3f}" for n, v in zip(VAR_NAMES, s1)))

    _plot(resp, viol1, viol2, s1, st, OUTPUT_DIR / "mc_output_dists.png")
    _write_stats(det, resp, viol1, viol2, s1, st, OUTPUT_DIR / "mc_stats.md")
    print(f"\nWrote outputs to {OUTPUT_DIR.relative_to(REPO_ROOT)}")

    # --- acceptance asserts -------------------------------------------------- #
    assert det["success"]
    assert abs(det["obj"] - 3.183394) < 1e-3, det["obj"]  # recovers canonical Sellar
    assert 0.4 < viol1 < 0.6, viol1            # det. optimum ~50% infeasible on g1
    assert viol2 < 0.01, viol2                 # g2 keeps a wide margin
    assert abs(s1.sum() - 1.0) < 0.25, s1.sum()  # first-order indices ~ sum to 1 here
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
