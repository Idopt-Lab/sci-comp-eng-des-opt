r"""Lesson 5 - Step 3: polynomial chaos with UQPCE (the cheap alternative to MC).

Monte Carlo is trustworthy but expensive - the previous script needed ~100,000
model runs to resolve the output statistics. A **polynomial chaos expansion (PCE)**
fits a polynomial surrogate in the random inputs from a small *designed* set of runs
(the UQPCE "run matrix"), then reads the mean and variance straight off the
polynomial coefficients and samples the cheap surrogate for the full distribution.

Here we use NASA's **UQPCE** exactly as intended:

1. the random variables + PCE order come from ``input.yaml``;
2. ``sellar_uncertain.ensure_run_matrix`` asks UQPCE for a reproducible run matrix
   (60 collocation points for the 15 order-2 basis terms);
3. we evaluate the Sellar model at those 60 points and ``pce.fit`` the expansion;
4. mean/variance come analytically from the coefficients; the confidence interval
   and the output distribution come from cheaply sampling the fitted PCE.

We compare every statistic against a large Monte Carlo reference computed on the
*same* input samples. The headline: PCE reproduces MC mean/std/CI (and the
probability of violating the active constraint ``g1``) to a few percent using
**60 model runs instead of 100,000**.

Run from the repository root (with the ``eng-des-opt-course`` environment active)::

    python lessons/lesson_05_uncertainty_quantification/03_pce_uqpce.py
"""

from __future__ import annotations

import contextlib
import io
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
    build_pce, deterministic_optimum, ensure_run_matrix, sample_matrix, solve_mda,
)

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
RESPONSES = ("obj", "con1", "con2")  # con1 (3.16 - y1) is the active constraint
N_MC = 100_000


def _stats(samples: np.ndarray) -> dict:
    lo, hi = np.percentile(samples, [2.5, 97.5])
    return {"mean": float(samples.mean()), "std": float(samples.std()),
            "ci": (float(lo), float(hi))}


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    det = deterministic_optimum()
    z, x = det["z"], det["x"]

    # 1. UQPCE run matrix (collocation points) + model responses there.
    matrix_path = ensure_run_matrix()
    Xt = np.loadtxt(matrix_path, ndmin=2)
    train = solve_mda(z, x, Xt[:, 0], Xt[:, 1], Xt[:, 2], Xt[:, 3])
    y_train = np.column_stack([train[r] for r in RESPONSES])
    n_train = Xt.shape[0]

    # 2. Fit the PCE (UQPCE prints some regression diagnostics - hush them).
    pce = build_pce()
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        pce.fit(Xt, y_train)
    pce_mean = np.real(pce._mean)
    pce_std = np.sqrt(np.real(pce._sigma_sq))

    # 3. Sample the cheap PCE surrogate, and Monte Carlo the true model, on the
    #    SAME input draws for a fair comparison.
    rng = np.random.default_rng(3)
    Xbig = sample_matrix(N_MC, rng)
    pce_pred = np.atleast_2d(pce.predict(Xbig)).reshape(N_MC, len(RESPONSES))
    mc_true = solve_mda(z, x, Xbig[:, 0], Xbig[:, 1], Xbig[:, 2], Xbig[:, 3])

    pce_stats = {r: _stats(pce_pred[:, i]) for i, r in enumerate(RESPONSES)}
    mc_stats = {r: _stats(mc_true[r]) for r in RESPONSES}
    # Constraints are feasible when <= 0, so a violation is a POSITIVE value.
    pce_viol = {r: float((pce_pred[:, i] > 0).mean()) for i, r in enumerate(RESPONSES)}
    mc_viol = {r: float((mc_true[r] > 0).mean()) for r in RESPONSES}

    print(f"PCE fit on {n_train} model runs; Monte Carlo reference on {N_MC:,} runs.")
    for i, r in enumerate(RESPONSES):
        print(f"  {r}: PCE mean={pce_stats[r]['mean']:.4f} std={pce_stats[r]['std']:.4f} "
              f"| MC mean={mc_stats[r]['mean']:.4f} std={mc_stats[r]['std']:.4f}")
    print(f"  P(const1>0): PCE={pce_viol['con1']:.1%}  MC={mc_viol['con1']:.1%}")

    _plot(pce_pred, mc_true, pce_viol, mc_viol, OUTPUT_DIR / "pce_vs_mc.png")
    _write_md(det, n_train, pce_mean, pce_std, pce_stats, mc_stats, pce_viol, mc_viol,
              OUTPUT_DIR / "pce_vs_mc.md")
    print(f"\nWrote outputs to {OUTPUT_DIR.relative_to(REPO_ROOT)}")

    # --- acceptance asserts -------------------------------------------------- #
    rel = abs(pce_stats["obj"]["mean"] - mc_stats["obj"]["mean"]) / abs(mc_stats["obj"]["mean"])
    assert rel < 0.02, rel                                   # PCE mean within 2% of MC
    assert abs(pce_stats["obj"]["std"] - mc_stats["obj"]["std"]) / mc_stats["obj"]["std"] < 0.1
    assert abs(pce_viol["con1"] - mc_viol["con1"]) < 0.05    # violation prob agrees
    return 0


def _plot(pce_pred, mc_true, pce_viol, mc_viol, outfile: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    bins = 80
    axes[0].hist(mc_true["obj"], bins=bins, density=True, alpha=0.5, color="C0",
                 label=f"Monte Carlo ({len(mc_true['obj']):,} runs)")
    axes[0].hist(pce_pred[:, 0], bins=bins, density=True, histtype="step", lw=2,
                 color="C3", label="PCE surrogate (60 runs)")
    axes[0].set_title("objective distribution")
    axes[0].set_xlabel("obj")
    axes[0].set_ylabel("density")
    axes[0].legend(fontsize=8)

    axes[1].hist(mc_true["con1"], bins=bins, density=True, alpha=0.5, color="C0",
                 label=f"MC (P[viol]={mc_viol['con1']:.0%})")
    axes[1].hist(pce_pred[:, 1], bins=bins, density=True, histtype="step", lw=2, color="C3",
                 label=f"PCE (P[viol]={pce_viol['con1']:.0%})")
    axes[1].axvline(0.0, color="k", lw=1.5)
    axes[1].set_title("const1 = 3.16 - y1  (feasible <= 0)")
    axes[1].set_xlabel("const1")
    axes[1].legend(fontsize=8)

    for ax in axes:
        ax.grid(True, alpha=0.3)
    fig.suptitle("Polynomial chaos vs. Monte Carlo (same input samples)")
    fig.tight_layout()
    fig.savefig(outfile, dpi=130)
    plt.close(fig)


def _write_md(det, n_train, pce_mean, pce_std, pce_stats, mc_stats, pce_viol, mc_viol,
              outfile: Path) -> None:
    lines = ["# Lesson 5 - Polynomial chaos (UQPCE) vs. Monte Carlo (generated)", ""]
    lines.append(f"Both evaluated at the deterministic optimum "
                 f"z = [{det['z'][0]:.4f}, {det['z'][1]:.4f}], x = {det['x']:.4f}.")
    lines.append(f"\n**PCE used {n_train} model runs; Monte Carlo used {N_MC:,}.**")
    lines.append("")
    lines.append("| Output | PCE mean (analytic) | MC mean | PCE std (analytic) | MC std | PCE 95% CI | MC 95% CI |")
    lines.append("|---|---|---|---|---|---|---|")
    for i, r in enumerate(RESPONSES):
        p, m = pce_stats[r], mc_stats[r]
        lines.append(
            f"| {r} | {pce_mean[i]:.4f} | {m['mean']:.4f} | {pce_std[i]:.4f} | {m['std']:.4f} "
            f"| [{p['ci'][0]:.3f}, {p['ci'][1]:.3f}] | [{m['ci'][0]:.3f}, {m['ci'][1]:.3f}] |"
        )
    lines.append("")
    lines.append(f"- Probability of violating the active constraint `const1` (y1 < 3.16): "
                 f"PCE **{pce_viol['con1']:.1%}** vs Monte Carlo **{mc_viol['con1']:.1%}**.")
    lines.append(f"- PCE mean/variance come straight from the {n_train}-run polynomial "
                 f"coefficients; the CI and\n  violation probability come from sampling the "
                 f"*cheap* fitted polynomial.")
    lines.append(f"- Same accuracy as Monte Carlo at ~{N_MC // n_train}x fewer true model "
                 f"evaluations - the reason PCE\n  is affordable *inside* an optimizer "
                 f"(next script).")
    outfile.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
