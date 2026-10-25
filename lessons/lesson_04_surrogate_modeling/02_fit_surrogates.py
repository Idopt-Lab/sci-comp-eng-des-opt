r"""Lesson 4 - Step 2: fit surrogates to the DoE data and judge how good they are.

We draw a Latin-hypercube DoE for each Sellar discipline, evaluate the true
function at those points (``sellar_functions``), and fit three surrogate families
from SMT:

* **QP**  - quadratic polynomial (a global least-squares fit, low flexibility);
* **RBF** - radial basis functions (an interpolator built from distance kernels);
* **KRG** - Kriging / Gaussian process (interpolator + a built-in error estimate).

Inputs are first scaled to the unit box ``[0, 1]`` using each discipline's known
bounds. Kriging rescales internally so it does not care, but distance-based RBF is
very sensitive to feature scale - unscaled, its ``R^2`` here is *negative*; scaled,
it is ~0.9998. "Normalize before you fit a distance-based surrogate" is itself a
lesson, so we do it for every model uniformly.

Then we *judge the fit* - the part students most often skip:

* **Held-out test error** (``R2``, ``RMSE``, ``MAE``) on points the surrogate
  never saw - the honest measure of predictive accuracy.
* **Parity plots** (predicted vs. true) - the fit is good when points hug the
  45-degree line.
* **k-fold cross-validation** - rotate which slice of the *training* data is held
  out, so every point is predicted once; robust when data is scarce.
* **A learning curve** (test RMSE vs. training-set size) - shows a sparse DoE
  under-resolves the response and the error grows. This is the headline lesson;
  we plot RMSE on a log axis because ``R^2`` saturates near 1 on these smooth
  functions and hides the trend.

A nice wrinkle: Discipline 1 (``y1 = z1^2 + z2 + x - 0.2 y2``) *is* a quadratic, so
QP fits it essentially exactly; Discipline 2 carries a ``sqrt(y1)`` term, so QP
keeps a small irreducible bias while the interpolators (RBF, KRG) drive the test
error lower as data grows. Match the model to the function.

Run from the repository root (with the ``eng-des-opt-course`` environment active)::

    python lessons/lesson_04_surrogate_modeling/02_fit_surrogates.py
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
from smt.sampling_methods import LHS  # noqa: E402
from smt.surrogate_models import KRG, QP, RBF  # noqa: E402

from sellar_functions import DISCIPLINES  # noqa: E402

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"

# Fresh-model factories (``train`` mutates, so we rebuild per fit).
MODEL_FACTORIES = {
    "QP": lambda: QP(print_global=False),
    "RBF": lambda: RBF(print_global=False),
    "KRG": lambda: KRG(print_global=False),
}
N_TRAIN = {"d1": 80, "d2": 50}  # D1 is 4-D, D2 is 3-D
N_TEST = 300
CV_FOLDS = 5
LC_SIZES = [8, 16, 32, 64, 128, 256, 512]  # learning-curve training sizes (doublings)


def _normalize(x: np.ndarray, xlimits: np.ndarray) -> np.ndarray:
    lo, hi = xlimits[:, 0], xlimits[:, 1]
    return (x - lo) / (hi - lo)


def _sample(disc_key: str, n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """LHS DoE for one discipline, returned as (normalized inputs, true outputs)."""
    disc = DISCIPLINES[disc_key]
    x_raw = LHS(xlimits=disc["xlimits"], criterion="ese", seed=seed)(n)
    return _normalize(x_raw, disc["xlimits"]), disc["func"](x_raw)


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float]:
    """Return (R2, RMSE, MAE)."""
    yt, yp = y_true.ravel(), y_pred.ravel()
    err = yp - yt
    rmse = float(np.sqrt(np.mean(err**2)))
    mae = float(np.mean(np.abs(err)))
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((yt - yt.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot
    return r2, rmse, mae


def _fit(factory, x: np.ndarray, y: np.ndarray):
    model = factory()
    model.set_training_values(x, y)
    model.train()
    return model


def _kfold_r2(factory, x: np.ndarray, y: np.ndarray, k: int, seed: int = 0) -> tuple[float, float]:
    idx = np.random.default_rng(seed).permutation(len(x))
    folds = np.array_split(idx, k)
    scores = []
    for i in range(k):
        test_idx = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(k) if j != i])
        model = _fit(factory, x[train_idx], y[train_idx])
        scores.append(_metrics(y[test_idx], model.predict_values(x[test_idx]))[0])
    return float(np.mean(scores)), float(np.std(scores))


def _learning_curve(disc_key: str, sizes: list[int]) -> dict[str, list[float]]:
    """Held-out test RMSE vs. training size for RBF and KRG on one discipline."""
    x_test, y_test = _sample(disc_key, N_TEST, seed=7)
    curves: dict[str, list[float]] = {"RBF": [], "KRG": []}
    for n in sizes:
        x_tr, y_tr = _sample(disc_key, n, seed=1)
        for name in curves:
            model = _fit(MODEL_FACTORIES[name], x_tr, y_tr)
            curves[name].append(_metrics(y_test, model.predict_values(x_test))[1])  # RMSE
    return curves


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Train the dense surrogates and score them on an independent test set.
    results: dict[str, dict[str, dict]] = {}
    for disc_key in ("d1", "d2"):
        x_tr, y_tr = _sample(disc_key, N_TRAIN[disc_key], seed=1)
        x_te, y_te = _sample(disc_key, N_TEST, seed=7)
        results[disc_key] = {}
        for name, factory in MODEL_FACTORIES.items():
            model = _fit(factory, x_tr, y_tr)
            y_pred = model.predict_values(x_te)
            r2, rmse, mae = _metrics(y_te, y_pred)
            cv_mean, cv_std = _kfold_r2(factory, x_tr, y_tr, CV_FOLDS)
            results[disc_key][name] = {
                "r2": r2, "rmse": rmse, "mae": mae,
                "cv_mean": cv_mean, "cv_std": cv_std,
                "y_true": y_te, "y_pred": y_pred,
            }

    curves = _learning_curve("d2", LC_SIZES)
    _plot_parity(results, OUTPUT_DIR / "surrogate_parity.png")
    _plot_cv(results, LC_SIZES, curves, OUTPUT_DIR / "surrogate_cv.png")
    _write_metrics(results, LC_SIZES, curves, OUTPUT_DIR / "surrogate_metrics.md")

    # Console summary.
    print("Held-out test R^2 (dense DoE):")
    for disc_key in ("d1", "d2"):
        row = "  ".join(f"{m}={results[disc_key][m]['r2']:.4f}" for m in MODEL_FACTORIES)
        print(f"  {disc_key}: {row}")
    krg_dense, krg_sparse = curves["KRG"][-1], curves["KRG"][0]
    print(f"\nD2 learning curve (KRG): RMSE {krg_sparse:.2e} at n={LC_SIZES[0]} "
          f"-> {krg_dense:.2e} at n={LC_SIZES[-1]} (more data -> less error)")
    print(f"\nWrote outputs to {OUTPUT_DIR.relative_to(REPO_ROOT)}")

    # --- acceptance asserts -------------------------------------------------- #
    assert results["d1"]["QP"]["r2"] > 0.999, "QP must nail the quadratic D1"
    assert results["d2"]["KRG"]["r2"] > 0.99, results["d2"]["KRG"]["r2"]
    assert results["d2"]["RBF"]["r2"] > 0.99, results["d2"]["RBF"]["r2"]
    assert krg_sparse > krg_dense, (krg_sparse, krg_dense)  # sparse DoE -> larger error
    return 0


def _plot_parity(results: dict, outfile: Path) -> None:
    models = list(MODEL_FACTORIES)
    fig, axes = plt.subplots(2, len(models), figsize=(4 * len(models), 8))
    for r, disc_key in enumerate(("d1", "d2")):
        out = DISCIPLINES[disc_key]["output_name"]
        for c, name in enumerate(models):
            ax = axes[r, c]
            data = results[disc_key][name]
            yt, yp = data["y_true"].ravel(), data["y_pred"].ravel()
            lo, hi = float(min(yt.min(), yp.min())), float(max(yt.max(), yp.max()))
            ax.plot([lo, hi], [lo, hi], "k--", lw=1)
            ax.scatter(yt, yp, s=12, color="C0", alpha=0.6)
            ax.set_title(f"{name} on {disc_key}  ($R^2$={data['r2']:.4f})")
            ax.set_xlabel(f"true {out}")
            ax.set_ylabel(f"predicted {out}")
            ax.grid(True, alpha=0.3)
    fig.suptitle("Parity plots: predicted vs. true on the held-out test set")
    fig.tight_layout()
    fig.savefig(outfile, dpi=130)
    plt.close(fig)


def _plot_cv(results: dict, sizes: list[int], curves: dict, outfile: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))

    models = list(MODEL_FACTORIES)
    xpos = np.arange(len(models))
    width = 0.38
    for i, disc_key in enumerate(("d1", "d2")):
        means = [results[disc_key][m]["cv_mean"] for m in models]
        errs = [results[disc_key][m]["cv_std"] for m in models]
        ax1.bar(xpos + (i - 0.5) * width, means, width, yerr=errs, capsize=3, label=disc_key)
    ax1.set_xticks(xpos)
    ax1.set_xticklabels(models)
    ax1.set_ylim(0.9, 1.001)
    ax1.set_ylabel(r"$k$-fold CV $R^2$")
    ax1.set_title(f"{CV_FOLDS}-fold cross-validation")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    for name, vals in curves.items():
        ax2.loglog(sizes, vals, "o-", label=name)
    ax2.set_xlabel("training-set size $n$")
    ax2.set_ylabel(r"held-out test RMSE")
    ax2.set_title("Learning curve on D2 (sqrt term): more data -> less error")
    ax2.legend()
    ax2.grid(True, alpha=0.3, which="both")

    fig.tight_layout()
    fig.savefig(outfile, dpi=130)
    plt.close(fig)


def _write_metrics(results: dict, sizes: list[int], curves: dict, outfile: Path) -> None:
    lines = ["# Lesson 4 - Surrogate goodness-of-fit (generated)", ""]
    lines.append(f"Held-out test set of {N_TEST} independent LHS points; inputs scaled to [0, 1].")
    lines.append("")
    lines.append("| Discipline | Model | R^2 | RMSE | MAE | 5-fold CV R^2 |")
    lines.append("|---|---|---|---|---|---|")
    for disc_key in ("d1", "d2"):
        for name in MODEL_FACTORIES:
            d = results[disc_key][name]
            lines.append(
                f"| {disc_key} | {name} | {d['r2']:.5f} | {d['rmse']:.3e} | "
                f"{d['mae']:.3e} | {d['cv_mean']:.5f} +/- {d['cv_std']:.1e} |"
            )
    lines.append("")
    lines.append("## Learning curve on D2 (held-out RMSE vs. training size)")
    lines.append("")
    lines.append("| n | " + " | ".join(curves) + " |")
    lines.append("|---|" + "|".join(["---"] * len(curves)) + "|")
    for i, n in enumerate(sizes):
        lines.append(f"| {n} | " + " | ".join(f"{curves[m][i]:.3e}" for m in curves) + " |")
    lines.append("")
    lines.append(
        "- QP fits the quadratic D1 to machine precision but keeps a small bias on "
        "D2's `sqrt` term.\n- RBF needs its inputs normalized (unscaled it fails); "
        "scaled, RBF and KRG interpolate\n  both disciplines well.\n- Test RMSE grows "
        "as the training set shrinks (learning curve) - the quantitative face of\n  "
        "\"not enough data.\""
    )
    outfile.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
