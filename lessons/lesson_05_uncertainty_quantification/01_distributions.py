r"""Lesson 5 - Step 1: what it means to "add uncertainty" (input distributions).

Up to now every input had a single fixed value. In reality a manufacturing
tolerance, a material property, or an operating condition is only known to within
a *distribution*. "Adding uncertainty" means replacing a fixed number with a random
variable and asking how that randomness flows through to the outputs.

This script introduces the four uncertain parameters of the Sellar problem
(declared in ``input.yaml``) and the two distribution families we use:

* **Normal** ``N(mu, sigma)`` - symmetric, unbounded; the default for additive
  noise and tolerances (``a0, a2, a3``).
* **Uniform** ``U(lo, hi)`` - every value in a range equally likely; the honest
  choice when you only know bounds (``a1``).

It also names the key distinction:

* **Aleatory** uncertainty is irreducible randomness (natural scatter) - modeled
  with a probability distribution; all four parameters here are aleatory.
* **Epistemic** uncertainty is lack of knowledge (could be reduced with more data)
  - often modeled with intervals / probability boxes.

We plot each PDF, overlay a histogram of random draws to show sampling converges to
the PDF, and print the mean / standard deviation of each.

Run from the repository root (with the ``eng-des-opt-course`` environment active)::

    python lessons/lesson_05_uncertainty_quantification/01_distributions.py
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
from scipy import stats  # noqa: E402

from sellar_uncertain import sample_uncertain  # noqa: E402

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"

# (name, scipy frozen distribution, x-range, pretty label) - matches input.yaml.
DISTS = [
    ("a0", stats.norm(1.0, 0.06), (0.75, 1.25), r"$a_0\sim N(1,\,0.06)$"),
    ("a1", stats.uniform(0.9, 0.2), (0.82, 1.18), r"$a_1\sim U(0.9,\,1.1)$"),
    ("a2", stats.norm(1.0, 0.05), (0.78, 1.22), r"$a_2\sim N(1,\,0.05)$"),
    ("a3", stats.norm(1.0, 0.10), (0.55, 1.45), r"$a_3\sim N(1,\,0.10)$"),
]


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    draws = sample_uncertain(50000, rng)

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    print("Uncertain inputs (analytic vs. sampled):")
    for ax, (name, dist, xr, label) in zip(axes.flat, DISTS):
        xs = np.linspace(*xr, 400)
        ax.plot(xs, dist.pdf(xs), "C3", lw=2, label="PDF")
        ax.hist(draws[name], bins=60, density=True, alpha=0.5, color="C0", label="samples")
        ax.axvline(dist.mean(), color="k", ls="--", lw=1, label="mean")
        ax.set_title(label)
        ax.set_xlabel(name)
        ax.set_ylabel("density")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        print(f"  {name}: analytic mean={dist.mean():.4f} std={dist.std():.4f} | "
              f"sampled mean={draws[name].mean():.4f} std={draws[name].std():.4f}")
    fig.suptitle("Lesson 5 - input distributions (all aleatory)")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "input_distributions.png", dpi=130)
    plt.close(fig)
    print(f"\nWrote outputs to {OUTPUT_DIR.relative_to(REPO_ROOT)}")

    # --- acceptance asserts -------------------------------------------------- #
    assert abs(draws["a0"].mean() - 1.0) < 0.01
    assert abs(draws["a1"].std() - 0.2 / np.sqrt(12)) < 0.01  # U(0.9,1.1) std
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
