r"""Lesson 4 - Step 1: Design of Experiments (how to pick the sample points).

Before we can fit a surrogate we have to *gather data*: evaluate the discipline at
a designed set of input points. The choice of points is the Design of Experiments
(DoE), and it matters - a good design covers the input box evenly with as few runs
as possible. This script generates and compares three classic designs on the first
two inputs of Discipline 1 (``z1``, ``z2``):

* **Full factorial** - a regular grid. Simple and exhaustive, but the number of
  runs is ``levels ** n_inputs`` - it explodes with dimension (the "curse of
  dimensionality").
* **Random (Monte Carlo)** - independent uniform draws. Trivial to generate and
  dimension-agnostic, but points clump together and leave gaps.
* **Latin Hypercube Sampling (LHS)** - stratified: each axis is split into ``N``
  bins and every bin gets exactly one sample, so the design is space-filling and
  its projection onto any axis is uniform. The go-to DoE for surrogate training.

All three come straight from SMT's ``sampling_methods``; the next script
(``02_fit_surrogates.py``) feeds an LHS design through the disciplines and fits
surrogates to the result.

Run from the repository root (with the ``eng-des-opt-course`` environment active)::

    python lessons/lesson_04_surrogate_modeling/01_doe_sampling.py
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
from smt.sampling_methods import LHS, FullFactorial, Random  # noqa: E402

from sellar_functions import D1_XLIMITS  # noqa: E402

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
SEED = 42
N = 49  # a 7x7 grid in 2-D, matched sample count for a fair visual comparison


def _designs_2d(xlimits: np.ndarray) -> dict[str, np.ndarray]:
    """Three DoE designs over a 2-D box, each with (about) N points."""
    full = FullFactorial(xlimits=xlimits)
    rand = Random(xlimits=xlimits, seed=SEED)
    lhs = LHS(xlimits=xlimits, criterion="ese", seed=SEED)
    return {
        "Full factorial": full(N),
        "Random (Monte Carlo)": rand(N),
        "Latin hypercube (LHS)": lhs(N),
    }


def _plot_designs(designs: dict[str, np.ndarray], xlimits: np.ndarray, outfile: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), sharex=True, sharey=True)
    for ax, (name, pts) in zip(axes, designs.items()):
        ax.scatter(pts[:, 0], pts[:, 1], s=22, color="C0", edgecolor="k", linewidth=0.3)
        ax.set_title(f"{name}\n({len(pts)} points)")
        ax.set_xlabel(r"$z_1$")
        ax.set_xlim(xlimits[0])
        ax.set_ylim(xlimits[1])
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel(r"$z_2$")
    fig.suptitle("DoE designs on the Discipline-1 $(z_1, z_2)$ plane")
    fig.tight_layout()
    fig.savefig(outfile, dpi=130)
    plt.close(fig)


def _curse_table() -> list[tuple[int, int, int, int]]:
    """Full-factorial run count = levels ** n_inputs, for a few (levels, dims)."""
    rows = []
    for levels in (3, 5, 10):
        for dims in (2, 4, 6):
            rows.append((levels, dims, dims, levels**dims))
    return rows


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    designs = _designs_2d(D1_XLIMITS[:2])
    _plot_designs(designs, D1_XLIMITS[:2], OUTPUT_DIR / "doe_designs.png")

    # LHS projection property: one sample per bin per axis (uniform marginals).
    lhs_pts = designs["Latin hypercube (LHS)"]
    edges = np.linspace(D1_XLIMITS[0, 0], D1_XLIMITS[0, 1], len(lhs_pts) + 1)
    per_bin = np.histogram(lhs_pts[:, 0], bins=edges)[0]
    print("DoE designs written to outputs/doe_designs.png")
    print(f"LHS stratification check: max samples in any z1 bin = {per_bin.max()} (want 1)")

    print("\nFull-factorial run count = levels ** n_inputs (the curse of dimensionality):")
    print("  levels | dims | runs")
    for levels, dims, _, runs in _curse_table():
        print(f"  {levels:>6} | {dims:>4} | {runs:>8,}")
    print(
        "\n-> LHS needs the same N points regardless of dimension; a full grid at "
        "10 levels\n   over the 4-D Discipline-1 box already costs 10,000 runs."
    )

    # --- acceptance asserts -------------------------------------------------- #
    assert per_bin.max() == 1, per_bin
    assert designs["Random (Monte Carlo)"].shape == (N, 2)
    assert designs["Latin hypercube (LHS)"].shape == (N, 2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
