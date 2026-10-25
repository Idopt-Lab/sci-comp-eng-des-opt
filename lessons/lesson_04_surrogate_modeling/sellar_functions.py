"""The Sellar disciplines as plain-numpy functions - the "truth" we sample.

Lesson 4 replaces the *analytic* Sellar disciplines with surrogates fitted to a
Design of Experiments (DoE). To do that we need the disciplines as cheap,
vectorized functions we can evaluate at an arbitrary batch of sample points: this
module is the ground-truth "black box" that the DoE calls and that we later score
the surrogates against.

The equations are identical to the Lesson 2 / Lesson 3 OpenMDAO components
(``lessons/lesson_02_iterative_methods/sellar/sellar_mda.py``):

    Discipline 1:  y1 = z1**2 + z2 + x - 0.2 * y2      inputs [z1, z2, x,  y2]
    Discipline 2:  y2 = sqrt(y1) + z1 + z2             inputs [z1, z2, y1]

Each function takes a 2-D array ``X`` of shape ``(n_samples, n_inputs)`` (the
column order given by ``*_INPUT_NAMES``) and returns a ``(n_samples, 1)`` column,
which is exactly the layout SMT's ``set_training_values`` expects.

The ``*_XLIMITS`` arrays are the sampling box for each discipline. The coupling
inputs (``y2`` for D1, ``y1`` for D2) are sampled over generous ranges that cover
the MDA trajectory from the Lesson-3 start point ``z=[5, 2], x=1`` (where
``y1~=25.6, y2~=12.1``) down to the optimum (``y1=3.16, y2=3.76``). A surrogate is
only trustworthy *inside* the box it was trained on - that "validity domain" is a
core lesson, so we deliberately keep the box explicit here.

Run from the repository root (with the ``eng-des-opt-course`` environment active)::

    python lessons/lesson_04_surrogate_modeling/sellar_functions.py
"""

from __future__ import annotations

import numpy as np

# --- Discipline 1:  y1 = z1^2 + z2 + x - 0.2 y2 --------------------------- #
D1_INPUT_NAMES = ["z1", "z2", "x", "y2"]
D1_XLIMITS = np.array(
    [
        [-10.0, 10.0],  # z1
        [0.0, 10.0],    # z2
        [0.0, 10.0],    # x
        [0.0, 24.0],    # y2  (bounded above by the g2 constraint y2 <= 24)
    ]
)

# --- Discipline 2:  y2 = sqrt(y1) + z1 + z2 ------------------------------- #
D2_INPUT_NAMES = ["z1", "z2", "y1"]
D2_XLIMITS = np.array(
    [
        [-10.0, 10.0],  # z1
        [0.0, 10.0],    # z2
        [0.0, 40.0],    # y1  (>= 0 for the sqrt; 40 covers the start point y1~=25.6)
    ]
)


def d1(X: np.ndarray) -> np.ndarray:
    """Discipline 1 truth. ``X`` columns ``[z1, z2, x, y2]`` -> ``y1`` ``(n, 1)``."""
    X = np.atleast_2d(np.asarray(X, dtype=float))
    z1, z2, x, y2 = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
    y1 = z1**2 + z2 + x - 0.2 * y2
    return y1.reshape(-1, 1)


def d2(X: np.ndarray) -> np.ndarray:
    """Discipline 2 truth. ``X`` columns ``[z1, z2, y1]`` -> ``y2`` ``(n, 1)``."""
    X = np.atleast_2d(np.asarray(X, dtype=float))
    z1, z2, y1 = X[:, 0], X[:, 1], X[:, 2]
    y1 = np.where(y1 < 0.0, -y1, y1)  # guard the sqrt, like the OpenMDAO component
    y2 = np.sqrt(y1) + z1 + z2
    return y2.reshape(-1, 1)


# Registry so the DoE / fitting scripts can loop over both disciplines uniformly.
DISCIPLINES = {
    "d1": {"func": d1, "xlimits": D1_XLIMITS, "input_names": D1_INPUT_NAMES, "output_name": "y1"},
    "d2": {"func": d2, "xlimits": D2_XLIMITS, "input_names": D2_INPUT_NAMES, "output_name": "y2"},
}


def main() -> None:
    # Sanity check at the Lesson-3 start point z=[5, 2], x=1, with the coupling
    # held at its converged value, so the numbers match the Lesson-2 MDA.
    y1 = d1(np.array([[5.0, 2.0, 1.0, 12.0585]]))[0, 0]
    y2 = d2(np.array([[5.0, 2.0, 25.5883]]))[0, 0]
    print("Sellar disciplines as functions (near the z=[5, 2], x=1 fixed point):")
    print(f"  d1([5, 2, 1, 12.0585]) -> y1 = {y1:.4f}  (expect ~25.5883)")
    print(f"  d2([5, 2, 25.5883])    -> y2 = {y2:.4f}  (expect ~12.0585)")


if __name__ == "__main__":
    main()
