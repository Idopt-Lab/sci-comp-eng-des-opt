"""NumPy backend: objective values only (no analytic gradient).

This is the reference evaluator and the population evaluator the GA uses.  It has
no autodiff, so gradient-based solvers pair it with finite differences in
:class:`~.objective.CountingObjective`.
"""

from __future__ import annotations

import numpy as np

from . import model
from .problem import TrimProblem

__all__ = ["NumpyBackend"]


class NumpyBackend:
    """Evaluate the trim objective with NumPy."""

    name = "numpy"
    device = "cpu"
    has_exact_grad = False

    def __init__(self, problem: TrimProblem) -> None:
        self.problem = problem
        self._mach = np.asarray(problem.mach, dtype=float)

    def value(self, x_physical: np.ndarray) -> float:
        theta, delta_e, omega = self.problem.unpack(np.asarray(x_physical, dtype=float))
        return float(
            model.trim_objective(self.problem.data, self._mach, theta, delta_e, omega, np)
        )

    def value_batch(self, x_physical_batch: np.ndarray) -> np.ndarray:
        """Objective for a population of designs, shape ``(P, 3N)`` -> ``(P,)``."""
        x = np.asarray(x_physical_batch, dtype=float)
        theta, delta_e, omega = self.problem.unpack(x)  # each (P, N)
        return np.asarray(
            model.trim_objective(self.problem.data, self._mach, theta, delta_e, omega, np)
        )
