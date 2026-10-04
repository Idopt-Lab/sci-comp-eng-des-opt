"""A counting, scaling objective wrapper that any backend can drive SLSQP through.

:class:`CountingObjective` is what ``scipy.optimize.minimize`` actually calls.  It
   * maps the optimizer's scaled ``z`` to physical ``x`` and back-scales the gradient,
   * supplies the gradient either from the backend's analytic derivative
     (``deriv="exact"``) or by finite-differencing the backend's *value*
     (``"fd-forward"`` / ``"fd-central"``), and
   * tallies function / gradient evaluations and the wall-clock spent inside the
     model, which the study reports.

Finite differences are taken in the scaled space (``z in [0, 1]``), so one step size
is appropriate for every design variable.  The forward step uses ``h = sqrt(eps)``
(balancing O(h) truncation against roundoff) and the central step ``h = eps**(1/3)``
(balancing O(h^2) truncation against roundoff).
"""

from __future__ import annotations

import time

import numpy as np

from .problem import TrimProblem

__all__ = ["CountingObjective", "DERIV_METHODS"]

DERIV_METHODS = ("exact", "fd-forward", "fd-central")

_EPS = float(np.finfo(float).eps)
_H_FORWARD = _EPS**0.5  # ~1.5e-8
_H_CENTRAL = _EPS ** (1.0 / 3.0)  # ~6.1e-6


class CountingObjective:
    """Scaled, instrumented objective + gradient for one backend and deriv method."""

    def __init__(self, problem: TrimProblem, backend, deriv: str = "exact") -> None:
        if deriv not in DERIV_METHODS:
            raise ValueError(f"deriv must be one of {DERIV_METHODS}, got {deriv!r}")
        if deriv == "exact" and not getattr(backend, "has_exact_grad", False):
            raise ValueError(f"backend {backend.name!r} has no analytic gradient")
        self.problem = problem
        self.backend = backend
        self.deriv = deriv
        self._dx_dz = problem.dx_dz()
        self.reset()

    def reset(self) -> None:
        self.n_value = 0
        self.n_grad = 0
        self.model_seconds = 0.0

    # --- raw value in model (physical) space, timed and counted ----------- #
    def _value_physical(self, x_physical: np.ndarray) -> float:
        start = time.perf_counter()
        val = self.backend.value(x_physical)
        self.model_seconds += time.perf_counter() - start
        self.n_value += 1
        return val

    # --- scipy entry points (scaled z space) ------------------------------ #
    def objective(self, z: np.ndarray) -> float:
        return self._value_physical(self.problem.unscale(np.asarray(z, dtype=float)))

    def gradient(self, z: np.ndarray) -> np.ndarray:
        z = np.asarray(z, dtype=float)
        if self.deriv == "exact":
            start = time.perf_counter()
            grad_x = self.backend.grad(self.problem.unscale(z))
            self.model_seconds += time.perf_counter() - start
            self.n_grad += 1
            return np.asarray(grad_x) * self._dx_dz  # chain rule d/dz = d/dx * dx/dz
        if self.deriv == "fd-forward":
            return self._fd_forward(z)
        return self._fd_central(z)

    # --- finite differences in z space ------------------------------------ #
    def _fd_forward(self, z: np.ndarray) -> np.ndarray:
        base = self.objective(z)  # reused for every column
        grad = np.empty_like(z)
        for i in range(z.size):
            step = np.zeros_like(z)
            step[i] = _H_FORWARD
            grad[i] = (self.objective(z + step) - base) / _H_FORWARD
        self.n_grad += 1
        return grad

    def _fd_central(self, z: np.ndarray) -> np.ndarray:
        grad = np.empty_like(z)
        for i in range(z.size):
            step = np.zeros_like(z)
            step[i] = _H_CENTRAL
            grad[i] = (self.objective(z + step) - self.objective(z - step)) / (2.0 * _H_CENTRAL)
        self.n_grad += 1
        return grad
