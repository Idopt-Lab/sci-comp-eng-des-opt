"""JAX backend: JIT-compiled objective value and reverse-mode gradient.

Both the primal and its gradient are traced once (per problem) and cached, so
repeated SLSQP iterations reuse the compiled kernels.  JAX runs on CPU on this
native-Windows machine (there are no native-Windows GPU wheels); the device is
recorded for the study stamp regardless.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from . import model
from .problem import TrimProblem

__all__ = ["JaxBackend"]


class JaxBackend:
    """Evaluate the trim objective and its gradient with JAX autodiff."""

    name = "jax"
    has_exact_grad = True

    def __init__(self, problem: TrimProblem) -> None:
        self.problem = problem
        self.device = jax.default_backend()
        mach = jnp.asarray(problem.mach)
        data = problem.data
        unpack = problem.unpack

        def objective(x):
            theta, delta_e, omega = unpack(x)
            return model.trim_objective(data, mach, theta, delta_e, omega, jnp)

        self._value = jax.jit(objective)
        self._grad = jax.jit(jax.grad(objective))

    def value(self, x_physical: np.ndarray) -> float:
        return float(self._value(jnp.asarray(x_physical, dtype=float)))

    def grad(self, x_physical: np.ndarray) -> np.ndarray:
        return np.asarray(self._grad(jnp.asarray(x_physical, dtype=float)))
