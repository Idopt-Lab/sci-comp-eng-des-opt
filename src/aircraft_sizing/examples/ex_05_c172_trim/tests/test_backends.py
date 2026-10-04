"""Cross-backend parity: every backend must agree on value and gradient.

The numpy backend is the value reference; the JAX backend is the gradient reference.
Optional backends (casadi, csdl, warp) are skipped when not installed.
"""

from __future__ import annotations

import unittest

import numpy as np

from aircraft_sizing.examples.ex_05_c172_trim.methods import backends as B
from aircraft_sizing.examples.ex_05_c172_trim.methods.backend_jax import JaxBackend
from aircraft_sizing.examples.ex_05_c172_trim.methods.backend_numpy import NumpyBackend
from aircraft_sizing.examples.ex_05_c172_trim.methods.objective import CountingObjective
from aircraft_sizing.examples.ex_05_c172_trim.methods.problem import TrimProblem

_NODE_COUNTS = (1, 3, 8)
_OPTIONAL = ("casadi", "csdl", "warp")


def _random_point(problem: TrimProblem, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return problem.unscale(rng.uniform(0.2, 0.8, size=problem.n_dv))


class ValueParityTests(unittest.TestCase):
    def test_all_backends_match_numpy_value(self) -> None:
        for n in _NODE_COUNTS:
            problem = TrimProblem.with_sweep(n)
            x = _random_point(problem, seed=n)
            reference = NumpyBackend(problem).value(x)
            for name in ("jax", *_OPTIONAL):
                if not B.is_available(name):
                    continue
                with self.subTest(backend=name, n=n):
                    value = B.make_backend(name, problem).value(x)
                    self.assertLessEqual(abs(value - reference), 1e-12 * (1 + abs(reference)))


class GradientParityTests(unittest.TestCase):
    def test_exact_backends_match_jax_gradient(self) -> None:
        for n in _NODE_COUNTS:
            problem = TrimProblem.with_sweep(n)
            x = _random_point(problem, seed=10 + n)
            reference = JaxBackend(problem).grad(x)
            norm = np.linalg.norm(reference)
            for name in _OPTIONAL:
                if not B.is_available(name):
                    continue
                with self.subTest(backend=name, n=n):
                    grad = B.make_backend(name, problem).grad(x)
                    self.assertLessEqual(np.linalg.norm(grad - reference), 1e-10 * (1 + norm))


class FiniteDifferenceTests(unittest.TestCase):
    def test_exact_matches_central_fd(self) -> None:
        for n in _NODE_COUNTS:
            problem = TrimProblem.with_sweep(n)
            z = problem.scale(_random_point(problem, seed=20 + n))
            exact = CountingObjective(problem, JaxBackend(problem), "exact").gradient(z)
            central = CountingObjective(problem, NumpyBackend(problem), "fd-central").gradient(z)
            with self.subTest(n=n):
                self.assertLessEqual(
                    np.linalg.norm(exact - central), 1e-6 * (1 + np.linalg.norm(exact))
                )


class MachPermutationTests(unittest.TestCase):
    def test_objective_is_permutation_invariant(self) -> None:
        # The objective sums independent per-node residuals, so permuting the nodes
        # (Mach and each DV block together) leaves the total unchanged.
        problem = TrimProblem.with_sweep(5)
        x = _random_point(problem, seed=99)
        theta, delta_e, omega = problem.unpack(x)
        perm = np.array([3, 1, 4, 0, 2])

        base = NumpyBackend(problem).value(x)
        permuted_problem = TrimProblem(data=problem.data, mach=problem.mach[perm])
        permuted_x = permuted_problem.pack(theta[perm], delta_e[perm], omega[perm])
        permuted = NumpyBackend(permuted_problem).value(permuted_x)
        self.assertAlmostEqual(base, permuted, places=12)


if __name__ == "__main__":
    unittest.main()
