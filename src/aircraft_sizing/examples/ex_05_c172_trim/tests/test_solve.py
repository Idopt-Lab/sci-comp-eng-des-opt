"""Solver tests: exact backends agree with each other and the reference; GA approximates.

Exact-gradient backends should take the *identical* SLSQP path (same iteration and
evaluation counts) because they supply the same value and gradient to machine
precision; finite-difference drivers land on the same optimum to looser tolerance;
the gradient-free GA lands nearby and is seed-reproducible.
"""

from __future__ import annotations

import unittest

import numpy as np

from aircraft_sizing.examples.ex_05_c172_trim.methods import backends as B
from aircraft_sizing.examples.ex_05_c172_trim.methods import solve as S
from aircraft_sizing.examples.ex_05_c172_trim.methods.backend_jax import JaxBackend
from aircraft_sizing.examples.ex_05_c172_trim.methods.backend_numpy import NumpyBackend
from aircraft_sizing.examples.ex_05_c172_trim.methods.problem import TrimProblem

_NODE_COUNTS = (1, 4)
_OPTIONAL_EXACT = ("casadi", "csdl", "warp")


class ExactBackendSolveTests(unittest.TestCase):
    def test_exact_backends_match_reference_and_each_other(self) -> None:
        for n in _NODE_COUNTS:
            problem = TrimProblem.with_sweep(n)
            reference = S.reference_solution(problem)

            baseline = None
            for name in ("jax", *_OPTIONAL_EXACT):
                if not B.is_available(name):
                    continue
                backend, setup = S.build_backend(name, problem)
                sol = S.solve_slsqp(problem, backend, deriv="exact", reference=reference,
                                    setup_seconds=setup)
                with self.subTest(backend=name, n=n):
                    self.assertLess(sol.error_vs_reference, 1e-9)
                    self.assertTrue(sol.within_bounds(problem))
                if baseline is None:
                    baseline = sol
                else:
                    with self.subTest(backend=name, n=n, check="identical-counts"):
                        self.assertEqual(sol.n_iter, baseline.n_iter)
                        self.assertEqual(sol.n_model_evals, baseline.n_model_evals)
                        self.assertEqual(sol.n_grad_evals, baseline.n_grad_evals)


class FiniteDifferenceSolveTests(unittest.TestCase):
    def test_fd_central_reaches_reference(self) -> None:
        problem = TrimProblem.with_sweep(4)
        reference = S.reference_solution(problem)
        sol = S.solve_slsqp(problem, NumpyBackend(problem), deriv="fd-central", reference=reference)
        self.assertLess(sol.error_vs_reference, 1e-9)

    def test_fd_forward_reaches_reference(self) -> None:
        problem = TrimProblem.with_sweep(4)
        reference = S.reference_solution(problem)
        sol = S.solve_slsqp(problem, NumpyBackend(problem), deriv="fd-forward", reference=reference)
        self.assertLess(sol.error_vs_reference, 1e-6)


class GeneticAlgorithmTests(unittest.TestCase):
    def test_ga_approximates_reference_and_is_reproducible(self) -> None:
        problem = TrimProblem.with_sweep(1)
        reference = S.reference_solution(problem)
        kwargs = dict(pop_size=60, seed=1, max_evals=15_000, reference=reference)

        first = S.solve_ga(problem, **kwargs)
        second = S.solve_ga(problem, **kwargs)

        # error_vs_reference is already in scaled space (radians and ~2000 RPM compare fairly).
        self.assertLess(first.error_vs_reference, 5e-3)
        self.assertTrue(first.within_bounds(problem))
        self.assertTrue(np.allclose(first.x_physical, second.x_physical))


if __name__ == "__main__":
    unittest.main()
