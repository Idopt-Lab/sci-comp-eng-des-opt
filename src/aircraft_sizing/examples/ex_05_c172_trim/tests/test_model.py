"""Reference-physics tests: atmosphere, the documented trim, sign conventions, parity.

These guard the physics port independently of any optimization backend.  The trim
numbers are the acceptance criterion for the whole port.
"""

from __future__ import annotations

import unittest

import numpy as np

from aircraft_sizing.examples.ex_05_c172_trim.methods import model
from aircraft_sizing.examples.ex_05_c172_trim.methods.aircraft import (
    CD_ALPHA,
    CL_ALPHA,
    CL_DELTA_E,
    C172Data,
    standard_atmosphere,
)
from aircraft_sizing.examples.ex_05_c172_trim.methods.problem import TrimProblem
from aircraft_sizing.examples.ex_05_c172_trim.methods.solve import reference_solution

# Documented corrected trim at M = 0.1 (see docs/ex_05_c172_trim.md).
TARGET_THETA_DEG = 9.505408
TARGET_DELTA_E_DEG = -9.581727
TARGET_OMEGA_RPM = 1823.38


class AtmosphereTests(unittest.TestCase):
    def test_ambiance_matches_source_constants(self) -> None:
        rho, a, g = standard_atmosphere(1000.0)
        # The original source hard-coded these 1000 m ISA values.
        self.assertAlmostEqual(rho, 1.1116589850558272, delta=1e-5)
        self.assertAlmostEqual(a, 336.43470050484996, delta=1e-3)
        self.assertAlmostEqual(g, 9.803565306802405, delta=1e-9)


class SignConventionTests(unittest.TestCase):
    def test_alpha_zero_gives_minus_drag_minus_lift(self) -> None:
        data = C172Data()
        mach = np.array([0.1])
        zero = np.array([0.0])
        fx, fz, _ = model.aero_forces(data, mach, zero, zero, np)

        speed = mach * data.speed_of_sound
        q_bar = 0.5 * data.rho * speed**2
        drag = q_bar * data.wing_area * CD_ALPHA[-1]  # CD(0 deg) = constant term
        lift = q_bar * data.wing_area * (CL_ALPHA[-1] + CL_DELTA_E[-1])  # CL(0, 0)
        self.assertTrue(np.allclose(fx, -drag))
        self.assertTrue(np.allclose(fz, -lift))


class TrimNumberTests(unittest.TestCase):
    """The acceptance gate: the corrected model reproduces the documented trim."""

    def test_m01_single_node_trim(self) -> None:
        problem = TrimProblem(data=C172Data(), mach=np.array([0.1]))
        x = reference_solution(problem)
        theta, delta_e, omega = problem.unpack(x)
        self.assertAlmostEqual(np.rad2deg(theta[0]), TARGET_THETA_DEG, places=4)
        self.assertAlmostEqual(np.rad2deg(delta_e[0]), TARGET_DELTA_E_DEG, places=4)
        self.assertAlmostEqual(omega[0], TARGET_OMEGA_RPM, delta=0.1)

    def test_residual_at_solution_is_zero(self) -> None:
        problem = TrimProblem(data=C172Data(), mach=np.array([0.1]))
        x = reference_solution(problem)
        theta, delta_e, omega = problem.unpack(x)
        residual = model.trim_residual(problem.data, problem.mach, theta, delta_e, omega, np)
        self.assertLess(float(np.linalg.norm(residual)), 1e-8)


class NumpyJaxParityTests(unittest.TestCase):
    def test_objective_matches_to_1e14(self) -> None:
        import jax.numpy as jnp

        problem = TrimProblem.with_sweep(3)
        data = problem.data
        rng = np.random.default_rng(0)
        x = problem.unscale(rng.uniform(0.1, 0.9, size=problem.n_dv))
        theta, delta_e, omega = problem.unpack(x)

        f_np = model.trim_objective(data, problem.mach, theta, delta_e, omega, np)
        f_jax = model.trim_objective(
            data, jnp.asarray(problem.mach), jnp.asarray(theta),
            jnp.asarray(delta_e), jnp.asarray(omega), jnp,
        )
        self.assertAlmostEqual(float(f_np), float(f_jax), delta=1e-14 * (1 + abs(float(f_np))))


class ProblemScalingTests(unittest.TestCase):
    def test_scale_unscale_roundtrip(self) -> None:
        problem = TrimProblem.with_sweep(4)
        z = np.linspace(0.05, 0.95, problem.n_dv)
        self.assertTrue(np.allclose(problem.scale(problem.unscale(z)), z))

    def test_z0_inside_bounds(self) -> None:
        problem = TrimProblem.with_sweep(2)
        z0 = problem.z0()
        self.assertTrue(np.all(z0 >= 0.0) and np.all(z0 <= 1.0))


if __name__ == "__main__":
    unittest.main()
