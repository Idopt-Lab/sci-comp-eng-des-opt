"""CSDL-alpha backend: build the trim objective in a recorder graph, differentiate it.

The objective is expressed with CSDL-alpha variables inside a :class:`csdl_alpha.Recorder`;
the three design-variable vectors are declared with ``set_as_design_variable`` and the
scalar objective with ``set_as_objective``.  A :class:`csdl_alpha.experimental.PySimulator`
then evaluates the objective and its exact derivatives -- driven directly through
``update_design_variables`` / ``compute_optimization_functions`` /
``compute_optimization_derivatives`` (no modopt).  Runs on CPU.

Design variables are declared in the order ``theta, delta_e, omega`` so the simulator's
flat design-variable vector matches the problem's block layout exactly.
"""

from __future__ import annotations

import math

import numpy as np

from .aircraft import (
    CD_ALPHA,
    CL_ALPHA,
    CL_DELTA_E,
    CM_ALPHA,
    CM_DELTA_E,
    CT_J,
    C172Data,
)
from .problem import TrimProblem

__all__ = ["CsdlBackend"]

_RAD2DEG = 180.0 / math.pi
_TWO_PI_OVER_60 = 2.0 * math.pi / 60.0
_TWO_OVER_PI_SQ = (2.0 / math.pi) ** 2


def _horner(coeffs: tuple[float, ...], x):
    acc = coeffs[0]
    for c in coeffs[1:]:
        acc = acc * x + c
    return acc


class CsdlBackend:
    """Evaluate the trim objective and its exact derivative with CSDL-alpha."""

    name = "csdl"
    device = "cpu"
    has_exact_grad = True

    def __init__(self, problem: TrimProblem) -> None:
        import csdl_alpha as csdl

        self.problem = problem
        data = problem.data
        mach = np.asarray(problem.mach, dtype=float)
        x0 = problem.unscale(problem.z0())
        theta0, delta_e0, omega0 = problem.unpack(x0)

        recorder = csdl.Recorder(inline=True)
        recorder.start()

        theta = csdl.Variable(value=theta0, name="theta")
        delta_e = csdl.Variable(value=delta_e0, name="delta_e")
        omega = csdl.Variable(value=omega0, name="omega")
        theta.set_as_design_variable()
        delta_e.set_as_design_variable()
        omega.set_as_design_variable()

        speed = csdl.Variable(value=mach, name="mach") * data.speed_of_sound
        alpha_deg = theta * _RAD2DEG
        delta_e_deg = delta_e * _RAD2DEG
        c_drag = _horner(CD_ALPHA, alpha_deg)
        c_lift = _horner(CL_ALPHA, alpha_deg) + _horner(CL_DELTA_E, delta_e_deg)
        c_moment = _horner(CM_ALPHA, alpha_deg) + _horner(CM_DELTA_E, delta_e_deg)

        q_bar = 0.5 * data.rho * speed**2
        lift = q_bar * data.wing_area * c_lift
        drag = q_bar * data.wing_area * c_drag
        pitch_moment = q_bar * data.wing_area * data.wing_chord * c_moment

        cos_a, sin_a = csdl.cos(theta), csdl.sin(theta)
        force_x_a = -drag * cos_a + lift * sin_a
        force_z_a = -drag * sin_a - lift * cos_a

        omega_rad = omega * _TWO_PI_OVER_60
        advance_ratio = math.pi * speed / (omega_rad * data.prop_radius)
        thrust = (
            _TWO_OVER_PI_SQ * data.rho * omega_rad**2 * data.prop_radius**4 * _horner(CT_J, advance_ratio)
        )

        weight = data.mass * data.gravity
        force_x = force_x_a + thrust - weight * sin_a
        force_z = force_z_a + weight * cos_a

        g = data.gravity
        u_dot = force_x / data.mass
        w_dot = force_z / data.mass
        q_dot = pitch_moment / data.I_yy
        objective = 0.5 * csdl.sum((u_dot / g) ** 2 + (w_dot / g) ** 2 + q_dot**2)
        objective.set_as_objective()

        recorder.stop()
        self._sim = csdl.experimental.PySimulator(recorder)

    def value(self, x_physical: np.ndarray) -> float:
        self._sim.update_design_variables(np.asarray(x_physical, dtype=float))
        # PySimulator returns {"f": objective, "c": constraints}.
        return float(np.asarray(self._sim.compute_optimization_functions()["f"]).ravel()[0])

    def grad(self, x_physical: np.ndarray) -> np.ndarray:
        self._sim.update_design_variables(np.asarray(x_physical, dtype=float))
        # Refresh the forward pass at this point first: compute_optimization_derivatives
        # differentiates the *last evaluated* state, so without this the gradient would be
        # taken at a stale point when grad() is called without a preceding value().
        self._sim.compute_optimization_functions()
        # {"f", "c", "df": (1, n_dv) objective gradient, "dc"}.
        return np.asarray(self._sim.compute_optimization_derivatives()["df"], dtype=float).ravel()
