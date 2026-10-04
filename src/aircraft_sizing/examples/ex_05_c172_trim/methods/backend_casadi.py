"""CasADi backend: a symbolic single-node objective, ``.map``-ed over the sweep.

A CasADi ``SX`` graph for one trim node is built once, vectorized across the ``N``
nodes with :meth:`casadi.Function.map` (CasADi's equivalent of ``vmap``), summed,
and differentiated with :func:`casadi.gradient`.  Everything is ``ca.*`` with
explicit block slices, so the same symbolic function supplies both the value and an
exact (symbolic, not finite-difference) gradient.  Runs on CPU.
"""

from __future__ import annotations

import casadi as ca
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

__all__ = ["CasadiBackend"]

_RAD2DEG = 180.0 / np.pi
_TWO_PI_OVER_60 = 2.0 * np.pi / 60.0
_TWO_OVER_PI_SQ = (2.0 / np.pi) ** 2


def _horner(coeffs: tuple[float, ...], x):
    acc = coeffs[0]
    for c in coeffs[1:]:
        acc = acc * x + c
    return acc


def _node_objective(data: C172Data) -> ca.Function:
    """Symbolic scaled-squared residual for a single trim node."""
    theta = ca.SX.sym("theta")
    delta_e = ca.SX.sym("delta_e")
    omega = ca.SX.sym("omega")
    mach = ca.SX.sym("mach")

    speed = mach * data.speed_of_sound
    alpha_deg = theta * _RAD2DEG
    delta_e_deg = delta_e * _RAD2DEG
    c_drag = _horner(CD_ALPHA, alpha_deg)
    c_lift = _horner(CL_ALPHA, alpha_deg) + _horner(CL_DELTA_E, delta_e_deg)
    c_moment = _horner(CM_ALPHA, alpha_deg) + _horner(CM_DELTA_E, delta_e_deg)

    q_bar = 0.5 * data.rho * speed**2
    lift = q_bar * data.wing_area * c_lift
    drag = q_bar * data.wing_area * c_drag
    pitch_moment = q_bar * data.wing_area * data.wing_chord * c_moment

    cos_a, sin_a = ca.cos(theta), ca.sin(theta)
    force_x_a = -drag * cos_a + lift * sin_a
    force_z_a = -drag * sin_a - lift * cos_a

    omega_rad = omega * _TWO_PI_OVER_60
    advance_ratio = np.pi * speed / (omega_rad * data.prop_radius)
    thrust = _TWO_OVER_PI_SQ * data.rho * omega_rad**2 * data.prop_radius**4 * _horner(
        CT_J, advance_ratio
    )

    weight = data.mass * data.gravity
    force_x = force_x_a + thrust - weight * sin_a
    force_z = force_z_a + weight * cos_a

    # Active residuals only (lateral accelerations vanish at symmetric trim).
    u_dot = force_x / data.mass
    w_dot = force_z / data.mass
    q_dot = pitch_moment / data.I_yy
    g = data.gravity
    objective = 0.5 * ((u_dot / g) ** 2 + (w_dot / g) ** 2 + q_dot**2)
    return ca.Function("node", [theta, delta_e, omega, mach], [objective])


class CasadiBackend:
    """Evaluate the trim objective and symbolic gradient with CasADi."""

    name = "casadi"
    device = "cpu"
    has_exact_grad = True

    def __init__(self, problem: TrimProblem) -> None:
        self.problem = problem
        n = problem.n_nodes
        node_map = _node_objective(problem.data).map(n)

        x = ca.SX.sym("x", 3 * n)
        mach = ca.SX.sym("mach", n)
        per_node = node_map(x[0:n].T, x[n : 2 * n].T, x[2 * n : 3 * n].T, mach.T)
        objective = ca.sum2(per_node)  # sum over the N columns -> scalar
        gradient = ca.gradient(objective, x)

        self._value_fn = ca.Function("value", [x, mach], [objective])
        self._grad_fn = ca.Function("grad", [x, mach], [gradient])
        self._mach = ca.DM(np.asarray(problem.mach, dtype=float))

    def value(self, x_physical: np.ndarray) -> float:
        return float(self._value_fn(ca.DM(np.asarray(x_physical, dtype=float)), self._mach))

    def grad(self, x_physical: np.ndarray) -> np.ndarray:
        g = self._grad_fn(ca.DM(np.asarray(x_physical, dtype=float)), self._mach)
        return np.asarray(g).ravel()
