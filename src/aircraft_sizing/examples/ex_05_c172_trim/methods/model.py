"""Reference C172 trim physics, generic over the array module ``xp``.

Every function takes ``xp`` -- either :mod:`numpy` or :mod:`jax.numpy` -- and uses
only elementwise algebra, so the *same* code produces the numpy reference values
and the JAX-autodiffed gradients.  The CasADi / CSDL / Warp backends re-express
this identical math in their own DSLs; the numbers must match this module to
machine precision.

Shapes: ``theta``, ``delta_e``, ``omega`` broadcast against ``mach``.  For a single
design with ``N`` trim nodes pass shape ``(N,)``; for a GA population of ``P``
members pass ``(P, N)`` and ``mach`` of shape ``(N,)``.  The residual gets a
trailing size-6 axis, and the objective reduces over the last two axes (nodes and
the six accelerations), leaving any leading population axis intact.

Two corrections relative to the ``mdo-jax-examples`` source (see the example docs):

* **Thrust uses R**4** (``T = Ct rho n^2 D^4`` with ``D = 2R``, ``n = omega/2pi``,
  i.e. ``T = (2/pi)^2 rho omega_rad^2 R^4 Ct``).  The source used ``R**2``.
* **Body-axis forces use the standard wind->body rotation**
  (``Fx = -D cos a + L sin a``, ``Fz = -D sin a - L cos a``).  The source
  transposed the direction-cosine matrix, which put the model at a different
  operating point.
"""

from __future__ import annotations

import math

# The reference values are float64; make any JAX tracing of this module match by
# enabling 64-bit mode as a side effect of import (JAX otherwise defaults to 32-bit).
try:  # pragma: no cover - jax is a core dependency, but keep numpy-only use working
    import jax

    jax.config.update("jax_enable_x64", True)
except ImportError:  # pragma: no cover
    pass

from .aircraft import (
    CD_ALPHA,
    CL_ALPHA,
    CL_DELTA_E,
    CM_ALPHA,
    CM_DELTA_E,
    CT_J,
    C172Data,
)

__all__ = [
    "aero_forces",
    "thrust",
    "inertial_forces",
    "eom_6dof",
    "trim_states",
    "trim_residual",
    "trim_objective",
    "residual_scale",
]

_RAD2DEG = 180.0 / math.pi
_TWO_PI_OVER_60 = 2.0 * math.pi / 60.0
_TWO_OVER_PI_SQ = (2.0 / math.pi) ** 2


def _horner(coeffs: tuple[float, ...], x):
    """Evaluate a polynomial (highest power first) at ``x`` via Horner's rule."""
    acc = coeffs[0]
    for c in coeffs[1:]:
        acc = acc * x + c
    return acc


def aero_forces(data: C172Data, mach, alpha, delta_e, xp):
    """Body-axis aerodynamic ``(Fx, Fz, pitching_moment)``; side force/roll/yaw are 0.

    ``alpha`` and ``delta_e`` are in radians.
    """
    speed = mach * data.speed_of_sound
    alpha_deg = alpha * _RAD2DEG
    delta_e_deg = delta_e * _RAD2DEG

    c_drag = _horner(CD_ALPHA, alpha_deg)
    c_lift = _horner(CL_ALPHA, alpha_deg) + _horner(CL_DELTA_E, delta_e_deg)
    c_moment = _horner(CM_ALPHA, alpha_deg) + _horner(CM_DELTA_E, delta_e_deg)

    q_bar = 0.5 * data.rho * speed**2
    lift = q_bar * data.wing_area * c_lift
    drag = q_bar * data.wing_area * c_drag
    pitch_moment = q_bar * data.wing_area * data.wing_chord * c_moment

    cos_a = xp.cos(alpha)
    sin_a = xp.sin(alpha)
    # Standard wind->body rotation about the (zero-sideslip) y-axis.
    force_x = -drag * cos_a + lift * sin_a
    force_z = -drag * sin_a - lift * cos_a
    return force_x, force_z, pitch_moment


def thrust(data: C172Data, mach, omega_rpm, xp):
    """Propeller thrust (N), body-x only, zero moment (thrust origin == ref)."""
    speed = mach * data.speed_of_sound
    omega_rad = omega_rpm * _TWO_PI_OVER_60
    advance_ratio = math.pi * speed / (omega_rad * data.prop_radius)
    c_thrust = _horner(CT_J, advance_ratio)
    return _TWO_OVER_PI_SQ * data.rho * omega_rad**2 * data.prop_radius**4 * c_thrust


def inertial_forces(data: C172Data, theta, xp):
    """Gravity resolved into body axes at bank angle phi = 0: ``(Fx, Fz)``."""
    weight = data.mass * data.gravity
    force_x = -weight * xp.sin(theta)
    force_z = weight * xp.cos(theta)
    return force_x, force_z


def eom_6dof(data: C172Data, u, v, w, p, q, r, force, moment, xp):
    """Flat-earth rigid-body accelerations ``[u_dot, v_dot, w_dot, p_dot, q_dot, r_dot]``.

    ``force`` and ``moment`` are ``(Fx, Fy, Fz)`` / ``(L, M, N)`` tuples.  Stacked on
    a trailing size-6 axis.
    """
    f_x, f_y, f_z = force
    l_mom, m_mom, n_mom = moment
    mass = data.mass
    i_xx, i_yy, i_zz, i_xz = data.I_xx, data.I_yy, data.I_zz, data.I_xz
    denom = i_xx * i_zz - i_xz**2

    du = f_x / mass + r * v - q * w
    dv = f_y / mass - r * u + p * w
    dw = f_z / mass + q * u - p * v
    dp = (
        l_mom * i_zz + n_mom * i_xz
        - q * r * (i_zz**2 - i_zz * i_yy + i_xz**2)
        + p * q * i_xz * (i_xx + i_zz - i_yy)
    ) / denom
    dq = (m_mom + (i_zz - i_xx) * p * r - i_xz * (p**2 - r**2)) / i_yy
    dr = (
        l_mom * i_xz + n_mom * i_xx
        + p * q * (i_xx**2 - i_xx * i_yy + i_xz**2)
        - q * r * i_xz * (i_zz + i_xx - i_yy)
    ) / denom
    return xp.stack([du, dv, dw, dp, dq, dr], axis=-1)


def trim_states(data: C172Data, mach, theta, xp):
    """Body velocities for a symmetric wings-level trim (alpha = theta)."""
    speed = mach * data.speed_of_sound
    zeros = xp.zeros_like(theta * mach)
    u = speed * xp.cos(theta) + zeros
    w = speed * xp.sin(theta) + zeros
    v = zeros
    return u, v, w, zeros, zeros, zeros  # u, v, w, p, q, r


def trim_residual(data: C172Data, mach, theta, delta_e, omega, xp):
    """Six rigid-body accelerations at the trim condition; trailing axis is size 6.

    ``alpha = theta``; bank, sideslip, and all body rates are zero, so the lateral
    accelerations vanish identically and the problem reduces to ``Fx = Fz = M = 0``.
    """
    force_x_a, force_z_a, pitch_moment = aero_forces(data, mach, theta, delta_e, xp)
    thrust_force = thrust(data, mach, omega, xp)
    force_x_i, force_z_i = inertial_forces(data, theta, xp)

    zero = xp.zeros_like(theta * mach)
    force = (force_x_a + thrust_force + force_x_i, zero, force_z_a + force_z_i)
    moment = (zero, pitch_moment + zero, zero)

    u, v, w, p, q, r = trim_states(data, mach, theta, xp)
    return eom_6dof(data, u, v, w, p, q, r, force, moment, xp)


def residual_scale(data: C172Data, xp):
    """Diagonal scale ``(g, g, g, 1, 1, 1)`` that non-dimensionalizes the residual."""
    g = data.gravity
    return xp.asarray([g, g, g, 1.0, 1.0, 1.0])


def trim_objective(data: C172Data, mach, theta, delta_e, omega, xp):
    """Half the squared norm of the scaled residual, reduced over nodes and the 6 axes.

    Returns a scalar for a single design ``(N,)`` or a ``(P,)`` vector for a
    population ``(P, N)``.
    """
    residual = trim_residual(data, mach, theta, delta_e, omega, xp)
    scaled = residual / residual_scale(data, xp)
    return 0.5 * xp.sum(scaled * scaled, axis=(-2, -1))
