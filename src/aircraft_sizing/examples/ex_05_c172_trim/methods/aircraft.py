"""Cessna-172 data: the single source of truth shared by every backend.

Geometry, mass properties, and the aerodynamic / propulsive polynomial
coefficients live here as plain Python floats so that the numpy, JAX, CasADi,
CSDL, and Warp backends all consume *identical* numbers.  Polynomials are stored
highest-power-first for Horner evaluation; ``alpha`` and ``delta_e`` are in
**degrees** inside every polynomial (the backends convert from radians at the
interface).

Atmosphere comes from :mod:`ambiance` at 1000 m.  The original
``mdo-jax-examples`` source hard-coded the 1000 m ISA values
(rho = 1.1116589850558272 kg/m^3, a = 336.43470050484996 m/s,
g = 9.803565306802405 m/s^2); ``ambiance`` reproduces them to < 1e-6 relative, so
using it changes the trim by less than the solver tolerance while removing the
magic constants.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from ambiance import Atmosphere

__all__ = [
    "C172Data",
    "standard_atmosphere",
    "CD_ALPHA",
    "CL_ALPHA",
    "CL_DELTA_E",
    "CM_ALPHA",
    "CM_DELTA_E",
    "CT_J",
    "DEFAULT_ALTITUDE_M",
]

DEFAULT_ALTITUDE_M = 1000.0

# --------------------------------------------------------------------------- #
# Aerodynamic / propulsive polynomial coefficients (Horner order, highest first)
# alpha, delta_e in DEGREES; advance ratio J non-dimensional.
# --------------------------------------------------------------------------- #
CD_ALPHA = (0.00033156, 0.00192141, 0.03451242)  # CD(alpha): a^2, a, 1
CL_ALPHA = (0.09460627, 0.16531678)  # CL(alpha): a, 1
CL_DELTA_E = (-4.64968867e-06, 3.95734084e-06, 8.26663557e-03, -1.81731015e-04)  # de^3..de^0
CM_ALPHA = (-0.00088295, -0.01230759, 0.01206867)  # Cm(alpha): a^2, a, 1
CM_DELTA_E = (1.11377133e-05, -9.96895700e-06, -2.03797109e-02, 1.37160466e-04)  # de^3..de^0
CT_J = (-0.1692121, 0.03545196, 0.10446359)  # Ct(J): J^2, J, 1


@lru_cache(maxsize=32)
def standard_atmosphere(altitude_m: float = DEFAULT_ALTITUDE_M) -> tuple[float, float, float]:
    """Return ``(density, speed_of_sound, gravity)`` in SI at a geometric altitude.

    Cached because a sweep re-solves at the same altitude many times.
    """
    atmosphere = Atmosphere(float(altitude_m))
    return (
        float(atmosphere.density[0]),
        float(atmosphere.speed_of_sound[0]),
        float(atmosphere.grav_accel[0]),
    )


_RHO, _A, _G = standard_atmosphere(DEFAULT_ALTITUDE_M)


@dataclass(frozen=True)
class C172Data:
    """Physical constants for the Cessna 172 (SI units)."""

    mass: float = 1043.2616  # kg
    I_xx: float = 1285.3154166  # kg m^2
    I_yy: float = 1824.9309607
    I_zz: float = 2666.89390765
    I_xz: float = 0.0
    wing_area: float = 16.2  # m^2
    wing_chord: float = 1.49352  # m (mean aerodynamic chord)
    wing_span: float = 10.91184  # m
    prop_radius: float = 0.94  # m
    altitude_m: float = DEFAULT_ALTITUDE_M
    # Atmosphere, resolved once from altitude via ambiance.  cg == ref point ==
    # thrust origin == 0, so thrust and gravity contribute no moment.
    rho: float = _RHO  # kg/m^3
    speed_of_sound: float = _A  # m/s
    gravity: float = _G  # m/s^2
