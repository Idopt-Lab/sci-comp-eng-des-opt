"""Warp backend: one GPU thread per trim node, reverse-mode gradient via ``wp.Tape``.

This is the only backend that reaches the GPU on this native-Windows machine: the
``warp-lang`` wheel ships its own CUDA runtime and kernel toolchain, so it runs on
the RTX 4060 with just the NVIDIA driver (no CUDA toolkit).  It falls back to CPU if
no CUDA device is present.

Everything is ``float64`` -- the model runs in double precision for parity with the
other backends, which on a consumer GeForce (FP64 ~ 1/64 FP32) is correct but not
necessarily fast.  Each node's scaled-squared residual is accumulated into a single
output with :func:`warp.atomic_add`; :class:`warp.Tape` then reverse-differentiates
that sum with respect to the three design-variable arrays.

The CUDA context creation and kernel JIT happen on a warm-up call in ``__init__`` so
they are charged to *setup*, never to a timed solve.  The coefficient constants are
derived from :mod:`.aircraft`, which remains the single source of truth.
"""

from __future__ import annotations

import math

import numpy as np
import warp as wp

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

__all__ = ["WarpBackend"]

wp.init()

_f64 = wp.float64

# Scalar constants (float64), sourced from aircraft.py so there is still one truth.
_RAD2DEG = wp.constant(_f64(180.0 / math.pi))
_TWO_PI_OVER_60 = wp.constant(_f64(2.0 * math.pi / 60.0))
_TWO_OVER_PI_SQ = wp.constant(_f64((2.0 / math.pi) ** 2))
_PI = wp.constant(_f64(math.pi))
_HALF = wp.constant(_f64(0.5))

_CD2, _CD1, _CD0 = (wp.constant(_f64(c)) for c in CD_ALPHA)
_CLA1, _CLA0 = (wp.constant(_f64(c)) for c in CL_ALPHA)
_CLE3, _CLE2, _CLE1, _CLE0 = (wp.constant(_f64(c)) for c in CL_DELTA_E)
_CMA2, _CMA1, _CMA0 = (wp.constant(_f64(c)) for c in CM_ALPHA)
_CME3, _CME2, _CME1, _CME0 = (wp.constant(_f64(c)) for c in CM_DELTA_E)
_CT2, _CT1, _CT0 = (wp.constant(_f64(c)) for c in CT_J)


@wp.struct
class _Params:
    mass: _f64
    i_yy: _f64
    wing_area: _f64
    wing_chord: _f64
    prop_radius: _f64
    rho: _f64
    speed_of_sound: _f64
    gravity: _f64


@wp.kernel
def _obj_kernel(
    theta: wp.array(dtype=_f64),
    delta_e: wp.array(dtype=_f64),
    omega: wp.array(dtype=_f64),
    mach: wp.array(dtype=_f64),
    params: _Params,
    out: wp.array(dtype=_f64),
):
    i = wp.tid()
    th = theta[i]
    de = delta_e[i]
    om = omega[i]
    speed = mach[i] * params.speed_of_sound

    alpha_deg = th * _RAD2DEG
    delta_e_deg = de * _RAD2DEG
    c_drag = (_CD2 * alpha_deg + _CD1) * alpha_deg + _CD0
    c_lift = (_CLA1 * alpha_deg + _CLA0) + (
        ((_CLE3 * delta_e_deg + _CLE2) * delta_e_deg + _CLE1) * delta_e_deg + _CLE0
    )
    c_moment = ((_CMA2 * alpha_deg + _CMA1) * alpha_deg + _CMA0) + (
        ((_CME3 * delta_e_deg + _CME2) * delta_e_deg + _CME1) * delta_e_deg + _CME0
    )

    q_bar = _HALF * params.rho * speed * speed
    lift = q_bar * params.wing_area * c_lift
    drag = q_bar * params.wing_area * c_drag
    pitch_moment = q_bar * params.wing_area * params.wing_chord * c_moment

    cos_a = wp.cos(th)
    sin_a = wp.sin(th)
    force_x_a = -drag * cos_a + lift * sin_a
    force_z_a = -drag * sin_a - lift * cos_a

    omega_rad = om * _TWO_PI_OVER_60
    advance_ratio = _PI * speed / (omega_rad * params.prop_radius)
    c_thrust = (_CT2 * advance_ratio + _CT1) * advance_ratio + _CT0
    radius = params.prop_radius
    thrust = _TWO_OVER_PI_SQ * params.rho * omega_rad * omega_rad * radius * radius * radius * radius * c_thrust

    weight = params.mass * params.gravity
    force_x = force_x_a + thrust - weight * sin_a
    force_z = force_z_a + weight * cos_a

    g = params.gravity
    u_dot = force_x / params.mass
    w_dot = force_z / params.mass
    q_dot = pitch_moment / params.i_yy
    obj_i = _HALF * ((u_dot / g) * (u_dot / g) + (w_dot / g) * (w_dot / g) + q_dot * q_dot)
    wp.atomic_add(out, 0, obj_i)


def _make_params(data: C172Data) -> _Params:
    params = _Params()
    params.mass = data.mass
    params.i_yy = data.I_yy
    params.wing_area = data.wing_area
    params.wing_chord = data.wing_chord
    params.prop_radius = data.prop_radius
    params.rho = data.rho
    params.speed_of_sound = data.speed_of_sound
    params.gravity = data.gravity
    return params


class WarpBackend:
    """Evaluate the trim objective and its gradient with Warp on the GPU (or CPU)."""

    name = "warp"
    has_exact_grad = True

    def __init__(self, problem: TrimProblem) -> None:
        self.problem = problem
        self._n = problem.n_nodes
        try:
            self._device = wp.get_preferred_device()
        except Exception:  # pragma: no cover - no CUDA at all
            self._device = wp.get_device("cpu")
        self.device = str(self._device)
        self._params = _make_params(problem.data)
        with wp.ScopedDevice(self._device):
            self._mach = wp.array(np.ascontiguousarray(problem.mach, dtype=np.float64), dtype=_f64)
        # Warm up: pay the CUDA-context + kernel-JIT cost here, not in a timed solve.
        guess = problem.unscale(problem.z0())
        self.value(guess)
        self.grad(guess)

    def _inputs(self, x_physical: np.ndarray):
        theta, delta_e, omega = self.problem.unpack(np.asarray(x_physical, dtype=np.float64))
        with wp.ScopedDevice(self._device):
            theta_a = wp.array(np.ascontiguousarray(theta), dtype=_f64, requires_grad=True)
            delta_e_a = wp.array(np.ascontiguousarray(delta_e), dtype=_f64, requires_grad=True)
            omega_a = wp.array(np.ascontiguousarray(omega), dtype=_f64, requires_grad=True)
            out = wp.zeros(1, dtype=_f64, requires_grad=True)
        return theta_a, delta_e_a, omega_a, out

    def value(self, x_physical: np.ndarray) -> float:
        theta_a, delta_e_a, omega_a, out = self._inputs(x_physical)
        with wp.ScopedDevice(self._device):
            wp.launch(
                _obj_kernel,
                dim=self._n,
                inputs=[theta_a, delta_e_a, omega_a, self._mach, self._params],
                outputs=[out],
            )
        return float(out.numpy()[0])

    def grad(self, x_physical: np.ndarray) -> np.ndarray:
        theta_a, delta_e_a, omega_a, out = self._inputs(x_physical)
        tape = wp.Tape()
        with wp.ScopedDevice(self._device), tape:
            wp.launch(
                _obj_kernel,
                dim=self._n,
                inputs=[theta_a, delta_e_a, omega_a, self._mach, self._params],
                outputs=[out],
            )
        tape.backward(loss=out)
        grad = np.concatenate(
            [theta_a.grad.numpy(), delta_e_a.grad.numpy(), omega_a.grad.numpy()]
        )
        tape.zero()
        return grad
