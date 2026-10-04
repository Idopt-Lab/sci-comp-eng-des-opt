"""The C172 trim optimization problem: Mach sweep, DV packing, and scaling.

The design vector is laid out in **blocks** -- all pitch angles, then all elevator
deflections, then all propeller speeds::

    x = [theta_0 .. theta_{N-1} | delta_e_0 .. delta_e_{N-1} | omega_0 .. omega_{N-1}]

Each physical variable is mapped onto ``z in [0, 1]`` by its bounds, so the
optimizer and the finite-difference steps see a well-scaled, O(1) design space
regardless of the wildly different physical magnitudes (radians vs. ~2000 RPM).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import radians

import numpy as np

from .aircraft import C172Data

__all__ = ["TrimProblem", "THETA_BOUNDS_DEG", "DELTA_E_BOUNDS_DEG", "OMEGA_BOUNDS_RPM"]

#: Physical design-variable bounds (the scaling box).
THETA_BOUNDS_DEG = (-10.0, 15.0)
DELTA_E_BOUNDS_DEG = (-15.0, 15.0)
OMEGA_BOUNDS_RPM = (1000.0, 2800.0)

#: A physical starting guess per node (comfortably inside the box and the basin).
_THETA0_DEG = 5.0
_DELTA_E0_DEG = -5.0
_OMEGA0_RPM = 2000.0


def _sweep_mach(n_nodes: int) -> np.ndarray:
    """Mach schedule for ``n_nodes`` trim points.

    A single node sits at M = 0.117 (matching the original worked example); two or
    more spread linearly over M = 0.10 .. 0.17, where a trim exists inside the RPM
    box (M = 0.18 would need more than 2700 RPM).
    """
    if n_nodes == 1:
        return np.array([0.117])
    return np.linspace(0.10, 0.17, n_nodes)


@dataclass(frozen=True)
class TrimProblem:
    """Immutable description of an ``N``-node trim problem."""

    data: C172Data
    mach: np.ndarray
    lower_physical: np.ndarray = field(init=False)
    upper_physical: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        n = self.n_nodes
        lower = np.concatenate([
            np.full(n, radians(THETA_BOUNDS_DEG[0])),
            np.full(n, radians(DELTA_E_BOUNDS_DEG[0])),
            np.full(n, OMEGA_BOUNDS_RPM[0]),
        ])
        upper = np.concatenate([
            np.full(n, radians(THETA_BOUNDS_DEG[1])),
            np.full(n, radians(DELTA_E_BOUNDS_DEG[1])),
            np.full(n, OMEGA_BOUNDS_RPM[1]),
        ])
        object.__setattr__(self, "lower_physical", lower)
        object.__setattr__(self, "upper_physical", upper)

    # --- constructors ----------------------------------------------------- #
    @classmethod
    def with_sweep(cls, n_nodes: int, data: C172Data | None = None) -> "TrimProblem":
        return cls(data=data or C172Data(), mach=_sweep_mach(n_nodes))

    # --- sizes ------------------------------------------------------------ #
    @property
    def n_nodes(self) -> int:
        return int(self.mach.shape[0])

    @property
    def n_dv(self) -> int:
        return 3 * self.n_nodes

    # --- block (un)packing ------------------------------------------------ #
    def unpack(self, x: np.ndarray):
        """Split a block vector into ``(theta, delta_e, omega)`` arrays of length N."""
        n = self.n_nodes
        return x[..., :n], x[..., n : 2 * n], x[..., 2 * n :]

    def pack(self, theta, delta_e, omega) -> np.ndarray:
        return np.concatenate([np.atleast_1d(theta), np.atleast_1d(delta_e), np.atleast_1d(omega)])

    # --- scaling (physical <-> z in [0, 1]) ------------------------------- #
    @property
    def _span(self) -> np.ndarray:
        return self.upper_physical - self.lower_physical

    def scale(self, x_physical: np.ndarray) -> np.ndarray:
        return (x_physical - self.lower_physical) / self._span

    def unscale(self, z: np.ndarray) -> np.ndarray:
        return self.lower_physical + z * self._span

    def dx_dz(self) -> np.ndarray:
        """Diagonal of the (constant) Jacobian d(physical)/d(z)."""
        return self._span

    # --- optimizer inputs ------------------------------------------------- #
    def z0(self) -> np.ndarray:
        """Scaled initial guess."""
        n = self.n_nodes
        x0 = np.concatenate([
            np.full(n, radians(_THETA0_DEG)),
            np.full(n, radians(_DELTA_E0_DEG)),
            np.full(n, _OMEGA0_RPM),
        ])
        return self.scale(x0)

    def bounds_z(self) -> list[tuple[float, float]]:
        return [(0.0, 1.0)] * self.n_dv
