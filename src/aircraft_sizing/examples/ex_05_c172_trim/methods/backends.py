"""Backend registry: which derivative backends are importable, and how to build one.

Availability is probed with :func:`importlib.util.find_spec` so that importing this
module never drags in an optional dependency.  ``numpy`` and ``jax`` are core
dependencies; ``casadi``, ``csdl`` (csdl-alpha), and ``warp`` are optional extras.
"""

from __future__ import annotations

import importlib
import importlib.util

from .problem import TrimProblem

__all__ = [
    "BACKEND_NAMES",
    "EXACT_GRAD_BACKENDS",
    "is_available",
    "available_backends",
    "make_backend",
]

#: name -> (module suffix, class name, importable package that must exist)
_REGISTRY: dict[str, tuple[str, str, str]] = {
    "numpy": ("backend_numpy", "NumpyBackend", "numpy"),
    "jax": ("backend_jax", "JaxBackend", "jax"),
    "casadi": ("backend_casadi", "CasadiBackend", "casadi"),
    "csdl": ("backend_csdl", "CsdlBackend", "csdl_alpha"),
    "warp": ("backend_warp", "WarpBackend", "warp"),
}

BACKEND_NAMES = tuple(_REGISTRY)
#: Backends that supply an analytic (non-finite-difference) gradient.
EXACT_GRAD_BACKENDS = ("jax", "casadi", "csdl", "warp")


def is_available(name: str) -> bool:
    """True if the package backing ``name`` can be imported."""
    try:
        _, _, package = _REGISTRY[name]
    except KeyError:
        raise ValueError(f"unknown backend {name!r}; choose from {BACKEND_NAMES}") from None
    try:
        return importlib.util.find_spec(package) is not None
    except ModuleNotFoundError:
        return False


def available_backends(*, exact_grad_only: bool = False) -> list[str]:
    """List the installed backends, optionally only those with analytic gradients."""
    names = EXACT_GRAD_BACKENDS if exact_grad_only else BACKEND_NAMES
    return [n for n in names if is_available(n)]


def make_backend(name: str, problem: TrimProblem):
    """Import (lazily) and construct the backend ``name`` for ``problem``."""
    module_suffix, class_name, _ = _REGISTRY[name]
    module = importlib.import_module(f".{module_suffix}", __package__)
    return getattr(module, class_name)(problem)
