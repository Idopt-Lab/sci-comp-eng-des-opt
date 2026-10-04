"""OpenMDAO components that wrap the pure-JAX disciplines.

The teaching point of this module is how analytic gradients get into an MDAO
framework: each explicit discipline is a thin ``om.ExplicitComponent`` whose
``compute`` calls a JAX function and whose ``compute_partials`` fills the Jacobian
from ``jax.jacobian`` of that same function.  The coupling that closes the sizing
loop lives in the implicit ``Sizing`` component, which exposes a residual so a
Group-level nonlinear solver (fixed-point / Newton / Broyden) can drive it to zero.

The autodiff functions are JIT-compiled and cached (``_JITTED``) so that repeated
solves -- e.g. the Streamlit sweep, or a Newton run that re-linearizes every
iteration -- reuse the compiled derivative instead of re-tracing it each call.

Every component optionally increments a shared :class:`CallCounter` so Lesson 2 can
report how many function / derivative evaluations each solver actually costs.
"""

from __future__ import annotations

from collections import defaultdict

import jax
import jax.numpy as jnp
import numpy as np
import openmdao.api as om

from . import disciplines as D


class CallCounter:
    """Tallies component evaluations, keyed by ``(component, kind)``.

    ``kind`` is e.g. ``"compute"`` (a function evaluation) or ``"partials"`` (a
    derivative evaluation).  Reset it before a run, read ``counts`` afterwards.
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.counts: dict[tuple[str, str], int] = defaultdict(int)

    def tick(self, component: str, kind: str) -> None:
        self.counts[(component, kind)] += 1

    def total(self, kind: str | None = None) -> int:
        return sum(n for (_, k), n in self.counts.items() if kind is None or k == kind)

    def as_rows(self) -> list[tuple[str, str, int]]:
        return sorted((c, k, n) for (c, k), n in self.counts.items())


# Cache of (jitted primal, jitted Jacobian) keyed by component class, so the
# expensive trace/compile happens once and is reused across every solve.
_JITTED: dict[type, tuple] = {}


def _jitted_pair(primal_fn, n_consts: int):
    static = tuple(range(1, 1 + n_consts))  # constant args are compile-time static
    jitted_primal = jax.jit(primal_fn, static_argnums=static)
    jitted_jacobian = jax.jit(jax.jacobian(primal_fn, argnums=0), static_argnums=static)
    return jitted_primal, jitted_jacobian


class JaxExplicitComponent(om.ExplicitComponent):
    """Base class: turn a pure-JAX ``primal_fn(x, *consts) -> y`` into a component.

    Subclasses declare ``input_names`` / ``output_names`` (scalars),
    ``const_names`` (option names passed to the primal as compile-time constants),
    and a static ``primal_fn``.  This base fills ``compute`` and
    ``compute_partials`` generically, so the autodiff wiring is written once.
    """

    input_names: tuple[str, ...] = ()
    output_names: tuple[str, ...] = ()
    const_names: tuple[str, ...] = ()
    #: Optional sparsity: {output: (inputs it depends on, ...)}. ``None`` == dense.
    dependencies: dict[str, tuple[str, ...]] | None = None

    @staticmethod
    def primal_fn(x, *consts):  # noqa: D401 - simple hook
        """Map a 1-D JAX array of inputs (+ constants) to a 1-D array of outputs."""
        raise NotImplementedError

    def initialize(self) -> None:
        self.options.declare("counter", default=None, recordable=False)
        self.options.declare(
            "deriv", default="jax", values=("jax", "fd"),
            desc="How partials are supplied: analytic JAX, or OpenMDAO finite difference.",
        )

    # --- shared machinery ------------------------------------------------- #
    def _pack(self, inputs) -> jnp.ndarray:
        return jnp.array([float(inputs[name][0]) for name in self.input_names])

    def _consts(self) -> tuple:
        return tuple(self.options[name] for name in self.const_names)

    def _dependencies(self, output_name: str) -> tuple[str, ...]:
        if self.dependencies is None:
            return self.input_names
        return self.dependencies[output_name]

    def setup(self) -> None:
        cls = type(self)
        if cls not in _JITTED:
            _JITTED[cls] = _jitted_pair(cls.primal_fn, len(self.const_names))
        self._jit_primal, self._jit_jacobian = _JITTED[cls]

        for name in self.input_names:
            self.add_input(name)
        for name in self.output_names:
            self.add_output(name)
        for out_name in self.output_names:
            wrt = list(self._dependencies(out_name))
            if self.options["deriv"] == "fd":
                self.declare_partials(out_name, wrt, method="fd")
            else:
                self.declare_partials(out_name, wrt)

    def compute(self, inputs, outputs) -> None:
        counter = self.options["counter"]
        if counter is not None:
            counter.tick(type(self).__name__, "compute")
        y = np.asarray(self._jit_primal(self._pack(inputs), *self._consts()))
        for name, value in zip(self.output_names, y):
            outputs[name] = value

    def compute_partials(self, inputs, partials) -> None:
        if self.options["deriv"] == "fd":
            return  # OpenMDAO finite-differences these; no analytic work here.
        counter = self.options["counter"]
        if counter is not None:
            counter.tick(type(self).__name__, "partials")
        # One (cached) autodiff call yields the full output-by-input Jacobian.
        jacobian = np.asarray(self._jit_jacobian(self._pack(inputs), *self._consts()))
        in_index = {name: j for j, name in enumerate(self.input_names)}
        for i, out_name in enumerate(self.output_names):
            for in_name in self._dependencies(out_name):
                partials[out_name, in_name] = jacobian[i, in_index[in_name]]


class Aerodynamics(JaxExplicitComponent):
    """AR, S_wet/S_ref -> L/D at cruise and loiter (Raymer Eq. 3.12)."""

    input_names = ("wing_aspect_ratio", "wetted_area_ratio")
    output_names = ("lift_to_drag_max", "cruise_lift_to_drag", "loiter_lift_to_drag")
    const_names = ("k_ld", "cruise_ld_factor")

    def initialize(self) -> None:
        super().initialize()
        self.options.declare("k_ld", default=14.0)
        self.options.declare("cruise_ld_factor", default=0.866)

    @staticmethod
    def primal_fn(x, k_ld, cruise_ld_factor):
        ld_max, ld_cruise, ld_loiter = D.aerodynamics(x[0], x[1], k_ld, cruise_ld_factor)
        return jnp.stack([ld_max, ld_cruise, ld_loiter])


class Propulsion(JaxExplicitComponent):
    """Flight condition + engine deck -> cruise speed and per-second TSFCs."""

    input_names = ("mach", "speed_of_sound", "tsfc_cruise_per_hr", "tsfc_loiter_per_hr")
    output_names = ("cruise_speed", "sfc_cruise", "sfc_loiter")
    dependencies = {
        "cruise_speed": ("mach", "speed_of_sound"),
        "sfc_cruise": ("tsfc_cruise_per_hr",),
        "sfc_loiter": ("tsfc_loiter_per_hr",),
    }

    @staticmethod
    def primal_fn(x):
        speed, sfc_cruise, sfc_loiter = D.propulsion(x[0], x[1], x[2], x[3])
        return jnp.stack([speed, sfc_cruise, sfc_loiter])


class MissionFuel(JaxExplicitComponent):
    """Breguet mission analysis -> fuel weight fraction ``Wf/WTO``."""

    input_names = (
        "cruise_lift_to_drag",
        "loiter_lift_to_drag",
        "cruise_speed",
        "sfc_cruise",
        "sfc_loiter",
        "range_ft",
        "station_endurance_s",
        "preland_endurance_s",
    )
    output_names = ("fuel_weight_fraction",)
    const_names = ("warmup", "climb", "landing", "reserve", "trapped")

    def initialize(self) -> None:
        super().initialize()
        self.options.declare("warmup", default=0.970)
        self.options.declare("climb", default=0.985)
        self.options.declare("landing", default=0.995)
        self.options.declare("reserve", default=0.05)
        self.options.declare("trapped", default=0.01)

    @staticmethod
    def primal_fn(x, warmup, climb, landing, reserve, trapped):
        wf = D.fuel_weight_fraction(
            x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7],
            warmup, climb, landing, reserve, trapped,
        )
        return jnp.stack([wf])


class Structures(JaxExplicitComponent):
    """Takeoff gross weight -> empty-weight fraction ``We/WTO`` (the feedback)."""

    input_names = ("takeoff_gross_weight",)
    output_names = ("empty_weight_fraction",)
    const_names = ("coefficient", "exponent", "material_factor")

    def initialize(self) -> None:
        super().initialize()
        self.options.declare("coefficient", default=0.93)
        self.options.declare("exponent", default=-0.07)
        self.options.declare("material_factor", default=1.0)

    @staticmethod
    def primal_fn(x, coefficient, exponent, material_factor):
        we = D.empty_weight_fraction(x[0], coefficient, exponent, material_factor)
        return jnp.stack([we])


class SizingResidual(JaxExplicitComponent):
    """Explicit form of the sizing residual, for the *optimization* formulations.

    ``R = WTO*(1 - Wf/WTO - We/WTO) - W_fixed``.  Where the implicit ``Sizing``
    component lets a solver drive ``R`` to zero by owning ``W_TO`` as a state, this
    exposes ``R`` as an ordinary output so an optimizer can own ``W_TO`` as a design
    variable -- either minimizing ``W_TO`` subject to ``R >= 0`` or minimizing
    ``(R/ref)^2``.  Same JAX/FD partials machinery as every other discipline.
    """

    input_names = (
        "takeoff_gross_weight",
        "fuel_weight_fraction",
        "empty_weight_fraction",
        "fixed_weight",
    )
    output_names = ("sizing_residual",)

    @staticmethod
    def primal_fn(x):
        return jnp.stack([D.sizing_residual(x[0], x[1], x[2], x[3])])


# JIT the sizing residual and its gradient once (Newton re-linearizes every step).
_sizing_residual = jax.jit(D.sizing_residual)
_sizing_grads = jax.jit(jax.grad(D.sizing_residual, argnums=(0, 1, 2, 3)))


class Sizing(om.ImplicitComponent):
    """Implicit closure of the sizing loop.

    State ``takeoff_gross_weight`` with residual
    ``WTO*(1 - Wf/WTO - We/WTO) - W_fixed``.  ``solve_nonlinear`` inverts the
    identity analytically, so a Group ``NonlinearBlockGS`` reproduces the classic
    Raymer fixed-point iteration exactly; ``apply_nonlinear`` + ``linearize`` let a
    Group Newton / Broyden solver drive the same residual with JAX gradients.
    """

    def initialize(self) -> None:
        self.options.declare("counter", default=None, recordable=False)
        self.options.declare("deriv", default="jax", values=("jax", "fd"))

    def setup(self) -> None:
        self.add_input("fuel_weight_fraction")
        self.add_input("empty_weight_fraction")
        self.add_input("fixed_weight", val=10_800.0)
        # lower keeps We/WTO = a*WTO^b real (a negative WTO would be NaN) for the
        # bounds-enforcing linesearches Newton and Broyden run; NonlinearBlockGS
        # calls solve_nonlinear instead, which clamps the denominator itself.
        # ref scales the ~5e4 lb state so Newton is well conditioned.
        self.add_output("takeoff_gross_weight", val=50_000.0, lower=1.0, ref=1.0e4)

        wrt = ["takeoff_gross_weight", "fuel_weight_fraction", "empty_weight_fraction", "fixed_weight"]
        if self.options["deriv"] == "fd":
            self.declare_partials("takeoff_gross_weight", wrt, method="fd")
        else:
            self.declare_partials("takeoff_gross_weight", wrt)

    def apply_nonlinear(self, inputs, outputs, residuals) -> None:
        counter = self.options["counter"]
        if counter is not None:
            counter.tick("Sizing", "apply_nonlinear")
        residuals["takeoff_gross_weight"] = _sizing_residual(
            outputs["takeoff_gross_weight"][0],
            inputs["fuel_weight_fraction"][0],
            inputs["empty_weight_fraction"][0],
            inputs["fixed_weight"][0],
        )

    def solve_nonlinear(self, inputs, outputs) -> None:
        counter = self.options["counter"]
        if counter is not None:
            counter.tick("Sizing", "solve_nonlinear")
        denominator = (
            1.0 - inputs["fuel_weight_fraction"][0] - inputs["empty_weight_fraction"][0]
        )
        # A non-positive denominator means fuel plus structure already consume the
        # whole aircraft, and the raw update would be negative -- after which
        # We/WTO = a*WTO^b is NaN (JAX) or complex (Python) and the run dies with a
        # type error instead of a diagnosis.  Clamping keeps W_TO real and positive
        # so the solver simply fails to converge; ``solve`` reports that as a
        # SizingDivergedError.
        outputs["takeoff_gross_weight"] = inputs["fixed_weight"][0] / max(
            denominator, D.MINIMUM_SIZING_DENOMINATOR
        )

    def linearize(self, inputs, outputs, partials) -> None:
        if self.options["deriv"] == "fd":
            return  # OpenMDAO finite-differences the residual; no analytic work here.
        counter = self.options["counter"]
        if counter is not None:
            counter.tick("Sizing", "linearize")
        grads = _sizing_grads(
            outputs["takeoff_gross_weight"][0],
            inputs["fuel_weight_fraction"][0],
            inputs["empty_weight_fraction"][0],
            inputs["fixed_weight"][0],
        )
        partials["takeoff_gross_weight", "takeoff_gross_weight"] = float(grads[0])
        partials["takeoff_gross_weight", "fuel_weight_fraction"] = float(grads[1])
        partials["takeoff_gross_weight", "empty_weight_fraction"] = float(grads[2])
        partials["takeoff_gross_weight", "fixed_weight"] = float(grads[3])
