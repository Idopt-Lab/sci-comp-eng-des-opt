"""The cost groups and the problem builders everything else uses.

Two groups, one per aircraft class:

- :class:`MilitaryCostGroup` -- DAPCA IV acquisition, the Brandt programme
  roll-up, operations and support, life-cycle cost, COCOMO II software and the
  low-observables adder.
- :class:`CommercialDOCGroup` -- the twelve direct-operating-cost elements of
  AIAA 2025-3499 Eq. 1 and the cost per available seat kilometre.

Both are **feed-forward**: cost flows one way, from design parameters to dollars,
so there is no loop to converge and no nonlinear solver.  That is the structural
difference from ``ex_01_asw``, where empty weight depends on takeoff weight which
depends on empty weight.

Every subsystem promotes all of its variables, so a name that is an output of one
component and an input of another is connected automatically, and a name that is
an input to several components (``quantity``, ``block_time_hr``) becomes a single
shared input you set once.
"""

from __future__ import annotations

import openmdao.api as om

from .components import (
    AmprWeight,
    CostEscalation,
    CrewCost,
    DapcaCost,
    DocTotal,
    EnergyCost,
    EngineCount,
    FeeCost,
    LifeCycleCost,
    MaintenanceCost,
    MaterialFactor,
    MaxVelocity,
    OperatingCost,
    OwnershipCost,
    ProgramCost,
    ReportedCosts,
    RoskamAcquisition,
    RoskamOperating,
    RoskamRdte,
    SoftwareCost,
    StealthTreatmentCost,
)


class MilitaryCostGroup(om.Group):
    """Military aircraft life-cycle cost, from design parameters to dollars."""

    def initialize(self) -> None:
        self.options.declare("counter", default=None, recordable=False)
        self.options.declare("deriv", default="jax", values=("jax", "fd"))
        self.options.declare(
            "coefficient_set",
            default="1999",
            values=("1999", "2012"),
            desc="Which Raymer dollar-year coefficient set to pair with the wrap rates.",
        )
        self.options.declare(
            "signature_level",
            default="none",
            values=("none", "A", "B", "C", "C_radome"),
            desc="Low-observables treatment level; 'none' reproduces the Brandt workbook.",
        )
        self.options.declare(
            "cost_factors",
            default=(1.0, 0.9, 0.9, 1.0, 1.5),
            desc="Per-material cost factors (Brandt workbook): Al, carbon fibre, glass, steel, Ti.",
        )

    def setup(self) -> None:
        shared = dict(counter=self.options["counter"], deriv=self.options["deriv"])

        # Feed-forward with no solver: add producers before consumers.
        self.add_subsystem(
            "material",
            MaterialFactor(cost_factors=self.options["cost_factors"], **shared),
            promotes=["*"],
        )
        self.add_subsystem("speed", MaxVelocity(**shared), promotes=["*"])
        self.add_subsystem("engines", EngineCount(**shared), promotes=["*"])
        self.add_subsystem(
            "stealth",
            StealthTreatmentCost(
                signature_level=self.options["signature_level"],
                coefficient_set=self.options["coefficient_set"],
                **shared,
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            "dapca",
            DapcaCost(coefficient_set=self.options["coefficient_set"], **shared),
            promotes=["*"],
        )
        self.add_subsystem("software", SoftwareCost(**shared), promotes=["*"])
        self.add_subsystem("program", ProgramCost(**shared), promotes=["*"])
        self.add_subsystem("operations", OperatingCost(**shared), promotes=["*"])
        self.add_subsystem("lifecycle", LifeCycleCost(**shared), promotes=["*"])
        self.add_subsystem("reported", ReportedCosts(**shared), promotes=["*"])


class RoskamCostGroup(om.Group):
    """Roskam Part VIII whole-programme cost: RDT&E, operations, acquisition, LCC.

    Where DAPCA IV estimates acquisition and leaves operations to a separate
    model, Roskam covers the whole programme in one consistent set of CERs -- and
    exposes judgement factors DAPCA has no equivalent for: technology
    aggressiveness, CAD experience, material choice and low-observability.

    Note the ordering: operations must be computed *before* acquisition, because
    the production flight-test term is priced at the operating cost per hour.
    """

    def initialize(self) -> None:
        self.options.declare("counter", default=None, recordable=False)
        self.options.declare("deriv", default="jax", values=("jax", "fd"))
        self.options.declare("rdte_overhead_fractions", default=(0.1, 0.1, 0.1))
        self.options.declare("ops_overhead_fractions", default=(0.2, 0.125, 0.15, 0.055))
        self.options.declare("disposal_fraction", default=0.01)

    def setup(self) -> None:
        shared = dict(counter=self.options["counter"], deriv=self.options["deriv"])

        self.add_subsystem("ampr", AmprWeight(**shared), promotes=["*"])
        self.add_subsystem("escalation", CostEscalation(**shared), promotes=["*"])
        # The software component's ``quantity`` is Roskam's buy, ``n_production``.
        self.add_subsystem(
            "software", SoftwareCost(**shared),
            promotes_outputs=["*"],
            promotes_inputs=[
                "ksloc", "scale_factor_sum", "effort_multiplier_product",
                "usd_per_person_month", ("quantity", "n_production"),
            ],
        )
        self.add_subsystem(
            "rdte",
            RoskamRdte(rdte_overhead_fractions=self.options["rdte_overhead_fractions"], **shared),
            promotes=["*"],
        )
        self.add_subsystem(
            "operations",
            RoskamOperating(
                ops_overhead_fractions=self.options["ops_overhead_fractions"], **shared
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            "acquisition",
            RoskamAcquisition(disposal_fraction=self.options["disposal_fraction"], **shared),
            promotes=["*"],
        )


class CommercialDOCGroup(om.Group):
    """Direct operating cost per trip for a passenger transport aircraft."""

    def initialize(self) -> None:
        self.options.declare("counter", default=None, recordable=False)
        self.options.declare("deriv", default="jax", values=("jax", "fd"))
        self.options.declare("payments_per_year", default=12.0)

    def setup(self) -> None:
        shared = dict(counter=self.options["counter"], deriv=self.options["deriv"])

        self.add_subsystem(
            "ownership",
            OwnershipCost(payments_per_year=self.options["payments_per_year"], **shared),
            promotes=["*"],
        )
        self.add_subsystem("crew", CrewCost(**shared), promotes=["*"])
        self.add_subsystem("energy", EnergyCost(**shared), promotes=["*"])
        self.add_subsystem("maintenance", MaintenanceCost(**shared), promotes=["*"])
        self.add_subsystem("fees", FeeCost(**shared), promotes=["*"])
        self.add_subsystem("total", DocTotal(**shared), promotes=["*"])


def build_military_problem(input_values, *, deriv="jax", counter=None, **group_options):
    """Return a set-up ``om.Problem`` for :class:`MilitaryCostGroup`.

    ``input_values`` maps promoted variable names to values.  ``group_options``
    accepts ``coefficient_set``, ``signature_level`` and ``cost_factors``.
    """
    return _build(MilitaryCostGroup, input_values, deriv, counter, group_options)


def build_roskam_problem(input_values, *, deriv="jax", counter=None, **group_options):
    """Return a set-up ``om.Problem`` for :class:`RoskamCostGroup`."""
    return _build(RoskamCostGroup, input_values, deriv, counter, group_options)


def build_commercial_problem(input_values, *, deriv="jax", counter=None, **group_options):
    """Return a set-up ``om.Problem`` for :class:`CommercialDOCGroup`."""
    return _build(CommercialDOCGroup, input_values, deriv, counter, group_options)


def _build(group_class, input_values, deriv, counter, group_options):
    # reports=False keeps OpenMDAO from writing a per-problem reports directory on
    # every solve; the app and the sweeps build a great many problems.
    prob = om.Problem(reports=False)
    prob.model = group_class(deriv=deriv, counter=counter, **group_options)
    prob.setup(force_alloc_complex=False)
    for name, value in input_values.items():
        prob.set_val(name, value)
    return prob
