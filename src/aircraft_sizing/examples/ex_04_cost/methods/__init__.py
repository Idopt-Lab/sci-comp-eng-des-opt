"""OpenMDAO + JAX aircraft cost models, military and commercial.

Public entry points:

- :class:`~.config.MilitaryCostInputs` / :func:`~.config.solve_military` -- DAPCA IV
  acquisition, the Brandt programme roll-up, O&M, life-cycle cost, COCOMO II
  software and the low-observables adder.
- :class:`~.config.RoskamCostInputs` / :func:`~.config.solve_roskam` -- Roskam Part
  VIII whole-programme cost: RDT&E, operations, acquisition and life-cycle cost.
- :class:`~.config.CommercialCostInputs` / :func:`~.config.solve_commercial` -- the
  twelve direct-operating-cost elements of AIAA 2025-3499 Eq. 1.
- :mod:`~.military` / :mod:`~.commercial` -- the pure-JAX cost relations.
- :mod:`~.aircraft` -- baseline input sets and published costs to compare against.
"""

from .aircraft import (
    COMMERCIAL_BASELINES,
    MILITARY_AIRCRAFT,
    PUBLISHED_FLYAWAY,
    REPORT_DOLLAR_YEAR,
    SCENARIOS,
    VALIDATION_AIRCRAFT,
    commercial_inputs_for,
    commercial_names,
    escalate_published,
    military_names,
)
from .config import (
    CommercialCostInputs,
    InputError,
    MilitaryCostInputs,
    RoskamCostInputs,
    input_sensitivity_sweep,
    solve_commercial,
    solve_military,
    solve_roskam,
)
from .group import (
    CommercialDOCGroup,
    MilitaryCostGroup,
    RoskamCostGroup,
    build_commercial_problem,
    build_military_problem,
    build_roskam_problem,
)

__all__ = [
    "MilitaryCostInputs",
    "RoskamCostInputs",
    "CommercialCostInputs",
    "InputError",
    "solve_military",
    "solve_roskam",
    "solve_commercial",
    "input_sensitivity_sweep",
    "MilitaryCostGroup",
    "RoskamCostGroup",
    "CommercialDOCGroup",
    "build_military_problem",
    "build_roskam_problem",
    "build_commercial_problem",
    "MILITARY_AIRCRAFT",
    "VALIDATION_AIRCRAFT",
    "COMMERCIAL_BASELINES",
    "REPORT_DOLLAR_YEAR",
    "SCENARIOS",
    "PUBLISHED_FLYAWAY",
    "commercial_inputs_for",
    "military_names",
    "commercial_names",
    "escalate_published",
]
