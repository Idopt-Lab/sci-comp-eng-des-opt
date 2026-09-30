"""Public surface: input objects, a ``solve`` for each cost model, and a sweep helper.

The two input dataclasses hold every visible assumption with a documented default.
Field names are the same names the OpenMDAO group promotes, so building the
problem is a direct hand-off and there is no translation table to keep in step.

Results come back as a plain ``dict`` of every group output keyed by variable
name.  That is deliberately simpler than a forty-field result class: the app, the
sweeps and the tests all just index it.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace

from .group import build_commercial_problem, build_military_problem, build_roskam_problem

__all__ = [
    "MilitaryCostInputs",
    "RoskamCostInputs",
    "CommercialCostInputs",
    "solve_military",
    "solve_roskam",
    "solve_commercial",
    "batch_solve_military",
    "batch_solve_commercial",
    "elasticities",
    "military_elasticities",
    "input_sensitivity_sweep",
    "MILITARY_GROUP_OPTIONS",
    "ROSKAM_GROUP_OPTIONS",
    "COMMERCIAL_GROUP_OPTIONS",
]

#: Fields that configure the group rather than feed a variable.
MILITARY_GROUP_OPTIONS = ("coefficient_set", "signature_level", "cost_factors")
ROSKAM_GROUP_OPTIONS = (
    "rdte_overhead_fractions",
    "ops_overhead_fractions",
    "disposal_fraction",
)
COMMERCIAL_GROUP_OPTIONS = ("payments_per_year",)


class InputError(ValueError):
    """Raised when an input is outside the range a cost model can honour."""


@dataclass(frozen=True, slots=True)
class MilitaryCostInputs:
    """Every assumption behind a military life-cycle cost estimate.

    Defaults are the F-16A as recorded in ``Brandt-F16-A.xls`` sheet ``Cost``, so
    ``MilitaryCostInputs()`` reproduces the published workbook exactly.
    """

    # --- airframe ---------------------------------------------------------- #
    we_lb: float = 19_980.7005781593
    mach_max: float = 2.0
    # --- programme --------------------------------------------------------- #
    quantity: float = 200.0
    flight_test_aircraft: float = 3.0
    #: Engines are bought per airframe; the fleet total the engine CER needs is
    #: computed in the group, so it cannot fall out of step with the buy.
    engines_per_aircraft: float = 1.0
    # --- propulsion -------------------------------------------------------- #
    thrust_max_lbf: float = 23_770.0
    turbine_inlet_temp_r: float = 3_400.0
    # --- material mix (percent of structure) ------------------------------- #
    aluminium_percent: float = 65.0
    carbon_fibre_percent: float = 20.0
    fibreglass_percent: float = 0.0
    steel_percent: float = 5.0
    titanium_percent: float = 10.0
    # --- wrap rates, same dollar year as coefficient_set ------------------- #
    rate_engineering_usd_per_hr: float = 86.0
    rate_tooling_usd_per_hr: float = 88.0
    rate_manufacturing_usd_per_hr: float = 73.0
    rate_quality_usd_per_hr: float = 81.0
    # --- programme economics ----------------------------------------------- #
    avionics_factor: float = 0.5
    investment_factor: float = 0.0
    escalation_factor: float = 0.16
    # --- operations and support -------------------------------------------- #
    design_mission_fuel_lb: float = 6_000.43227294689
    design_mission_time_hr: float = 1.56763275690196
    average_mission_factor: float = 1.0
    flight_hours_per_year: float = 500.0
    fuel_usd_per_lb: float = 0.3
    crew_ratio: float = 1.1
    crew_hours_per_year: float = 300.0
    maintenance_hours_per_flight_hour: float = 15.0
    life_years: float = 20.0
    # --- software (COCOMO II) ---------------------------------------------- #
    #: Zero rate by default: the workbook has no software line.
    ksloc: float = 200.0
    scale_factor_sum: float = 14.67
    effort_multiplier_product: float = 2.5636114
    usd_per_person_month: float = 0.0
    # --- low-observables geometry ------------------------------------------ #
    skin_wetted_area_ft2: float = 1_479.5800436568
    inlet_lip_length_ft: float = 11.5429484714568
    inlet_duct_area_ft2: float = 196.230124014765
    treated_edge_length_ft: float = 162.599342880023
    vertical_tail_area_ft2: float = 90.7324981111512
    vertical_tail_outer_area_ft2: float = 90.7324981111512
    hinge_line_length_ft: float = 83.2826512546281
    radar_bulkhead_area_ft2: float = 6.424
    access_panel_perimeter_ft: float = 45.0
    exhaust_area_ft2: float = 34.6288454143703
    # --- dollar years ------------------------------------------------------- #
    #: The year the escalated model outputs are already in -- for the workbook,
    #: 1999 wrap rates escalated by EF = 0.16 lands in 2006 dollars.
    model_dollar_year: float = 2006.0
    #: Everything reported is moved to this year, so one chart has one unit.
    report_dollar_year: float = 2026.0
    # --- group options, not variables -------------------------------------- #
    coefficient_set: str = "1999"
    signature_level: str = "none"
    cost_factors: tuple = (1.0, 0.9, 0.9, 1.0, 1.5)

    def __post_init__(self) -> None:
        _validate_military(self)

    @classmethod
    def baseline(cls) -> "MilitaryCostInputs":
        """The F-16A workbook baseline."""
        return cls()

    def with_value(self, name: str, value) -> "MilitaryCostInputs":
        """Return a copy with one field replaced (used by the sweep helper)."""
        return _replace_one(self, name, value)

    def input_values(self) -> dict:
        """Promoted variable name -> value, ready for ``build_military_problem``."""
        return _variable_fields(self, MILITARY_GROUP_OPTIONS)

    def group_options(self) -> dict:
        """The non-variable settings the group takes as options."""
        return {name: getattr(self, name) for name in MILITARY_GROUP_OPTIONS}


@dataclass(frozen=True, slots=True)
class RoskamCostInputs:
    """Every assumption behind a Roskam Part VIII whole-programme estimate.

    Defaults describe a twin-engine stealth strike fighter of about 60,000 lb
    takeoff weight bought in quantity 500 -- representative of the senior-design
    programmes this example is written for, not a reproduction of any one of them.
    """

    # --- airframe and programme -------------------------------------------- #
    w_to_lb: float = 60_000.0
    v_max_keas: float = 965.0
    n_production: float = 500.0
    n_rdte: float = 8.0
    n_static_test: float = 2.0
    engines_per_aircraft: float = 2.0
    # --- judgement factors Roskam exposes and DAPCA does not ---------------- #
    difficulty_factor: float = 1.3
    cad_factor: float = 0.8
    material_factor_roskam: float = 1.75
    observables_factor: float = 1.3
    # --- year of economics and the 1989-base labour rates ------------------- #
    year_of_economics: float = 2025.0
    rate_engineering_1989_usd_per_hr: float = 92.0
    rate_manufacturing_1989_usd_per_hr: float = 51.0
    rate_tooling_1989_usd_per_hr: float = 65.0
    rate_maintenance_1989_usd_per_hr: float = 45.0
    rate_consumables_1989_usd_per_hr: float = 6.5
    # --- bought-out items ---------------------------------------------------- #
    roskam_engine_cost_usd: float = 10.0e6
    roskam_avionics_cost_usd: float = 13.25e6
    rdte_production_rate_per_month: float = 0.33
    production_rate_per_month: float = 8.0
    # --- software (COCOMO II) ------------------------------------------------ #
    ksloc: float = 8_000.0
    scale_factor_sum: float = 14.67
    effort_multiplier_product: float = 2.5636114
    usd_per_person_month: float = 25_000.0
    # --- operations ---------------------------------------------------------- #
    reserve_fraction: float = 0.1
    loss_rate_per_hour: float = 2.1e-6
    utilization_hr_per_year: float = 300.0
    service_years: float = 30.0
    mission_fuel_lb: float = 20_300.0
    mission_time_hr: float = 2.5
    fuel_usd_per_gal: float = 3.85
    fuel_density_lb_per_gal: float = 6.82
    oil_lubricant_factor: float = 1.005
    crew_per_aircraft: float = 1.0
    crew_ratio: float = 1.1
    crew_pay_usd_per_year: float = 118_000.0
    crew_overhead_factor: float = 3.0
    maintenance_hours_per_flight_hour: float = 20.0
    # --- group options, not variables ---------------------------------------- #
    rdte_overhead_fractions: tuple = (0.1, 0.1, 0.1)
    ops_overhead_fractions: tuple = (0.2, 0.125, 0.15, 0.055)
    disposal_fraction: float = 0.01

    def __post_init__(self) -> None:
        _validate_roskam(self)

    @classmethod
    def baseline(cls) -> "RoskamCostInputs":
        """The representative fighter programme."""
        return cls()

    def with_value(self, name: str, value) -> "RoskamCostInputs":
        """Return a copy with one field replaced (used by the sweep helper)."""
        return _replace_one(self, name, value)

    def input_values(self) -> dict:
        """Promoted variable name -> value, ready for ``build_roskam_problem``."""
        return _variable_fields(self, ROSKAM_GROUP_OPTIONS)

    def group_options(self) -> dict:
        """The non-variable settings the group takes as options."""
        return {name: getattr(self, name) for name in ROSKAM_GROUP_OPTIONS}


@dataclass(frozen=True, slots=True)
class CommercialCostInputs:
    """Every assumption behind a direct-operating-cost estimate.

    Defaults are the ATR 72-500 from AIAA 2025-3499 Table 13 with the fixed rates
    from Table 15 and the 2030 scenario from Table 14.  The mission -- block time,
    fuel burnt, stage length -- is an input because the paper does not publish the
    block-speed or fuel-burn model behind its own results.
    """

    # --- acquisition and ownership ----------------------------------------- #
    aircraft_cost_usd: float = 20.1e6
    airframe_cost_usd: float = 16.9e6
    engine_cost_usd: float = 1.6e6
    n_engines: float = 2.0
    motor_cost_usd: float = 0.0
    n_motors: float = 0.0
    residual_value_fraction: float = 0.20
    financial_life_years: float = 14.0
    utilization_hr_per_year: float = 1_700.0
    annual_interest_rate: float = 0.05
    loan_years: float = 14.0
    insurance_premium_fraction: float = 0.036
    electrified_premium_increase: float = 0.0
    # --- battery (zero for a conventional aircraft) ------------------------ #
    battery_cost_usd: float = 0.0
    battery_residual_fraction: float = 0.225
    battery_reference_cycles: float = 1_000.0
    charge_c_rate: float = 1.26
    # --- crew --------------------------------------------------------------- #
    flight_crew_usd_per_hr: float = 286.0
    cabin_crew_usd_per_hr: float = 39.1
    international_factor: float = 0.0
    n_cabin_crew: float = 2.0
    # --- mission ------------------------------------------------------------ #
    # ATR 72-500 1,000 km mission, as derived by ``commercial_inputs_for``.
    block_time_hr: float = 2.730952380952381
    stage_length_km: float = 1_000.0
    n_passengers: float = 70.0
    mtom_ton: float = 22.0
    oew_kg: float = 12_400.0
    thrust_per_engine_kn: float = 10.231765815324167
    # --- energy and carbon -------------------------------------------------- #
    fuel_mass_kg: float = 1_720.5
    fuel_density_kg_per_gal: float = 3.1
    fuel_usd_per_gal: float = 2.12
    saf_blend: float = 0.06
    saf_price_factor: float = 3.0
    mission_energy_kwh: float = 0.0
    electricity_usd_per_kwh: float = 0.23
    carbon_price_usd_per_ton: float = 70.0
    carbon_taxed_indicator: float = 1.0
    # --- maintenance -------------------------------------------------------- #
    maintenance_reduction_factor: float = 0.0
    # --- fees --------------------------------------------------------------- #
    tans_usd_per_ton: float = 208.0
    en_route_usd_per_ton_km: float = 95.3
    # --- group options, not variables --------------------------------------- #
    payments_per_year: float = 12.0

    def __post_init__(self) -> None:
        _validate_commercial(self)

    @classmethod
    def baseline(cls) -> "CommercialCostInputs":
        """The ATR 72-500 baseline, 1,000 km, 2030 scenario."""
        return cls()

    def with_value(self, name: str, value) -> "CommercialCostInputs":
        """Return a copy with one field replaced (used by the sweep helper)."""
        return _replace_one(self, name, value)

    def input_values(self) -> dict:
        """Promoted variable name -> value, ready for ``build_commercial_problem``."""
        return _variable_fields(self, COMMERCIAL_GROUP_OPTIONS)

    def group_options(self) -> dict:
        """The non-variable settings the group takes as options."""
        return {name: getattr(self, name) for name in COMMERCIAL_GROUP_OPTIONS}


# --------------------------------------------------------------------------- #
# Solving
# --------------------------------------------------------------------------- #
def solve_military(inputs: MilitaryCostInputs, *, deriv: str = "jax") -> dict:
    """Run the military cost model and return every output keyed by name."""
    problem = build_military_problem(
        inputs.input_values(), deriv=deriv, **inputs.group_options()
    )
    return _run(problem)


def solve_roskam(inputs: RoskamCostInputs, *, deriv: str = "jax") -> dict:
    """Run the Roskam whole-programme model and return every output keyed by name."""
    problem = build_roskam_problem(inputs.input_values(), deriv=deriv, **inputs.group_options())
    return _run(problem)


def solve_commercial(inputs: CommercialCostInputs, *, deriv: str = "jax") -> dict:
    """Run the commercial DOC model and return every output keyed by name."""
    problem = build_commercial_problem(
        inputs.input_values(), deriv=deriv, **inputs.group_options()
    )
    return _run(problem)


def batch_solve(inputs, overrides, build, *, deriv: str = "jax") -> list:
    """Solve many variations of one case by reusing a single ``om.Problem``.

    ``overrides`` is a sequence of ``{variable: value}`` dicts applied on top of
    ``inputs``.  Building an ``om.Problem`` costs far more than running one, so a
    sweep or a contour that re-builds per point is dominated by setup; this
    rebuilds nothing.  Group *options* cannot vary within a batch -- change those
    and start a new batch.

    Every override dict must carry the full set of varying names, because values
    persist between runs.
    """
    problem = build(inputs.input_values(), deriv=deriv, **inputs.group_options())
    results = []
    for override in overrides:
        for name, value in override.items():
            problem.set_val(name, value)
        problem.run_model()
        results.append(_collect(problem))
    return results


def batch_solve_military(inputs: MilitaryCostInputs, overrides, *, deriv: str = "jax") -> list:
    """Batch form of :func:`solve_military`."""
    return batch_solve(inputs, overrides, build_military_problem, deriv=deriv)


def batch_solve_commercial(inputs: CommercialCostInputs, overrides, *, deriv: str = "jax") -> list:
    """Batch form of :func:`solve_commercial`."""
    return batch_solve(inputs, overrides, build_commercial_problem, deriv=deriv)


def elasticities(inputs, wrt_names, of_name, build):
    """Percentage change in ``of_name`` per percentage change in each input.

    ``elasticity = (dC/dx) * (x/C)``, so an elasticity of 0.8 means a 10% rise in
    that input raises the cost by 8%.  Being dimensionless, it puts a production
    quantity and a labour rate on the same axis -- which a tornado chart in dollars
    cannot do.

    The derivatives are the analytic JAX ones, which is what they are for.
    """
    problem = build(inputs.input_values(), **inputs.group_options())
    problem.run_model()
    base = float(problem.get_val(of_name)[0])
    if base == 0.0:
        return []

    usable = [name for name in wrt_names if float(getattr(inputs, name)) != 0.0]
    totals = problem.compute_totals(of=[of_name], wrt=usable)
    result = []
    for name in usable:
        gradient = float(totals[of_name, name][0, 0])
        result.append((name, gradient * float(getattr(inputs, name)) / base))
    result.sort(key=lambda row: abs(row[1]), reverse=True)
    return result


def military_elasticities(inputs: MilitaryCostInputs, wrt_names, of_name):
    """Elasticities of a military cost output with respect to each input."""
    return elasticities(inputs, wrt_names, of_name, build_military_problem)


def input_sensitivity_sweep(inputs, name: str, values, solve):
    """Vary one input over ``values``, returning ``[(value, results), ...]``.

    This is the one-at-a-time sweep behind most of the app's trade studies.  It
    deliberately re-solves from the baseline each time rather than marching, so a
    failure at one point cannot contaminate the rest of the curve.
    """
    return [(value, solve(inputs.with_value(name, value))) for value in values]


def _run(problem) -> dict:
    problem.run_model()
    return _collect(problem)


def _collect(problem) -> dict:
    outputs = problem.model.list_outputs(out_stream=None, prom_name=True, val=True)
    return {meta["prom_name"]: float(meta["val"][0]) for _, meta in outputs}


def _variable_fields(inputs, option_names) -> dict:
    return {
        field.name: getattr(inputs, field.name)
        for field in fields(inputs)
        if field.name not in option_names
    }


def _replace_one(inputs, name: str, value):
    known = {field.name for field in fields(inputs)}
    if name not in known:
        raise InputError(f"Unknown input {name!r}.")
    return replace(inputs, **{name: value})


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def _positive(value, name: str) -> None:
    if not value > 0.0:
        raise InputError(f"{name} must be positive; got {value!r}.")


def _fraction(value, name: str) -> None:
    if not 0.0 <= value <= 1.0:
        raise InputError(f"{name} must lie in [0, 1]; got {value!r}.")


def _validate_military(inputs: MilitaryCostInputs) -> None:
    # Fractional powers of non-positive values give NaN, not an error.
    for name in ("we_lb", "mach_max", "quantity", "flight_test_aircraft", "life_years"):
        _positive(getattr(inputs, name), name)
    percents = (
        inputs.aluminium_percent
        + inputs.carbon_fibre_percent
        + inputs.fibreglass_percent
        + inputs.steel_percent
        + inputs.titanium_percent
    )
    if abs(percents - 100.0) > 1.0e-6:
        raise InputError(f"Material percentages must sum to 100; got {percents!r}.")
    if inputs.coefficient_set not in ("1999", "2012"):
        raise InputError(f"coefficient_set must be '1999' or '2012'; got {inputs.coefficient_set!r}.")
    if inputs.signature_level not in ("none", "A", "B", "C", "C_radome"):
        raise InputError(f"Unknown signature_level {inputs.signature_level!r}.")


def _validate_roskam(inputs: RoskamCostInputs) -> None:
    for name in (
        "w_to_lb",
        "v_max_keas",
        "n_production",
        "n_rdte",
        "service_years",
        "utilization_hr_per_year",
        "mission_time_hr",
        "ksloc",
    ):
        _positive(getattr(inputs, name), name)
    if inputs.n_static_test >= inputs.n_rdte:
        raise InputError(
            "n_static_test must be fewer than n_rdte: the flight-test aircraft are "
            f"n_rdte - n_static_test, got {inputs.n_rdte!r} and {inputs.n_static_test!r}."
        )
    # Each phase total is (everything else) / (1 - sum of overhead fractions), so
    # fractions summing to 1 or more make the cost infinite or negative.
    for label, fractions in (
        ("rdte_overhead_fractions", inputs.rdte_overhead_fractions),
        ("ops_overhead_fractions", inputs.ops_overhead_fractions),
    ):
        if sum(fractions) >= 1.0:
            raise InputError(f"{label} must sum to less than 1; got {sum(fractions)!r}.")
    _fraction(inputs.disposal_fraction, "disposal_fraction")


def _validate_commercial(inputs: CommercialCostInputs) -> None:
    for name in (
        "block_time_hr",
        "stage_length_km",
        "n_passengers",
        "utilization_hr_per_year",
        "financial_life_years",
        "loan_years",
        "mtom_ton",
    ):
        _positive(getattr(inputs, name), name)
    for name in ("saf_blend", "residual_value_fraction", "maintenance_reduction_factor"):
        _fraction(getattr(inputs, name), name)
