"""OpenMDAO components that wrap the pure-JAX cost relations.

Each component is a thin ``om.ExplicitComponent`` whose ``compute`` calls one of
the functions in :mod:`~.military` or :mod:`~.commercial`, and whose
``compute_partials`` fills the Jacobian from ``jax.jacobian`` of that same
function.  The wiring is written once in ``JaxExplicitComponent``, which this
example imports from ``ex_01_asw`` rather than copying.

Cost models are feed-forward -- nothing here needs a solver.  What they *do* need
is derivatives, because the interesting questions ("which parameter actually
drives my unit cost?", "what empty weight can I afford at this buy quantity?")
are gradient questions.

As everywhere in this repository, units live in the variable name, not in a
``units=`` argument: ``_usd``, ``_lb``, ``_kts``, ``_hr``, ``_kg``, ``_km``.
"""

from __future__ import annotations

import jax.numpy as jnp
import openmdao.api as om

from aircraft_sizing.examples.ex_01_asw.methods.components import (  # noqa: F401
    CallCounter,
    JaxExplicitComponent,
)

from . import commercial as C
from . import military as M


# --------------------------------------------------------------------------- #
# Military -- DAPCA IV and the Brandt programme roll-up
# --------------------------------------------------------------------------- #
class MaterialFactor(JaxExplicitComponent):
    """Structural material mix -> design/fabrication factor (Brandt Cost!D47)."""

    input_names = (
        "aluminium_percent",
        "carbon_fibre_percent",
        "fibreglass_percent",
        "steel_percent",
        "titanium_percent",
    )
    output_names = ("material_factor",)
    const_names = ("cost_factors",)

    def initialize(self) -> None:
        super().initialize()
        # Per-material cost factors, in input order; default is the Brandt workbook's.
        self.options.declare("cost_factors", default=(1.0, 0.9, 0.9, 1.0, 1.5))

    @staticmethod
    def primal_fn(x, cost_factors):
        return jnp.stack([M.material_factor(x, cost_factors)])


class MaxVelocity(JaxExplicitComponent):
    """Mach limit -> maximum velocity in knots (Brandt Cost!F12)."""

    input_names = ("mach_max",)
    output_names = ("v_max_kts",)

    @staticmethod
    def primal_fn(x):
        return jnp.stack([M.max_velocity_kts(x[0])])


class EngineCount(JaxExplicitComponent):
    """Engines bought for the whole programme: ``N_eng = Q * engines per aircraft``.

    The engine CER prices a *fleet* total, so this has to move with the buy.
    Keeping it a computed output means a quantity sweep cannot leave it behind.
    """

    input_names = ("quantity", "engines_per_aircraft")
    output_names = ("engines_total",)

    @staticmethod
    def primal_fn(x):
        return jnp.stack([x[0] * x[1]])


class DapcaCost(JaxExplicitComponent):
    """RAND DAPCA IV labour hours and the eight cost elements (Raymer Ch. 18).

    Lead coefficients and wrap rates must come from the same dollar year;
    ``coefficient_set`` selects the pair.
    """

    input_names = (
        "we_lb",
        "v_max_kts",
        "quantity",
        "flight_test_aircraft",
        "material_factor",
        "rate_engineering_usd_per_hr",
        "rate_tooling_usd_per_hr",
        "rate_manufacturing_usd_per_hr",
        "rate_quality_usd_per_hr",
        "thrust_max_lbf",
        "mach_max",
        "turbine_inlet_temp_r",
        "engines_total",
    )
    output_names = (
        "h_eng_hr",
        "h_tool_hr",
        "h_mfg_hr",
        "h_qc_hr",
        "c_eng_usd",
        "c_tool_usd",
        "c_mfg_usd",
        "c_qc_usd",
        "c_ds_usd",
        "c_ft_usd",
        "c_mm_usd",
        "c_ep_usd",
        "c_subtotal_usd",
    )
    const_names = ("coefficient_set",)

    def initialize(self) -> None:
        super().initialize()
        self.options.declare("coefficient_set", default="1999", values=("1999", "2012"))

    @staticmethod
    def primal_fn(x, coefficient_set):
        hour_k = M.DAPCA_HOURS_1999 if coefficient_set == "1999" else M.DAPCA_HOURS_2012
        other_k = M.DAPCA_OTHER_1999 if coefficient_set == "1999" else M.DAPCA_OTHER_2012
        hours = M.dapca_hours(x[0], x[1], x[2], x[4], hour_k)
        costs = M.dapca_costs(
            we_lb=x[0],
            v_max_kts=x[1],
            quantity=x[2],
            flight_test_aircraft=x[3],
            material_factor_value=x[4],
            rates=(x[5], x[6], x[7], x[8]),
            thrust_max_lbf=x[9],
            mach_max=x[10],
            turbine_inlet_temp_r=x[11],
            engines_total=x[12],
            hour_coefficients=hour_k,
            other_coefficients=other_k,
        )
        return jnp.stack(
            [
                hours["h_eng_hr"],
                hours["h_tool_hr"],
                hours["h_mfg_hr"],
                hours["h_qc_hr"],
                costs["c_eng_usd"],
                costs["c_tool_usd"],
                costs["c_mfg_usd"],
                costs["c_qc_usd"],
                costs["c_ds_usd"],
                costs["c_ft_usd"],
                costs["c_mm_usd"],
                costs["c_ep_usd"],
                costs["c_subtotal_usd"],
            ]
        )


class ProgramCost(JaxExplicitComponent):
    """Brandt programme roll-up: avionics, investment, signature and escalation."""

    input_names = (
        "c_tool_usd",
        "c_mfg_usd",
        "c_qc_usd",
        "c_mm_usd",
        "c_ep_usd",
        "c_subtotal_usd",
        "c_stealth_usd",
        "c_software_usd",
        "quantity",
        "avionics_factor",
        "investment_factor",
        "escalation_factor",
    )
    output_names = (
        "c_avionics_usd",
        "c_invest_usd",
        "c_stealth_program_usd",
        "c_total_base_usd",
        "c_program_usd",
        "c_unit_usd",
        "c_recurring_unit_usd",
        "c_avg_flyaway_usd",
        "c_nre_usd",
    )

    @staticmethod
    def primal_fn(x):
        costs = {
            "c_tool_usd": x[0],
            "c_mfg_usd": x[1],
            "c_qc_usd": x[2],
            "c_mm_usd": x[3],
            "c_ep_usd": x[4],
            "c_subtotal_usd": x[5],
        }
        out = M.brandt_program_cost(
            costs=costs,
            quantity=x[8],
            avionics_factor=x[9],
            investment_factor=x[10],
            escalation_factor=x[11],
            stealth_usd_per_airframe=x[6],
            software_usd=x[7],
        )
        return jnp.stack([out[name] for name in ProgramCost.output_names])


class OperatingCost(JaxExplicitComponent):
    """Brandt annual and life operations & maintenance for one aircraft."""

    input_names = (
        "design_mission_fuel_lb",
        "design_mission_time_hr",
        "average_mission_factor",
        "flight_hours_per_year",
        "fuel_usd_per_lb",
        "crew_ratio",
        "crew_hours_per_year",
        "rate_engineering_usd_per_hr",
        "maintenance_hours_per_flight_hour",
        "rate_manufacturing_usd_per_hr",
        "escalation_factor",
        "life_years",
    )
    output_names = (
        "c_annual_fuel_usd",
        "c_annual_crew_usd",
        "c_annual_maint_usd",
        "c_om_annual_usd",
        "c_om_life_usd",
    )

    @staticmethod
    def primal_fn(x):
        out = M.brandt_operating_cost(
            design_mission_fuel_lb=x[0],
            design_mission_time_hr=x[1],
            average_mission_factor=x[2],
            flight_hours_per_year=x[3],
            fuel_usd_per_lb=x[4],
            crew_ratio=x[5],
            crew_hours_per_year=x[6],
            engineering_rate_usd_per_hr=x[7],
            maintenance_hours_per_flight_hour=x[8],
            manufacturing_rate_usd_per_hr=x[9],
            escalation_factor=x[10],
            life_years=x[11],
        )
        return jnp.stack(
            [
                out["c_annual_fuel_usd"],
                out["c_annual_crew_usd"],
                out["c_annual_maint_usd"],
                out["c_om_annual_usd"],
                out["c_om_life_usd"],
            ]
        )


class LifeCycleCost(JaxExplicitComponent):
    """Life-cycle cost per aircraft, and the share of it that is ownership."""

    input_names = ("c_om_life_usd", "c_unit_usd")
    output_names = ("c_lcc_usd", "ownership_share")

    @staticmethod
    def primal_fn(x):
        lcc = M.life_cycle_cost(x[0], x[1])
        return jnp.stack([lcc, x[0] / lcc])


class ReportedCosts(JaxExplicitComponent):
    """Put every headline figure in one stated dollar year.

    A cost is meaningless without its dollar year, and this model mixes them: the
    Brandt wrap rates are 1999 dollars escalated to 2006, Raymer's other set is
    2012.  Rather than ask the reader to keep track, everything reported is moved
    to a single ``report_dollar_year`` with Roskam's cost escalation factor.
    """

    input_names = (
        "c_recurring_unit_usd",
        "c_avg_flyaway_usd",
        "c_unit_usd",
        "c_nre_usd",
        "c_program_usd",
        "c_om_life_usd",
        "c_lcc_usd",
        "model_dollar_year",
        "report_dollar_year",
    )
    output_names = (
        "dollar_year_scale",
        "reported_recurring_unit_usd",
        "reported_avg_flyaway_usd",
        "reported_unit_usd",
        "reported_nre_usd",
        "reported_program_usd",
        "reported_om_life_usd",
        "reported_lcc_usd",
    )

    @staticmethod
    def primal_fn(x):
        scale = M.roskam_cef(x[8]) / M.roskam_cef(x[7])
        return jnp.stack([scale, *[x[i] * scale for i in range(7)]])


class SoftwareCost(JaxExplicitComponent):
    """COCOMO II software development cost, and its share per aircraft.

    This is the path from source lines of code to dollars.  It is not a rounding
    error: on a modern combat aircraft software development can run to a third of
    the entire RDT&E bill.
    """

    input_names = (
        "ksloc",
        "scale_factor_sum",
        "effort_multiplier_product",
        "usd_per_person_month",
        "quantity",
    )
    output_names = (
        "software_exponent",
        "effort_person_months",
        "devtime_months",
        "c_software_usd",
        "c_software_per_aircraft_usd",
    )

    @staticmethod
    def primal_fn(x):
        out = M.cocomo_ii(x[0], x[1], x[2])
        cost = M.software_cost(out["effort_pm"], x[3])
        return jnp.stack(
            [out["exponent"], out["effort_pm"], out["devtime_months"], cost, cost / x[4]]
        )


class StealthTreatmentCost(JaxExplicitComponent):
    """Low-observables treatment cost per airframe (Brandt Cost!A114:G167).

    Escalated from the 2005 rates to the dollar year of ``coefficient_set``, so it
    adds to the DAPCA subtotal in the same dollars.
    """

    input_names = (
        "skin_wetted_area_ft2",
        "inlet_lip_length_ft",
        "inlet_duct_area_ft2",
        "treated_edge_length_ft",
        "vertical_tail_area_ft2",
        "vertical_tail_outer_area_ft2",
        "hinge_line_length_ft",
        "radar_bulkhead_area_ft2",
        "access_panel_perimeter_ft",
        "exhaust_area_ft2",
    )
    output_names = ("c_stealth_usd",)
    const_names = ("signature_level", "coefficient_set")

    def initialize(self) -> None:
        super().initialize()
        self.options.declare("signature_level", default="B", values=M.LO_LEVELS)
        self.options.declare("coefficient_set", default="1999", values=("1999", "2012"))

    @staticmethod
    def primal_fn(x, signature_level, coefficient_set):
        return jnp.stack(
            [
                M.stealth_treatment_cost(
                    level=signature_level,
                    skin_wetted_area_ft2=x[0],
                    inlet_lip_length_ft=x[1],
                    inlet_duct_area_ft2=x[2],
                    treated_edge_length_ft=x[3],
                    vertical_tail_area_ft2=x[4],
                    vertical_tail_outer_area_ft2=x[5],
                    hinge_line_length_ft=x[6],
                    radar_bulkhead_area_ft2=x[7],
                    access_panel_perimeter_ft=x[8],
                    exhaust_area_ft2=x[9],
                    dollar_year=float(coefficient_set),
                )
            ]
        )


# --------------------------------------------------------------------------- #
# Military -- Roskam Part VIII, the whole-programme alternative to DAPCA
# --------------------------------------------------------------------------- #
class AmprWeight(JaxExplicitComponent):
    """Takeoff weight -> AMPR weight, Roskam's regression."""

    input_names = ("w_to_lb",)
    output_names = ("w_ampr_lb",)

    @staticmethod
    def primal_fn(x):
        return jnp.stack([M.ampr_weight_from_takeoff(x[0])])


class CostEscalation(JaxExplicitComponent):
    """Calendar year -> Roskam cost escalation factor, and the escalated wrap rates.

    Making the year of economics an input rather than a buried constant is the
    difference between a model you can re-baseline and one you cannot.
    """

    input_names = (
        "year_of_economics",
        "rate_engineering_1989_usd_per_hr",
        "rate_manufacturing_1989_usd_per_hr",
        "rate_tooling_1989_usd_per_hr",
        "rate_maintenance_1989_usd_per_hr",
        "rate_consumables_1989_usd_per_hr",
    )
    output_names = (
        "cef",
        "roskam_rate_engineering_usd_per_hr",
        "roskam_rate_manufacturing_usd_per_hr",
        "roskam_rate_tooling_usd_per_hr",
        "roskam_rate_maintenance_usd_per_hr",
        "roskam_rate_consumables_usd_per_hr",
    )

    @staticmethod
    def primal_fn(x):
        cef = M.roskam_cef(x[0])
        scale = cef / M.roskam_cef(1989.0)
        return jnp.stack([cef, x[1] * scale, x[2] * scale, x[3] * scale, x[4] * scale, x[5] * scale])


class RoskamRdte(JaxExplicitComponent):
    """Roskam Ch. 3 research, development, test and evaluation cost."""

    input_names = (
        "w_ampr_lb",
        "v_max_keas",
        "n_rdte",
        "n_static_test",
        "roskam_rate_engineering_usd_per_hr",
        "roskam_rate_manufacturing_usd_per_hr",
        "roskam_rate_tooling_usd_per_hr",
        "cef",
        "roskam_engine_cost_usd",
        "engines_per_aircraft",
        "roskam_avionics_cost_usd",
        "rdte_production_rate_per_month",
        "difficulty_factor",
        "cad_factor",
        "material_factor_roskam",
        "observables_factor",
        "c_software_usd",
    )
    output_names = (
        "c_aed_usd",
        "c_dst_usd",
        "c_fta_usd",
        "c_fto_usd",
        "c_man_rdte_usd",
        "c_mat_rdte_usd",
        "c_tool_rdte_usd",
        "c_tsf_usd",
        "c_pro_rdte_usd",
        "c_fin_rdte_usd",
        "c_rdte_usd",
    )
    const_names = ("rdte_overhead_fractions",)

    def initialize(self) -> None:
        super().initialize()
        # (test and simulation facilities, profit, financing)
        self.options.declare("rdte_overhead_fractions", default=(0.1, 0.1, 0.1))

    @staticmethod
    def primal_fn(x, rdte_overhead_fractions):
        tsf, profit, finance = rdte_overhead_fractions
        out = M.roskam_rdte_cost(
            w_ampr_lb=x[0],
            v_max_keas=x[1],
            n_rdte=x[2],
            n_static_test=x[3],
            engineering_rate_usd_per_hr=x[4],
            manufacturing_rate_usd_per_hr=x[5],
            tooling_rate_usd_per_hr=x[6],
            cef=x[7],
            engine_cost_usd=x[8],
            engines_per_aircraft=x[9],
            avionics_cost_usd=x[10],
            rdte_production_rate_per_month=x[11],
            difficulty_factor=x[12],
            cad_factor=x[13],
            material_factor_roskam=x[14],
            observables_factor=x[15],
            software_cost_usd=x[16],
            test_facilities_fraction=tsf,
            profit_fraction=profit,
            finance_fraction=finance,
        )
        return jnp.stack([out[name] for name in RoskamRdte.output_names])


class RoskamOperating(JaxExplicitComponent):
    """Roskam Ch. 6 programme operating cost over the service life."""

    input_names = (
        "n_production",
        "reserve_fraction",
        "loss_rate_per_hour",
        "utilization_hr_per_year",
        "service_years",
        "mission_fuel_lb",
        "mission_time_hr",
        "fuel_usd_per_gal",
        "fuel_density_lb_per_gal",
        "oil_lubricant_factor",
        "crew_per_aircraft",
        "crew_ratio",
        "crew_pay_usd_per_year",
        "crew_overhead_factor",
        "maintenance_hours_per_flight_hour",
        "roskam_rate_maintenance_usd_per_hr",
        "roskam_rate_consumables_usd_per_hr",
    )
    output_names = (
        "n_service",
        "c_pol_usd",
        "c_crew_usd",
        "c_maint_personnel_usd",
        "c_consumables_usd",
        "c_indirect_personnel_usd",
        "c_spares_usd",
        "c_depot_usd",
        "c_misc_usd",
        "c_ops_usd",
        "c_ops_per_hour_usd",
    )
    const_names = ("ops_overhead_fractions",)

    def initialize(self) -> None:
        super().initialize()
        # (indirect personnel, spares, depot, miscellaneous)
        self.options.declare("ops_overhead_fractions", default=(0.2, 0.125, 0.15, 0.055))

    @staticmethod
    def primal_fn(x, ops_overhead_fractions):
        indirect, spares, depot, misc = ops_overhead_fractions
        out = M.roskam_operating_cost(
            n_acquired=x[0],
            reserve_fraction=x[1],
            loss_rate_per_hour=x[2],
            utilization_hr_per_year=x[3],
            service_years=x[4],
            mission_fuel_lb=x[5],
            mission_time_hr=x[6],
            fuel_price_usd_per_gal=x[7],
            fuel_density_lb_per_gal=x[8],
            oil_lubricant_factor=x[9],
            crew_per_aircraft=x[10],
            crew_ratio=x[11],
            crew_pay_usd_per_year=x[12],
            crew_overhead_factor=x[13],
            maintenance_hours_per_flight_hour=x[14],
            maintenance_rate_usd_per_hr=x[15],
            consumable_rate_usd_per_hr=x[16],
            indirect_personnel_fraction=indirect,
            spares_fraction=spares,
            depot_fraction=depot,
            misc_fraction=misc,
        )
        return jnp.stack([out[name] for name in RoskamOperating.output_names])


class RoskamAcquisition(JaxExplicitComponent):
    """Roskam Ch. 4 manufacturing and acquisition, and Ch. 2/7 life-cycle cost."""

    input_names = (
        "w_ampr_lb",
        "v_max_keas",
        "n_production",
        "n_rdte",
        "c_aed_usd",
        "c_man_rdte_usd",
        "c_mat_rdte_usd",
        "c_tool_rdte_usd",
        "c_rdte_usd",
        "roskam_rate_engineering_usd_per_hr",
        "roskam_rate_manufacturing_usd_per_hr",
        "roskam_rate_tooling_usd_per_hr",
        "cef",
        "roskam_engine_cost_usd",
        "engines_per_aircraft",
        "roskam_avionics_cost_usd",
        "production_rate_per_month",
        "difficulty_factor",
        "cad_factor",
        "material_factor_roskam",
        "c_ops_per_hour_usd",
        "c_ops_usd",
    )
    output_names = (
        "c_aed_m_usd",
        "c_apc_m_usd",
        "c_fto_m_usd",
        "c_fin_m_usd",
        "c_man_usd",
        "c_profit_usd",
        "c_acq_usd",
        "aep_usd",
        "c_roskam_lcc_usd",
        "c_disposal_usd",
    )
    const_names = ("disposal_fraction",)

    def initialize(self) -> None:
        super().initialize()
        self.options.declare("disposal_fraction", default=0.01)

    @staticmethod
    def primal_fn(x, disposal_fraction):
        rdte_costs = {
            "c_aed_usd": x[4],
            "c_man_rdte_usd": x[5],
            "c_mat_rdte_usd": x[6],
            "c_tool_rdte_usd": x[7],
            "c_rdte_usd": x[8],
        }
        acq = M.roskam_acquisition_cost(
            w_ampr_lb=x[0],
            v_max_keas=x[1],
            n_production=x[2],
            n_rdte=x[3],
            rdte_costs=rdte_costs,
            engineering_rate_usd_per_hr=x[9],
            manufacturing_rate_usd_per_hr=x[10],
            tooling_rate_usd_per_hr=x[11],
            cef=x[12],
            engine_cost_usd=x[13],
            engines_per_aircraft=x[14],
            avionics_cost_usd=x[15],
            production_rate_per_month=x[16],
            difficulty_factor=x[17],
            cad_factor=x[18],
            material_factor_roskam=x[19],
            operating_cost_per_hour_usd=x[20],
        )
        lcc = M.roskam_life_cycle_cost(x[8], acq["c_acq_usd"], x[21], disposal_fraction)
        return jnp.stack(
            [
                acq["c_aed_m_usd"],
                acq["c_apc_m_usd"],
                acq["c_fto_m_usd"],
                acq["c_fin_m_usd"],
                acq["c_man_usd"],
                acq["c_profit_usd"],
                acq["c_acq_usd"],
                acq["aep_usd"],
                lcc["c_lcc_usd"],
                lcc["c_disposal_usd"],
            ]
        )


# --------------------------------------------------------------------------- #
# Commercial -- the twelve elements of AIAA 2025-3499 Eq. 1
# --------------------------------------------------------------------------- #
class OwnershipCost(JaxExplicitComponent):
    """Depreciation, battery depreciation, interest and insurance (Eqs. 2-5)."""

    input_names = (
        "airframe_cost_usd",
        "engine_cost_usd",
        "n_engines",
        "motor_cost_usd",
        "n_motors",
        "residual_value_fraction",
        "financial_life_years",
        "utilization_hr_per_year",
        "block_time_hr",
        "battery_cost_usd",
        "battery_residual_fraction",
        "battery_reference_cycles",
        "charge_c_rate",
        "aircraft_cost_usd",
        "annual_interest_rate",
        "loan_years",
        "insurance_premium_fraction",
        "electrified_premium_increase",
    )
    output_names = (
        "c_depreciation_usd",
        "c_battery_depreciation_usd",
        "c_interest_usd",
        "c_insurance_usd",
    )
    const_names = ("payments_per_year",)

    def initialize(self) -> None:
        super().initialize()
        self.options.declare("payments_per_year", default=12.0)

    @staticmethod
    def primal_fn(x, payments_per_year):
        depreciation = C.depreciation_cost(
            airframe_cost_usd=x[0],
            engine_cost_usd=x[1],
            n_engines=x[2],
            motor_cost_usd=x[3],
            n_motors=x[4],
            residual_value_fraction=x[5],
            financial_life_years=x[6],
            utilization_hr_per_year=x[7],
            block_time_hr=x[8],
        )
        battery = C.battery_depreciation_cost(x[9], x[10], x[11], x[12])
        interest = C.interest_cost(x[13], x[14], x[15], payments_per_year, x[7], x[8])
        insurance = C.insurance_cost(x[13], x[16], x[17], x[7], x[8])
        return jnp.stack([depreciation, battery, interest, insurance])


class CrewCost(JaxExplicitComponent):
    """Flight and cabin crew for one trip (Eqs. 6-7)."""

    input_names = (
        "flight_crew_usd_per_hr",
        "cabin_crew_usd_per_hr",
        "international_factor",
        "n_cabin_crew",
        "block_time_hr",
    )
    output_names = ("c_flight_crew_usd", "c_cabin_crew_usd")

    @staticmethod
    def primal_fn(x):
        return jnp.stack(
            [
                C.flight_crew_cost(x[0], x[4]),
                C.cabin_crew_cost(x[1], x[2], x[3], x[4]),
            ]
        )


class EnergyCost(JaxExplicitComponent):
    """Fuel, electricity and the carbon allowance for one trip (Eqs. 8, 9, 15)."""

    input_names = (
        "fuel_mass_kg",
        "fuel_density_kg_per_gal",
        "fuel_usd_per_gal",
        "saf_blend",
        "saf_price_factor",
        "mission_energy_kwh",
        "electricity_usd_per_kwh",
        "carbon_price_usd_per_ton",
        "carbon_taxed_indicator",
    )
    output_names = ("c_fuel_usd", "c_electricity_usd", "c_carbon_usd")

    @staticmethod
    def primal_fn(x):
        return jnp.stack(
            [
                C.fuel_cost(x[0], x[1], x[2], x[3], x[4]),
                C.electricity_cost(x[5], x[6]),
                C.carbon_tax_cost(x[0], x[7], x[3], x[8]),
            ]
        )


class MaintenanceCost(JaxExplicitComponent):
    """Airframe and powerplant maintenance for one trip (Eqs. 10-12)."""

    input_names = (
        "oew_kg",
        "thrust_per_engine_kn",
        "n_engines",
        "n_motors",
        "maintenance_reduction_factor",
        "block_time_hr",
    )
    output_names = ("c_maint_airframe_usd", "c_maint_per_engine_usd", "c_maint_usd")

    @staticmethod
    def primal_fn(x):
        out = C.maintenance_cost(x[0], x[1], x[2], x[3], x[4], x[5])
        return jnp.stack(
            [out["c_maint_airframe_usd"], out["c_maint_per_engine_usd"], out["c_maint_usd"]]
        )


class FeeCost(JaxExplicitComponent):
    """Terminal and en-route charges for one trip (Eqs. 13-14)."""

    input_names = (
        "tans_usd_per_ton",
        "en_route_usd_per_ton_km",
        "stage_length_km",
        "mtom_ton",
    )
    output_names = ("c_landing_usd", "c_navigation_usd")

    @staticmethod
    def primal_fn(x):
        return jnp.stack([C.landing_fee(x[0], x[3]), C.navigation_fee(x[1], x[2], x[3])])


class DocTotal(JaxExplicitComponent):
    """Sum the twelve elements and report the unit cost per seat-kilometre (Eq. 1)."""

    input_names = (
        "c_depreciation_usd",
        "c_battery_depreciation_usd",
        "c_interest_usd",
        "c_insurance_usd",
        "c_flight_crew_usd",
        "c_cabin_crew_usd",
        "c_fuel_usd",
        "c_electricity_usd",
        "c_maint_usd",
        "c_landing_usd",
        "c_navigation_usd",
        "c_carbon_usd",
        "n_passengers",
        "stage_length_km",
    )
    output_names = ("doc_usd", "unit_cost_usd_per_ask")

    @staticmethod
    def primal_fn(x):
        doc = jnp.sum(x[:12])
        return jnp.stack([doc, C.unit_cost_per_ask(doc, x[12], x[13])])
