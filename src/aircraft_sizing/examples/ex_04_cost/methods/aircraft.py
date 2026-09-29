"""Baseline aircraft: the input sets the app and the docs start from.

Three military aircraft, each with **both** a DAPCA IV input set and a Roskam Part
VIII input set, so either model can be run on any of them and the two compared:

- **F-16A** -- read from ``Brandt-F16-A.xls`` sheet ``Cost``, reproduced exactly.
- **Silver Scythe** -- a student senior-design carrier-based stealth strike
  fighter, re-run through the *published* equations rather than the programme's
  own transcription of them.
- **Project SPEAR** -- a student senior-design uninhabited strike aircraft, same
  treatment.

Two commercial aircraft from Table 13 of AIAA 2025-3499: the conventional ATR
72-500 and the hybrid-electric Do328HE.

Separately, :data:`VALIDATION_AIRCRAFT` holds five real aircraft used only to check
that the model lands in the right neighbourhood; their production quantities and
avionics factors are estimates, not procurement records.

Nothing here is a cost quotation.  A CER fitted to 1970s fighter programmes and
escalated forty years is an order-of-magnitude tool.
"""

from __future__ import annotations

import math

from .config import CommercialCostInputs, MilitaryCostInputs, RoskamCostInputs

__all__ = [
    "MILITARY_AIRCRAFT",
    "VALIDATION_AIRCRAFT",
    "COMMERCIAL_BASELINES",
    "COMMERCIAL_MISSIONS",
    "SCENARIOS",
    "PUBLISHED_FLYAWAY",
    "CPI_U",
    "GROUND_TIME_HR",
    "REPORT_DOLLAR_YEAR",
    "commercial_inputs_for",
    "escalate_published",
    "military_names",
    "commercial_names",
]

#: Everything the app reports is stated in this dollar year.
REPORT_DOLLAR_YEAR = 2026.0


def _keas(mach: float, altitude_ft: float = 30_000.0) -> float:
    """Equivalent airspeed in knots from a Mach number at altitude.

    Roskam's cost CERs take maximum level speed in KEAS.  ``V_EAS = M * a *
    sqrt(sigma)`` with the standard-atmosphere speed of sound and density ratio;
    at 30,000 ft that is 589.3 kt and 0.3741.
    """
    speed_of_sound_kts = 589.323
    density_ratio = 0.3741
    del altitude_ft  # only 30,000 ft is tabulated here
    return mach * speed_of_sound_kts * math.sqrt(density_ratio)


# --------------------------------------------------------------------------- #
# Military -- three aircraft, each with a DAPCA and a Roskam input set
# --------------------------------------------------------------------------- #
MILITARY_AIRCRAFT = {
    "F-16A": {
        # The workbook itself carries no software line; a representative early-F-16
        # code size and person-month rate are priced in here so the software
        # sliders are live.  Set the rate to zero to recover the sheet exactly.
        "dapca": MilitaryCostInputs(ksloc=200.0, usd_per_person_month=25_000.0),
        "roskam": RoskamCostInputs(
            w_to_lb=31_377.0,          # Brandt Wt!B3
            v_max_keas=_keas(2.0),
            n_production=200.0,
            engines_per_aircraft=1.0,
            difficulty_factor=1.0,     # 1970s technology, no stretch
            cad_factor=1.0,            # manual drafting
            material_factor_roskam=1.2,
            observables_factor=1.0,    # not a low-observable aircraft
            year_of_economics=2026.0,
            roskam_engine_cost_usd=3.0e6,
            roskam_avionics_cost_usd=2.5e6,
            utilization_hr_per_year=500.0,
            service_years=20.0,
            mission_fuel_lb=6_000.0,
            mission_time_hr=1.57,
            maintenance_hours_per_flight_hour=15.0,
            ksloc=200.0,
        ),
        "note": "Brandt workbook reference case. Signature 'none' and a zero person-month rate reproduce the sheet exactly.",
    },
    "Silver Scythe": {
        # Gobbler Ghost, a carrier-based stealth strike fighter.  Empty weight,
        # Mach, quantity and flight-test count are the programme's own; the
        # coefficients, rates and equations are the published ones.
        "dapca": MilitaryCostInputs(
            we_lb=33_970.0,
            mach_max=1.65,
            quantity=500.0,
            flight_test_aircraft=4.0,
            engines_per_aircraft=2.0,
            thrust_max_lbf=22_000.0,   # F414-GE-400, afterburning
            turbine_inlet_temp_r=3_200.0,
            aluminium_percent=40.0,
            carbon_fibre_percent=45.0,
            fibreglass_percent=0.0,
            steel_percent=5.0,
            titanium_percent=10.0,
            rate_engineering_usd_per_hr=115.0,
            rate_tooling_usd_per_hr=118.0,
            rate_manufacturing_usd_per_hr=98.0,
            rate_quality_usd_per_hr=108.0,
            avionics_factor=0.25,
            investment_factor=0.0,
            escalation_factor=0.0,
            coefficient_set="2012",
            signature_level="C",
            model_dollar_year=2012.0,
            design_mission_fuel_lb=20_300.0,
            design_mission_time_hr=2.5,
            flight_hours_per_year=200.0,
            maintenance_hours_per_flight_hour=12.5,
            life_years=30.0,
            ksloc=8_000.0,
            usd_per_person_month=25_000.0,
            skin_wetted_area_ft2=2_380.52,
        ),
        "roskam": RoskamCostInputs(
            w_to_lb=58_372.0,
            v_max_keas=_keas(1.64),
            n_production=500.0,
            engines_per_aircraft=2.0,
            difficulty_factor=1.3,
            cad_factor=0.8,
            material_factor_roskam=1.75,
            observables_factor=1.3,
            year_of_economics=2026.0,
            roskam_engine_cost_usd=10.0e6,
            roskam_avionics_cost_usd=13.25e6,
            utilization_hr_per_year=200.0,
            service_years=30.0,
            mission_fuel_lb=20_300.0,
            mission_time_hr=2.5,
            maintenance_hours_per_flight_hour=12.5,
            ksloc=8_000.0,
        ),
        "note": "Senior-design stealth strike fighter, run through the published equations.",
    },
    "Project SPEAR": {
        # An uninhabited strike aircraft.  Weights and speed are the programme's.
        "dapca": MilitaryCostInputs(
            we_lb=12_674.0,
            mach_max=1.39,             # 794 kt at sea level
            quantity=1_000.0,
            flight_test_aircraft=6.0,
            engines_per_aircraft=1.0,
            thrust_max_lbf=16_000.0,
            turbine_inlet_temp_r=3_200.0,
            aluminium_percent=30.0,
            carbon_fibre_percent=55.0,
            fibreglass_percent=0.0,
            steel_percent=5.0,
            titanium_percent=10.0,
            rate_engineering_usd_per_hr=115.0,
            rate_tooling_usd_per_hr=118.0,
            rate_manufacturing_usd_per_hr=98.0,
            rate_quality_usd_per_hr=108.0,
            avionics_factor=0.2,
            investment_factor=0.0,
            escalation_factor=0.0,
            coefficient_set="2012",
            signature_level="C",
            model_dollar_year=2012.0,
            design_mission_fuel_lb=13_400.0,
            design_mission_time_hr=5.2,
            flight_hours_per_year=200.0,
            crew_ratio=0.0,            # uninhabited: no aircrew
            crew_hours_per_year=0.0,
            maintenance_hours_per_flight_hour=30.0,
            life_years=20.0,
            ksloc=15_000.0,
            usd_per_person_month=25_000.0,
            skin_wetted_area_ft2=1_100.0,
        ),
        "roskam": RoskamCostInputs(
            w_to_lb=27_754.0,
            v_max_keas=793.768899,     # programme's own max level speed at sea level
            n_production=1_000.0,
            n_rdte=8.0,
            n_static_test=2.0,
            engines_per_aircraft=1.0,
            difficulty_factor=1.85,
            cad_factor=0.9,
            material_factor_roskam=2.0,
            observables_factor=2.6,
            year_of_economics=2026.0,
            roskam_engine_cost_usd=4.479e6,
            roskam_avionics_cost_usd=3.7e6,
            utilization_hr_per_year=200.0,
            service_years=20.0,
            mission_fuel_lb=13_400.0,
            mission_time_hr=5.2,
            maintenance_hours_per_flight_hour=30.0,
            crew_per_aircraft=1.0,     # remote crew
            ksloc=15_000.0,
        ),
        "note": "Senior-design uninhabited strike aircraft, run through the published equations.",
    },
}


def military_names() -> tuple:
    """Aircraft names for the military selector, reference case first."""
    return tuple(MILITARY_AIRCRAFT)


# --------------------------------------------------------------------------- #
# Five real aircraft, used only to check the model lands in the right place
# --------------------------------------------------------------------------- #
VALIDATION_AIRCRAFT = {
    "F-16A": MilitaryCostInputs.baseline(),
    "F-15C Eagle": MilitaryCostInputs(
        we_lb=28_600.0, mach_max=2.5, quantity=483.0, flight_test_aircraft=6.0,
        engines_per_aircraft=2.0, thrust_max_lbf=23_770.0, turbine_inlet_temp_r=3_400.0,
        aluminium_percent=60.0, carbon_fibre_percent=5.0, fibreglass_percent=0.0,
        steel_percent=10.0, titanium_percent=25.0, avionics_factor=0.45,
    ),
    "F/A-18E Super Hornet": MilitaryCostInputs(
        we_lb=32_081.0, mach_max=1.8, quantity=500.0, flight_test_aircraft=7.0,
        engines_per_aircraft=2.0, thrust_max_lbf=22_000.0, turbine_inlet_temp_r=3_200.0,
        aluminium_percent=45.0, carbon_fibre_percent=25.0, fibreglass_percent=0.0,
        steel_percent=10.0, titanium_percent=20.0, avionics_factor=0.5,
    ),
    "A-10A Thunderbolt II": MilitaryCostInputs(
        we_lb=24_959.0, mach_max=0.75, quantity=713.0, flight_test_aircraft=6.0,
        engines_per_aircraft=2.0, thrust_max_lbf=9_065.0, turbine_inlet_temp_r=2_600.0,
        aluminium_percent=85.0, carbon_fibre_percent=0.0, fibreglass_percent=0.0,
        steel_percent=10.0, titanium_percent=5.0, avionics_factor=0.2,
        maintenance_hours_per_flight_hour=10.0,
    ),
    "T-38A Talon": MilitaryCostInputs(
        we_lb=7_200.0, mach_max=1.3, quantity=1_187.0, flight_test_aircraft=4.0,
        engines_per_aircraft=2.0, thrust_max_lbf=2_900.0, turbine_inlet_temp_r=2_400.0,
        aluminium_percent=90.0, carbon_fibre_percent=0.0, fibreglass_percent=0.0,
        steel_percent=8.0, titanium_percent=2.0, avionics_factor=0.1,
        maintenance_hours_per_flight_hour=6.0, ksloc=200.0,
    ),
}

#: Published unit **flyaway** costs, their dollar year, the number actually built,
#: and engines per aircraft.  Sources are USAF/US Navy fact sheets and the standard
#: public references.
#:
#: Flyaway cost is the right comparison for ``c_avg_flyaway_usd``.  Comparing it
#: against ``c_unit_usd`` -- which carries the entire non-recurring bill -- is the
#: mistake this table exists to prevent.
PUBLISHED_FLYAWAY = {
    "F-16A": (14.6e6, 1998, 4_600.0, 1.0),
    "F-15C Eagle": (29.9e6, 1998, 1_198.0, 2.0),
    "F/A-18E Super Hornet": (57.5e6, 2009, 500.0, 2.0),
    "A-10A Thunderbolt II": (18.8e6, 1998, 716.0, 2.0),
    "T-38A Talon": (0.756e6, 1961, 1_187.0, 2.0),
}

#: US CPI-U annual average, for putting published costs in a common dollar year.
CPI_U = {1961: 29.9, 1998: 163.0, 2006: 201.6, 2009: 214.5, 2012: 229.6, 2025: 322.3}


def escalate_published(cost_usd: float, from_year: int, to_year: int) -> float:
    """Move a published cost between dollar years with the CPI."""
    return cost_usd * CPI_U[to_year] / CPI_U[from_year]


# --------------------------------------------------------------------------- #
# Commercial -- AIAA 2025-3499 Table 13
# --------------------------------------------------------------------------- #
#: Shaft horsepower to kilowatts.
_SHP_TO_KW = 0.7457
#: Taxi, climb and descent allowance added to the cruise leg, our assumption.
GROUND_TIME_HR = 0.35

#: name -> (block speed [km/h], cruise TAS [km/h], shaft power per engine [shp],
#: cruise fuel burn [kg/h], electrical energy per km [kWh/km]).
#:
#: The paper publishes no block-speed or fuel-burn model, so these are from
#: published performance data.  Block speed sets block time; cruise TAS sets the
#: equivalent thrust in Eq. 12.  Table 13's 900 kWh is taken over a 1,000 km
#: mission, hence 0.9 kWh/km.
COMMERCIAL_MISSIONS = {
    "ATR 72-500": (420.0, 509.0, 1_940.0, 630.0, 0.0),
    "Do328HE (hybrid-electric)": (440.0, 540.0, 274.0, 150.0, 0.9),
}

COMMERCIAL_BASELINES = {
    "ATR 72-500": CommercialCostInputs.baseline(),
    # Table 13 gives no battery capacity, so battery cost is zero.
    "Do328HE (hybrid-electric)": CommercialCostInputs(
        aircraft_cost_usd=14.2e6,
        airframe_cost_usd=13.9e6,
        engine_cost_usd=70_000.0,
        n_engines=1.0,
        motor_cost_usd=80_500.0,
        n_motors=2.0,
        n_passengers=32.0,
        n_cabin_crew=1.0,
        mtom_ton=14.0,
        oew_kg=10_600.0,
        electrified_premium_increase=0.063,
        carbon_taxed_indicator=0.0,
        # 1,000 km mission, as derived by ``commercial_inputs_for``.
        block_time_hr=2.622727272727273,
        fuel_mass_kg=393.4090909090909,
        thrust_per_engine_kn=1.362145333333333,
        mission_energy_kwh=900.0,
    ),
}

#: AIAA 2025-3499 Table 14 -- the three scenarios, as (SAF blend, SAF price factor,
#: carbon allowance price, maintenance reduction factor for electric motors,
#: battery charging C-rate).
SCENARIOS = {
    "2030": (0.06, 3.00, 70.0, 0.25, 1.26),
    "2040": (0.34, 2.24, 130.0, 0.50, 2.0),
    "2050": (0.70, 2.24, 500.0, 0.75, 2.0),
}


def commercial_inputs_for(name: str, stage_length_km: float, scenario: str = "2030"):
    """Build a :class:`CommercialCostInputs` for a baseline on a given stage.

    Block time is the stage length over the aircraft's **block** speed plus a fixed
    ground allowance; the equivalent thrust in Eq. 12 uses **cruise** TAS.  Those,
    the fuel burn and the electrical energy are stated in
    :data:`COMMERCIAL_MISSIONS` because the paper does not publish the model behind
    its own figures.
    """
    if name not in COMMERCIAL_BASELINES:
        raise KeyError(f"Unknown commercial baseline {name!r}.")
    if scenario not in SCENARIOS:
        raise KeyError(f"Unknown scenario {scenario!r}; expected one of {tuple(SCENARIOS)}.")

    (block_speed_km_h, cruise_speed_km_h, shaft_power_shp, fuel_burn_kg_per_hr,
     energy_kwh_per_km) = COMMERCIAL_MISSIONS[name]
    block_time_hr = GROUND_TIME_HR + stage_length_km / block_speed_km_h
    thrust_kn = shaft_power_shp * _SHP_TO_KW / (cruise_speed_km_h / 3.6)

    saf_blend, saf_price_factor, carbon_price, maintenance_reduction, c_rate = SCENARIOS[scenario]
    base = COMMERCIAL_BASELINES[name]
    derived = (
        "block_time_hr",
        "stage_length_km",
        "fuel_mass_kg",
        "mission_energy_kwh",
        "thrust_per_engine_kn",
        "saf_blend",
        "saf_price_factor",
        "carbon_price_usd_per_ton",
        "maintenance_reduction_factor",
        "charge_c_rate",
    )
    return CommercialCostInputs(
        **{field: getattr(base, field) for field in base.__slots__ if field not in derived},
        block_time_hr=block_time_hr,
        stage_length_km=stage_length_km,
        fuel_mass_kg=fuel_burn_kg_per_hr * block_time_hr,
        mission_energy_kwh=energy_kwh_per_km * stage_length_km,
        thrust_per_engine_kn=thrust_kn,
        saf_blend=saf_blend,
        saf_price_factor=saf_price_factor,
        carbon_price_usd_per_ton=carbon_price,
        maintenance_reduction_factor=(maintenance_reduction if base.n_motors > 0.0 else 0.0),
        charge_c_rate=c_rate,
    )


def commercial_names() -> tuple:
    """Aircraft names for the commercial selector."""
    return tuple(COMMERCIAL_BASELINES)
