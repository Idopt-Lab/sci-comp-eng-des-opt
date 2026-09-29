"""Pure-JAX direct operating cost (DOC) model for passenger transport aircraft.

Implements the model proposed in Espinosa-Juarez, Jouannet, Amadori and Sanchez
Mata, "Comparative Analysis on Aircraft Direct Operating Cost Models", AIAA
AVIATION 2025, DOI 10.2514/6.2025-3499, Eqs. 1-15.  Equation numbers are cited per
function.

The paper surveys nine DOC models and then proposes its own, cherry-picking the
best-matching element from each: depreciation and airframe maintenance from
Jenkinson, engine maintenance from TU Berlin, and new terms for battery
depreciation, amortised interest, SAF blending and carbon tax so electrified
concepts can be compared with conventional ones.

**Block time, fuel mass and mission energy are inputs, not outputs.**  The paper
does not publish the block-speed or fuel-burn model behind its results, so this
module takes ``t_b``, ``m_fuel`` and ``E_mission`` from the caller.  That keeps
every cost element exactly reproducible and puts the mission assumptions where a
student can see and change them.
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402  (import after enabling x64 on purpose)

#: Consumer price index ratios the paper uses to bring older CERs to 2025 dollars
#: (Eqs. 11 and 12).  Jenkinson's airframe maintenance fit is 1994; TU Berlin's
#: engine maintenance fit is 2010.
CPI_2025_OVER_1994 = 322.3 / 148.2
CPI_2025_OVER_2010 = 322.3 / 218.1

#: Fuel densities in kg per US gallon (Eq. 8).
RHO_AVGAS_KG_PER_GAL = 2.72
RHO_JET_A1_KG_PER_GAL = 3.1

#: Kilograms of CO2-equivalent per kilogram of fuel burnt, ICAO 2024 (Eq. 15).
CO2_PER_FUEL_MASS = 3.16

#: Spares allowance folded into the depreciation base, Eq. 2 (the 1.125 term).
SPARES_ALLOWANCE = 1.125


# --------------------------------------------------------------------------- #
# Ownership
# --------------------------------------------------------------------------- #
def depreciation_cost(
    airframe_cost_usd,
    engine_cost_usd,
    n_engines,
    motor_cost_usd,
    n_motors,
    residual_value_fraction,
    financial_life_years,
    utilization_hr_per_year,
    block_time_hr,
    cpi=1.0,
):
    """Straight-line depreciation charged to one trip (Eq. 2, after Jenkinson).

    The 1.125 adds 12.5% of the powerplant price for spares.  Depreciation is
    inversely proportional to utilization: an aircraft that flies twice as many
    hours a year carries half the ownership cost per trip.
    """
    powerplant = engine_cost_usd * n_engines + motor_cost_usd * n_motors
    base = airframe_cost_usd + SPARES_ALLOWANCE * powerplant
    annual_fraction = (1.0 - residual_value_fraction) / (
        financial_life_years * utilization_hr_per_year
    )
    return base * annual_fraction * block_time_hr * cpi


def fast_charge_factor(c_rate):
    """Battery cycle-life derating from fast charging (Eq. 3 inset).

    ``k_fast = -0.139 * C_rate + 1.155``, a linear fit to Li-ion cycle data.  Set
    ``C_rate`` so ``k_fast = 1`` (about 1.115) to switch the effect off, as the
    paper suggests for other chemistries or for battery swapping.
    """
    return -0.139 * c_rate + 1.155


def battery_depreciation_cost(
    battery_cost_usd, residual_value_fraction, reference_cycles, c_rate
):
    """Battery replacement charged to one trip, one trip being one cycle (Eq. 3).

    Batteries wear out far sooner than the airframe, so they depreciate per cycle
    rather than per hour -- which is why this term does not scale with block time.
    """
    return (
        battery_cost_usd
        * (1.0 - residual_value_fraction)
        / (reference_cycles * fast_charge_factor(c_rate))
    )


def interest_cost(
    aircraft_cost_usd,
    annual_interest_rate,
    loan_years,
    payments_per_year,
    utilization_hr_per_year,
    block_time_hr,
):
    """Amortised loan interest charged to one trip (Eq. 4).

    The level-payment formula gives the monthly payment; multiplying by the number
    of payments and subtracting the principal leaves the interest, spread over the
    flying hours of the loan term.

    The paper labels Eq. 4 [USD/trip] but the expression is per flight *hour* --
    its own Fig. 8 plots the result in USD/h.  We follow the units and multiply by
    block time.
    """
    monthly_rate = annual_interest_rate / payments_per_year
    n_payments = payments_per_year * loan_years
    growth = (1.0 + monthly_rate) ** n_payments
    payment = aircraft_cost_usd * monthly_rate * growth / (growth - 1.0)
    loan_flight_hours = loan_years * utilization_hr_per_year
    interest_per_hour = (payment * n_payments - aircraft_cost_usd) / loan_flight_hours
    return interest_per_hour * block_time_hr


def insurance_cost(
    aircraft_cost_usd,
    annual_premium_fraction,
    electrified_premium_increase,
    utilization_hr_per_year,
    block_time_hr,
):
    """Hull insurance charged to one trip (Eq. 5).

    The premium is a percentage of hull value per year.  ``electrified_premium_
    increase`` is the paper's allowance for the thinner actuarial record behind
    electric propulsion (0 for conventional, 0.063 for electrified).
    """
    annual = aircraft_cost_usd * annual_premium_fraction * (1.0 + electrified_premium_increase)
    return annual / utilization_hr_per_year * block_time_hr


# --------------------------------------------------------------------------- #
# Crew
# --------------------------------------------------------------------------- #
def flight_crew_cost(hourly_salary_usd, block_time_hr):
    """Flight crew cost for one trip (Eq. 6).

    The paper's 286 USD/h is one captain plus one first officer at US regional
    rates.  Note this is paid by the hour, so a slower aircraft on the same route
    costs more crew -- a point several of the nine surveyed models miss by
    treating crew as a fixed annual salary.
    """
    return hourly_salary_usd * block_time_hr


def cabin_crew_cost(hourly_salary_usd, international_factor, n_cabin_crew, block_time_hr):
    """Cabin crew cost for one trip (Eq. 7).

    Written exactly as the paper prints it: the international allowance is an
    additive ``+1.75`` USD/h, not a 1.75x multiplier, even though the surrounding
    prose reads like the latter.  At 39.1 USD/h base the choice is worth about 4%
    of this term, and this term is a few percent of DOC.
    """
    return (hourly_salary_usd + 1.75 * international_factor) * n_cabin_crew * block_time_hr


# --------------------------------------------------------------------------- #
# Energy and carbon
# --------------------------------------------------------------------------- #
def fuel_cost(
    fuel_mass_kg, fuel_density_kg_per_gal, fuel_price_usd_per_gal, saf_blend, saf_price_factor
):
    """Fuel for one trip, with optional SAF blending (Eq. 8).

    ``saf_blend`` runs 0 (none) to 1 (100% SAF); ``saf_price_factor`` is how many
    times conventional fuel the SAF costs (2.26 for HEFA up to 7.89 for SIP).
    """
    gallons = fuel_mass_kg / fuel_density_kg_per_gal
    return gallons * fuel_price_usd_per_gal * (1.0 + saf_blend * (saf_price_factor - 1.0))


def electricity_cost(mission_energy_kwh, electricity_price_usd_per_kwh):
    """Electrical energy to replenish the battery after one trip (Eq. 9)."""
    return mission_energy_kwh * electricity_price_usd_per_kwh


def carbon_tax_cost(fuel_mass_kg, allowance_price_usd_per_ton, saf_blend, taxed_indicator):
    """EU ETS carbon allowance cost for one trip (Eq. 15).

    CO2 is taken as 3.16 kg per kg of fuel (ICAO).  SAF is assumed untaxed, hence
    ``(1 - saf_blend)``; electrified aircraft are exempt via ``taxed_indicator = 0``.
    """
    co2_tons = CO2_PER_FUEL_MASS * fuel_mass_kg / 1000.0
    return allowance_price_usd_per_ton * co2_tons * (1.0 - saf_blend) * taxed_indicator


# --------------------------------------------------------------------------- #
# Maintenance
# --------------------------------------------------------------------------- #
def airframe_maintenance_cost(oew_kg, block_time_hr):
    """Airframe labour and material for one trip (Eq. 11, Jenkinson at 2025 prices).

    ``(15.6 + 3.65 * OEW/1000) * t_b * CPI``, with OEW in kg.
    """
    return (15.6 + 3.65 * oew_kg / 1000.0) * block_time_hr * CPI_2025_OVER_1994


def engine_maintenance_cost(thrust_per_engine_kn, block_time_hr):
    """Engine labour and material per engine for one trip (Eq. 12, TU Berlin).

    ``0.106 * (14.71*T + 30.5*t_b + 10.6) * t_b * CPI``, thrust in kN.  This is a
    CER, so the units do not balance -- do not try to make them.
    """
    return (
        0.106
        * (14.71 * thrust_per_engine_kn + 30.5 * block_time_hr + 10.6)
        * block_time_hr
        * CPI_2025_OVER_2010
    )


def propeller_equivalent_thrust_kn(shaft_power_kw, cruise_speed_m_per_s):
    """Equivalent thrust for a propeller aircraft, ``T = P / V`` (Eq. 12 note)."""
    return shaft_power_kw / cruise_speed_m_per_s


def maintenance_cost(
    oew_kg,
    thrust_per_engine_kn,
    n_engines,
    n_motors,
    maintenance_reduction_factor,
    block_time_hr,
):
    """Total maintenance for one trip (Eq. 10).

    Electric motors are charged at the engine rate reduced by
    ``maintenance_reduction_factor`` -- the paper's allowance for a powerplant with
    far fewer moving parts.
    """
    airframe = airframe_maintenance_cost(oew_kg, block_time_hr)
    per_engine = engine_maintenance_cost(thrust_per_engine_kn, block_time_hr)
    powerplant_count = n_engines + n_motors * (1.0 - maintenance_reduction_factor)
    return {
        "c_maint_airframe_usd": airframe,
        "c_maint_per_engine_usd": per_engine,
        "c_maint_usd": airframe + per_engine * powerplant_count,
    }


# --------------------------------------------------------------------------- #
# Airport and en-route fees
# --------------------------------------------------------------------------- #
def landing_fee(tans_unit_rate_usd_per_ton, mtom_ton):
    """Terminal air navigation (landing) fee for one trip (Eq. 13).

    ``k_TANS * (MTOM/50)^0.7`` -- sublinear in mass, so a heavier aircraft pays
    proportionally less per tonne.
    """
    return tans_unit_rate_usd_per_ton * (mtom_ton / 50.0) ** 0.7


def navigation_fee(en_route_unit_rate_usd_per_ton_km, stage_length_km, mtom_ton):
    """En-route navigation fee for one trip (Eq. 14).

    ``k_en-route * (sl/100) * sqrt(MTOM/50)`` -- linear in distance, square-root
    in mass.
    """
    return (
        en_route_unit_rate_usd_per_ton_km
        * (stage_length_km / 100.0)
        * jnp.sqrt(mtom_ton / 50.0)
    )


# --------------------------------------------------------------------------- #
# Totals
# --------------------------------------------------------------------------- #
#: The twelve elements of Eq. 1, in the order the paper lists them.
DOC_ELEMENTS = (
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
)


def doc_total(elements):
    """Sum the twelve cost elements into the trip DOC (Eq. 1)."""
    return sum(elements[name] for name in DOC_ELEMENTS)


def unit_cost_per_ask(doc_usd, n_passengers, stage_length_km):
    """DOC per available seat kilometre -- the metric airlines actually compare.

    Dividing by seats *and* distance is what makes short stages look so expensive:
    the fixed per-trip costs are spread over a small denominator.
    """
    return doc_usd / (n_passengers * stage_length_km)
