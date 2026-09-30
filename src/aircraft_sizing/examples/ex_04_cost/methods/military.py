"""Pure-JAX cost-estimating relationships for military and commercial aircraft.

Every function here is written with ``jax.numpy`` so OpenMDAO can take *exact*
partials by automatic differentiation (see ``components.py``), exactly as
``ex_01_asw`` does for the sizing physics.

A cost-estimating relationship (CER) is a statistical fit to historical programme
data, not a physical law.  Nothing below is dimensionally consistent, the
exponents carry no physics, and each CER is only valid inside the dollar-year and
the aircraft class it was fitted to.  That is the whole point of the example.

Sources, cited per function:

- **DAPCA IV** -- RAND, as published in Raymer, *Aircraft Design: A Conceptual
  Approach*, Ch. 18.  Two dollar-year coefficient sets are provided.
- **Brandt workbook** -- ``Brandt-F16-A.xls`` sheet ``Cost``, which wraps DAPCA IV
  in a programme roll-up, an O&M model and a low-observables adder.
- **Roskam** -- *Airplane Design Part VIII: Airplane Cost Estimation*, Ch. 2-4, 6-7.
- **COCOMO** -- Boehm, *Software Engineering Economics* (basic) and *COCOMO II*.
- **Espinosa-Juarez et al.** -- "Comparative Analysis on Aircraft Direct Operating
  Cost Models", AIAA 2025-3499, Eqs. 1-15.
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402  (import after enabling x64 on purpose)

#: Brandt Cost!F12: tropopause speed of sound [ft/s] and the workbook's ft/s per knot.
SPEED_OF_SOUND_TROPOPAUSE_FT_PER_S = 968.1
FT_PER_S_PER_KNOT = 1.69

#: DAPCA IV labour-hour lead coefficients, keyed by dollar year.  The exponents
#: are identical across sets; only these leads and the wrap rates they pair with
#: change.  Pairing a 1999 coefficient with a 2012 wrap rate is a common error.
#: ``(engineering, tooling, manufacturing)``, with quality control a fixed share
#: of manufacturing.
DAPCA_HOURS_1999 = (7.07, 8.71, 10.72)
DAPCA_HOURS_2012 = (4.86, 5.99, 7.37)

#: DAPCA IV non-labour lead coefficients ``(development support, flight test,
#: manufacturing materials)`` for the same two dollar years.
DAPCA_OTHER_1999 = (66.0, 1807.1, 16.0)
DAPCA_OTHER_2012 = (91.3, 2498.0, 22.1)

#: Raymer's wrap rates in each dollar year, ``(eng, tool, mfg, qc)`` in $/hour.
DAPCA_RATES_1999 = (86.0, 88.0, 73.0, 81.0)
DAPCA_RATES_2012 = (115.0, 118.0, 98.0, 108.0)

#: Quality-control hours as a share of manufacturing hours (Raymer Ch. 18).
QC_HOURS_FRACTION = 0.133


# --------------------------------------------------------------------------- #
# Material mix
# --------------------------------------------------------------------------- #
#: Per-material cost factors, in the order aluminium, carbon fibre, fibreglass,
#: steel, titanium.
#:
#: ``WORKBOOK`` is Brandt ``Cost!C41:C45`` (0.9 for composites, below aluminium).
#: ``RAYMER`` is midpoints of Raymer Ch. 18: graphite-epoxy 1.1-1.8, fibreglass
#: 1.1-1.2, steel 1.5-2.0, titanium 1.7-2.2.
MATERIAL_FACTORS_WORKBOOK = (1.0, 0.9, 0.9, 1.0, 1.5)
MATERIAL_FACTORS_RAYMER = (1.0, 1.45, 1.15, 1.75, 1.95)


def material_factor(percents, factors):
    """Design/fabrication factor ``D47`` from the structural material mix.

    Brandt Cost!D41:D47: ``D47 = sum(percent_i * factor_i) / 100``, over aluminium,
    carbon fibre, fibreglass, steel and titanium.  Scales the four DAPCA labour
    pools.  Empty weight is an input, so any composite weight saving must be
    applied to ``we_lb``, not netted into the factor.
    """
    percents = jnp.asarray(percents)
    factors = jnp.asarray(factors)
    return jnp.sum(percents * factors) / 100.0


def max_velocity_kts(mach_max):
    """Maximum velocity in knots from the Mach limit (Brandt Cost!F12)."""
    return mach_max * SPEED_OF_SOUND_TROPOPAUSE_FT_PER_S / FT_PER_S_PER_KNOT


# --------------------------------------------------------------------------- #
# DAPCA IV -- RAND, via Raymer Ch. 18
# --------------------------------------------------------------------------- #
def dapca_hours(we_lb, v_max_kts, quantity, material_factor_value, coefficients=DAPCA_HOURS_1999):
    """Engineering, tooling, manufacturing and quality-control hours.

    Raymer Ch. 18, for the whole production run of ``quantity`` aircraft::

        H_eng  = k_e * We^0.777 * V^0.894 * Q^0.163 * D47
        H_tool = k_t * We^0.777 * V^0.696 * Q^0.263 * D47
        H_mfg  = k_m * We^0.82  * V^0.484 * Q^0.641 * D47
        H_qc   = 0.133 * H_mfg

    ``We`` is empty weight in lb and ``V`` is maximum velocity in knots.

    The Brandt workbook multiplies the QC line by ``D47`` a second time
    (``Cost!C56``), which Raymer does not; this follows Raymer.
    """
    k_eng, k_tool, k_mfg = coefficients
    h_eng = k_eng * we_lb ** 0.777 * v_max_kts ** 0.894 * quantity ** 0.163 * material_factor_value
    h_tool = k_tool * we_lb ** 0.777 * v_max_kts ** 0.696 * quantity ** 0.263 * material_factor_value
    h_mfg = k_mfg * we_lb ** 0.82 * v_max_kts ** 0.484 * quantity ** 0.641 * material_factor_value
    h_qc = QC_HOURS_FRACTION * h_mfg
    return {"h_eng_hr": h_eng, "h_tool_hr": h_tool, "h_mfg_hr": h_mfg, "h_qc_hr": h_qc}


def engine_production_cost(thrust_max_lbf, mach_max, turbine_inlet_temp_r, engines_total):
    """Turbine-engine production cost for the whole buy (Raymer Ch. 18).

    ``C_ep = 2.251 * (0.043*T_max + 243.25*M_max + 0.969*Tt4 - 2228) * 1000 * N_eng``
    with thrust in lbf and turbine inlet temperature in degrees Rankine.
    """
    per_engine_thousands = 2.251 * (
        0.043 * thrust_max_lbf + 243.25 * mach_max + 0.969 * turbine_inlet_temp_r - 2228.0
    )
    return per_engine_thousands * 1000.0 * engines_total


def dapca_costs(
    we_lb,
    v_max_kts,
    quantity,
    flight_test_aircraft,
    material_factor_value,
    rates,
    thrust_max_lbf,
    mach_max,
    turbine_inlet_temp_r,
    engines_total,
    hour_coefficients=DAPCA_HOURS_1999,
    other_coefficients=DAPCA_OTHER_1999,
):
    """The eight DAPCA IV cost elements and their subtotal, in the rates' dollars.

    Four come from labour hours times a wrap rate; four are direct CERs::

        C_ds = k_ds * We^0.63  * V^1.3                  development support
        C_ft = k_ft * We^0.325 * V^0.822 * FTA^1.21     flight test
        C_mm = k_mm * We^0.921 * V^0.621 * Q^0.799      manufacturing materials
        C_ep = engine production (see engine_production_cost)

    ``rates`` is ``(engineering, tooling, manufacturing, quality control)`` in
    $/hour and must come from the same dollar year as the coefficients.
    """
    rate_eng, rate_tool, rate_mfg, rate_qc = rates
    k_ds, k_ft, k_mm = other_coefficients
    hours = dapca_hours(we_lb, v_max_kts, quantity, material_factor_value, hour_coefficients)

    c_eng = hours["h_eng_hr"] * rate_eng
    c_tool = hours["h_tool_hr"] * rate_tool
    c_mfg = hours["h_mfg_hr"] * rate_mfg
    c_qc = hours["h_qc_hr"] * rate_qc
    c_ds = k_ds * we_lb ** 0.63 * v_max_kts ** 1.3
    c_ft = k_ft * we_lb ** 0.325 * v_max_kts ** 0.822 * flight_test_aircraft ** 1.21
    c_mm = k_mm * we_lb ** 0.921 * v_max_kts ** 0.621 * quantity ** 0.799
    c_ep = engine_production_cost(thrust_max_lbf, mach_max, turbine_inlet_temp_r, engines_total)

    return {
        "c_eng_usd": c_eng,
        "c_tool_usd": c_tool,
        "c_mfg_usd": c_mfg,
        "c_qc_usd": c_qc,
        "c_ds_usd": c_ds,
        "c_ft_usd": c_ft,
        "c_mm_usd": c_mm,
        "c_ep_usd": c_ep,
        "c_subtotal_usd": c_eng + c_tool + c_mfg + c_qc + c_ds + c_ft + c_mm + c_ep,
    }


# --------------------------------------------------------------------------- #
# Brandt programme roll-up (Brandt Cost!C72:C80 and the I6/J6/L6 splits)
# --------------------------------------------------------------------------- #
def brandt_program_cost(
    costs,
    quantity,
    avionics_factor,
    investment_factor,
    escalation_factor,
    stealth_usd_per_airframe=0.0,
    software_usd=0.0,
):
    """Roll the eight DAPCA elements up into programme and per-aircraft costs.

    Brandt Cost!C72:C80::

        C_avionics   = AF  * C_subtotal
        C_invest     = ICF * (C_subtotal + C_avionics)
        C_total_base = C_subtotal + C_avionics + C_invest + stealth * Q + C_software
        C_program    = (1 + EF) * C_total_base
        C_unit       = C_program / Q

    Signature treatment is recurring (per airframe); software is non-recurring.
    Neither is in the workbook, so ``stealth = software = 0`` reproduces the sheet.

    Outputs:

    ``c_recurring_unit_usd``
        What the *next* airframe costs: production labour, materials, engines,
        avionics and signature treatment.  The marginal cost of one more.
    ``c_avg_flyaway_usd``
        Recurring plus tooling.  This is what a published "flyaway cost" means.
    ``c_unit_usd``
        The whole programme divided by the buy, including non-recurring cost.
    ``c_nre_usd``
        Total non-recurring: what is spent once, however many you build.
    ``c_program_usd``
        The whole bill for the programme.
    ``c_total_base_usd``
        The same, before escalation to the reporting dollar year.
    """
    subtotal = costs["c_subtotal_usd"]
    c_avionics = subtotal * avionics_factor
    c_invest = investment_factor * (subtotal + c_avionics)
    c_stealth_program = stealth_usd_per_airframe * quantity
    c_total_base = subtotal + c_avionics + c_invest + c_stealth_program + software_usd
    c_program = (1.0 + escalation_factor) * c_total_base
    c_unit = c_program / quantity

    recurring = costs["c_mfg_usd"] + costs["c_qc_usd"] + costs["c_mm_usd"] + costs["c_ep_usd"]
    c_recurring_unit = (
        (recurring + c_avionics + c_stealth_program) / quantity * (1.0 + escalation_factor)
    )
    c_avg_flyaway = c_recurring_unit + costs["c_tool_usd"] / quantity * (1.0 + escalation_factor)

    return {
        "c_avionics_usd": c_avionics,
        "c_invest_usd": c_invest,
        "c_stealth_program_usd": c_stealth_program,
        "c_total_base_usd": c_total_base,
        "c_program_usd": c_program,
        "c_unit_usd": c_unit,
        "c_recurring_unit_usd": c_recurring_unit,
        "c_avg_flyaway_usd": c_avg_flyaway,
        "c_nre_usd": c_program - c_recurring_unit * quantity,
    }


# --------------------------------------------------------------------------- #
# Brandt operations & maintenance (Brandt Cost!F87:F111)
# --------------------------------------------------------------------------- #
def brandt_operating_cost(
    design_mission_fuel_lb,
    design_mission_time_hr,
    average_mission_factor,
    flight_hours_per_year,
    fuel_usd_per_lb,
    crew_ratio,
    crew_hours_per_year,
    engineering_rate_usd_per_hr,
    maintenance_hours_per_flight_hour,
    manufacturing_rate_usd_per_hr,
    escalation_factor,
    life_years,
):
    """Annual and whole-life O&M cost for one aircraft (Brandt Cost!F87:F111).

    Three terms, each escalated to the same dollar year as the acquisition cost::

        fuel  = (FH / avg mission time) * avg mission fuel * fuel price
        crew  = crew ratio * annual crew hours * eng rate * (1 + EF)
        maint = MMH/FH * FH * mfg rate * (1 + EF)
    """
    avg_mission_fuel = design_mission_fuel_lb * average_mission_factor
    avg_mission_time = design_mission_time_hr * average_mission_factor
    c_fuel = flight_hours_per_year / avg_mission_time * avg_mission_fuel * fuel_usd_per_lb
    c_crew = crew_ratio * crew_hours_per_year * engineering_rate_usd_per_hr * (1.0 + escalation_factor)
    c_maint = (
        maintenance_hours_per_flight_hour
        * flight_hours_per_year
        * manufacturing_rate_usd_per_hr
        * (1.0 + escalation_factor)
    )
    c_annual = c_fuel + c_crew + c_maint
    return {
        "c_annual_fuel_usd": c_fuel,
        "c_annual_crew_usd": c_crew,
        "c_annual_maint_usd": c_maint,
        "c_om_annual_usd": c_annual,
        "c_om_life_usd": c_annual * life_years,
    }


def life_cycle_cost(life_om_usd, unit_usd):
    """Life-cycle cost per aircraft (Brandt Cost!F8).

    The workbook adds life O&M to the *escalated total per aircraft* (Cost!K6),
    not to the cell it labels "Avg Flyaway Cost" (Cost!L6).  Those are different
    quantities; using L6 here understates LCC by about 13% for the F-16A.
    """
    return life_om_usd + unit_usd


# --------------------------------------------------------------------------- #
# Low-observables treatment (Brandt Cost!A114:G167)
# --------------------------------------------------------------------------- #
#: Installed unit rates in 2005 dollars (Brandt Cost!E164:E167).
LO_RATE_DOLLAR_YEAR = 2005.0
LO_CONDUCTIVE_PAINT_USD_PER_FT2 = 8.0
LO_HF_RAM_USD_PER_FT = 400.0
LO_HIGH_TEMP_USD_PER_FT = 800.0
LO_EDGE_RAS_USD_PER_FT = 6000.0
#: Radome treatment, added on top of a Level C airframe (Brandt Cost!L9).
LO_RADOME_USD = 500_000.0

#: Which of the nine treatments each signature level switches on.  Level A is a
#: "good" front sector only; B and C widen it; C additionally buys volumetric
#: edge material (RAS) on the inlet lips and airframe perimeter.
LO_LEVELS = ("none", "A", "B", "C", "C_radome")


def stealth_treatment_cost(
    level,
    skin_wetted_area_ft2,
    inlet_lip_length_ft,
    inlet_duct_area_ft2,
    treated_edge_length_ft,
    vertical_tail_area_ft2,
    vertical_tail_outer_area_ft2,
    hinge_line_length_ft,
    radar_bulkhead_area_ft2,
    access_panel_perimeter_ft,
    exhaust_area_ft2,
    dollar_year=LO_RATE_DOLLAR_YEAR,
):
    """Per-airframe low-observables treatment cost (Brandt Cost!A114:G167).

    Nine treatments, each an area or a length times an installed unit rate.
    ``level`` is one of ``LO_LEVELS``.  Returns dollars (the sheet's "k$" label is
    wrong) in ``dollar_year``, escalated from 2005 with :func:`roskam_cef`.
    """
    if level not in LO_LEVELS:
        raise ValueError(f"level must be one of {LO_LEVELS}; got {level!r}.")
    if level == "none":
        return 0.0 * skin_wetted_area_ft2

    paint = LO_CONDUCTIVE_PAINT_USD_PER_FT2
    ram = LO_HF_RAM_USD_PER_FT
    ras = LO_EDGE_RAS_USD_PER_FT

    # Level A leaves the outer skin and the airframe edges untreated, and is the
    # only level that treats the radar bulkhead separately (B and C treat the
    # whole radome instead).
    if level == "A":
        skin = 0.0
        edges = 0.0
        hinges = 0.0
        panels = 0.0
        exhaust = 0.0
        vertical = vertical_tail_area_ft2 * paint
        lips = inlet_lip_length_ft * ram
        bulkhead = radar_bulkhead_area_ft2 * ram
    else:
        skin = skin_wetted_area_ft2 * paint
        hinges = hinge_line_length_ft * 2.0 * ram
        panels = access_panel_perimeter_ft * 1.25 * ram
        exhaust = exhaust_area_ft2 * LO_HIGH_TEMP_USD_PER_FT
        vertical = vertical_tail_outer_area_ft2 * ram
        bulkhead = 0.0
        # Level C adds volumetric absorbing edge material on top of the HF RAM,
        # on both the inlet lips and the airframe perimeter.
        edge_rate = ram + ras if level in ("C", "C_radome") else ram
        lips = inlet_lip_length_ft * edge_rate
        edges = treated_edge_length_ft * edge_rate

    duct = inlet_duct_area_ft2 * paint
    total = skin + lips + duct + edges + vertical + hinges + bulkhead + panels + exhaust
    if level == "C_radome":
        total = total + LO_RADOME_USD
    return escalate(total, LO_RATE_DOLLAR_YEAR, dollar_year)


# --------------------------------------------------------------------------- #
# COCOMO -- software cost (Boehm).  This is how lines of code become dollars.
# --------------------------------------------------------------------------- #
def cocomo_basic(kloc, a=3.6, b=1.2, c=2.5, d=0.32, effort_adjustment=1.0):
    """Basic COCOMO effort, schedule and team size (Boehm)::

        effort  = a * KLOC^b * EAF        [person-months]
        devtime = c * effort^d            [months]
        people  = effort / devtime

    ``(a, b, c, d)`` select the development mode: organic, semi-detached or
    embedded.  Avionics is embedded, which is why ``b > 1`` -- doubling the code
    more than doubles the effort.
    """
    effort_pm = a * kloc ** b * effort_adjustment
    devtime_months = c * effort_pm ** d
    return {
        "effort_pm": effort_pm,
        "devtime_months": devtime_months,
        "people": effort_pm / devtime_months,
    }


def cocomo_ii(ksloc, scale_factor_sum, effort_multiplier_product, a=2.94, b=0.91):
    """COCOMO II post-architecture effort and schedule (Boehm et al.)::

        E       = b + 0.01 * sum(scale factors)
        effort  = a * KSLOC^E * product(effort multipliers)   [person-months]
        devtime = 3.67 * effort^(0.28 + 0.2 * (E - b))        [months]

    The five scale factors move both exponents; the schedule exponent is 0.28
    only when their sum is zero.  The effort multipliers only scale the effort.
    """
    exponent = b + 0.01 * scale_factor_sum
    effort_pm = a * ksloc ** exponent * effort_multiplier_product
    return {
        "exponent": exponent,
        "effort_pm": effort_pm,
        "devtime_months": 3.67 * effort_pm ** (0.28 + 0.2 * (exponent - b)),
    }


def software_cost(effort_pm, usd_per_person_month):
    """Total software development cost from COCOMO effort."""
    return effort_pm * usd_per_person_month


# --------------------------------------------------------------------------- #
# Roskam Part VIII -- cost escalation and AMPR weight
# --------------------------------------------------------------------------- #
#: Roskam's cost escalation factor, linear in calendar year (3.394 for 1989).
CEF_INTERCEPT_2017 = 6.31752
CEF_SLOPE_PER_YEAR = 0.104415


def roskam_cef(year):
    """Roskam cost escalation factor for a calendar year."""
    return CEF_INTERCEPT_2017 + CEF_SLOPE_PER_YEAR * (year - 2017.0)


def escalate(cost, year_from, year_to):
    """Move a cost between dollar years with Roskam's escalation factor."""
    return cost * roskam_cef(year_to) / roskam_cef(year_from)


def ampr_weight_from_takeoff(w_to_lb):
    """AMPR weight from takeoff weight, Roskam's regression::

        W_ampr = 10^(0.1936 + 0.8645 * log10(W_TO))

    Quick, but it cannot see configuration changes.  Prefer
    :func:`ampr_weight_from_components` when a weight statement exists.
    """
    return 10.0 ** (0.1936 + 0.8645 * jnp.log10(w_to_lb))


def ampr_weight_from_components(w_empty_lb, component_weights):
    """AMPR weight as Roskam defines it: empty weight less bought-out items.

    ``component_weights`` is the eleven-item list Roskam subtracts -- wheels and
    brakes, engines, starter, cooling fluid, fuel cells, electrical supply,
    instruments and avionics, armament and fire control, air conditioning, APU and
    trapped fuel.  What is left is what the airframer actually builds, which is
    what the CERs were fitted against.
    """
    return w_empty_lb - jnp.sum(jnp.asarray(component_weights))


# --------------------------------------------------------------------------- #
# Roskam Part VIII, Ch. 3 -- research, development, test and evaluation
# --------------------------------------------------------------------------- #
def roskam_rdte_cost(
    w_ampr_lb,
    v_max_keas,
    n_rdte,
    n_static_test,
    engineering_rate_usd_per_hr,
    manufacturing_rate_usd_per_hr,
    tooling_rate_usd_per_hr,
    cef,
    engine_cost_usd,
    engines_per_aircraft,
    avionics_cost_usd,
    rdte_production_rate_per_month,
    difficulty_factor,
    cad_factor,
    material_factor_roskam,
    observables_factor,
    software_cost_usd=0.0,
    test_facilities_fraction=0.1,
    profit_fraction=0.1,
    finance_fraction=0.1,
):
    """Roskam Part VIII Ch. 3 RDT&E cost for a military aircraft.

    Test facilities, profit and financing are fractions of ``C_RDTE`` itself, so
    ``C_RDTE = (everything else) / (1 - sum of fractions)``.

    ``v_max_keas`` is in knots equivalent airspeed and ``w_ampr_lb`` is AMPR
    weight, not empty weight.
    """
    mhr_aed = (
        0.0396
        * w_ampr_lb ** 0.791
        * v_max_keas ** 1.526
        * n_rdte ** 0.183
        * difficulty_factor
        * cad_factor
    )
    c_aed = mhr_aed * engineering_rate_usd_per_hr
    c_dst = (
        0.008325
        * w_ampr_lb ** 0.873
        * v_max_keas ** 1.890
        * n_rdte ** 0.346
        * cef
        * difficulty_factor
    )
    # Static-test airframes carry no engines or avionics.
    c_engines_avionics = (
        (engine_cost_usd * engines_per_aircraft + avionics_cost_usd)
        * (n_rdte - n_static_test)
    )
    mhr_man = 28.984 * w_ampr_lb ** 0.740 * v_max_keas ** 0.543 * n_rdte ** 0.524 * difficulty_factor
    c_man = mhr_man * manufacturing_rate_usd_per_hr
    c_mat = (
        37.632
        * material_factor_roskam
        * w_ampr_lb ** 0.689
        * v_max_keas ** 0.624
        * n_rdte ** 0.792
        * cef
    )
    mhr_tool = (
        4.0127
        * w_ampr_lb ** 0.764
        * v_max_keas ** 0.899
        * n_rdte ** 0.178
        * rdte_production_rate_per_month ** 0.066
        * difficulty_factor
    )
    c_tool = mhr_tool * tooling_rate_usd_per_hr
    c_qc = 0.13 * c_man
    c_fta = c_engines_avionics + c_man + c_mat + c_tool + c_qc
    c_fto = (
        0.001244
        * w_ampr_lb ** 1.160
        * v_max_keas ** 1.371
        * (n_rdte - n_static_test) ** 1.281
        * cef
        * difficulty_factor
        * observables_factor
    )

    overhead = test_facilities_fraction + profit_fraction + finance_fraction
    c_rdte = (c_aed + c_dst + c_fta + c_fto + software_cost_usd) / (1.0 - overhead)
    return {
        "mhr_aed_hr": mhr_aed,
        "c_aed_usd": c_aed,
        "c_dst_usd": c_dst,
        "c_fta_usd": c_fta,
        "c_fto_usd": c_fto,
        "c_man_rdte_usd": c_man,
        "c_mat_rdte_usd": c_mat,
        "c_tool_rdte_usd": c_tool,
        "c_software_usd": software_cost_usd,
        "c_tsf_usd": test_facilities_fraction * c_rdte,
        "c_pro_rdte_usd": profit_fraction * c_rdte,
        "c_fin_rdte_usd": finance_fraction * c_rdte,
        "c_rdte_usd": c_rdte,
    }


# --------------------------------------------------------------------------- #
# Roskam Part VIII, Ch. 4 -- manufacturing and acquisition
# --------------------------------------------------------------------------- #
def roskam_acquisition_cost(
    w_ampr_lb,
    v_max_keas,
    n_production,
    n_rdte,
    rdte_costs,
    engineering_rate_usd_per_hr,
    manufacturing_rate_usd_per_hr,
    tooling_rate_usd_per_hr,
    cef,
    engine_cost_usd,
    engines_per_aircraft,
    avionics_cost_usd,
    production_rate_per_month,
    difficulty_factor,
    cad_factor,
    material_factor_roskam,
    operating_cost_per_hour_usd,
    interior_cost_usd=0.0,
    flight_test_hours_per_aircraft=20.0,
    flight_test_overhead_factor=4.0,
    finance_fraction=0.1,
    profit_fraction=0.1,
):
    """Roskam Part VIII Ch. 4 manufacturing and acquisition cost.

    Each CER is re-evaluated over the whole programme (``N_program = N_m +
    N_rdte``) and the RDT&E share already spent is subtracted -- that difference,
    not a fresh CER, is the production cost.  This is what makes unit cost fall
    with quantity: the learning is baked into the exponents.

    ``interior_cost_usd`` is zero for military aircraft.
    """
    n_program = n_production + n_rdte

    mhr_aed_program = (
        0.0396
        * w_ampr_lb ** 0.791
        * v_max_keas ** 1.526
        * n_program ** 0.183
        * difficulty_factor
        * cad_factor
    )
    c_aed_m = mhr_aed_program * engineering_rate_usd_per_hr - rdte_costs["c_aed_usd"]

    c_engines_avionics_m = (
        engine_cost_usd * engines_per_aircraft + avionics_cost_usd
    ) * n_production

    mhr_man_program = (
        28.984 * w_ampr_lb ** 0.740 * v_max_keas ** 0.543 * n_program ** 0.524 * difficulty_factor
    )
    c_man_m = mhr_man_program * manufacturing_rate_usd_per_hr - rdte_costs["c_man_rdte_usd"]

    c_mat_program = (
        37.632
        * material_factor_roskam
        * w_ampr_lb ** 0.689
        * v_max_keas ** 0.624
        * n_program ** 0.792
        * cef
    )
    c_mat_m = c_mat_program - rdte_costs["c_mat_rdte_usd"]

    mhr_tool_program = (
        4.0127
        * w_ampr_lb ** 0.764
        * v_max_keas ** 0.899
        * n_program ** 0.178
        * production_rate_per_month ** 0.066
        * difficulty_factor
    )
    c_tool_m = mhr_tool_program * tooling_rate_usd_per_hr - rdte_costs["c_tool_rdte_usd"]
    c_qc_m = 0.13 * c_man_m

    c_apc_m = c_engines_avionics_m + interior_cost_usd + c_man_m + c_mat_m + c_tool_m + c_qc_m
    c_fto_m = (
        n_production
        * operating_cost_per_hour_usd
        * flight_test_hours_per_aircraft
        * flight_test_overhead_factor
    )

    c_man_total = (c_aed_m + c_apc_m + c_fto_m) / (1.0 - finance_fraction)
    c_profit = profit_fraction * c_man_total
    c_acq = c_man_total + c_profit
    aep = (c_acq + rdte_costs["c_rdte_usd"]) / n_production
    return {
        "c_aed_m_usd": c_aed_m,
        "c_apc_m_usd": c_apc_m,
        "c_fto_m_usd": c_fto_m,
        "c_fin_m_usd": finance_fraction * c_man_total,
        "c_man_usd": c_man_total,
        "c_profit_usd": c_profit,
        "c_acq_usd": c_acq,
        "aep_usd": aep,
    }


# --------------------------------------------------------------------------- #
# Roskam Part VIII, Ch. 6 -- military operating cost
# --------------------------------------------------------------------------- #
def roskam_operating_cost(
    n_acquired,
    reserve_fraction,
    loss_rate_per_hour,
    utilization_hr_per_year,
    service_years,
    mission_fuel_lb,
    mission_time_hr,
    fuel_price_usd_per_gal,
    fuel_density_lb_per_gal,
    oil_lubricant_factor,
    crew_per_aircraft,
    crew_ratio,
    crew_pay_usd_per_year,
    crew_overhead_factor,
    maintenance_hours_per_flight_hour,
    maintenance_rate_usd_per_hr,
    consumable_rate_usd_per_hr,
    indirect_personnel_fraction=0.2,
    spares_fraction=0.125,
    depot_fraction=0.15,
    misc_fraction=0.055,
):
    """Roskam Part VIII Ch. 6 programme operating cost over the whole service life.

    The fleet shrinks: ``N_res`` aircraft sit in reserve and attrition removes
    more over ``N_yr`` years, so the cost is charged against the average number
    actually in service.  The four fractions are overheads charged on the total,
    hence the same ``/(1 - sum)`` closed form as the RDT&E phase.
    """
    n_reserve = reserve_fraction * n_acquired
    n_initial = n_acquired - n_reserve
    n_lost = loss_rate_per_hour * n_initial * utilization_hr_per_year * service_years
    n_service = n_acquired - n_reserve - 0.5 * n_lost

    missions_per_year = utilization_hr_per_year / mission_time_hr
    c_pol = (
        oil_lubricant_factor
        * mission_fuel_lb
        * (fuel_price_usd_per_gal / fuel_density_lb_per_gal)
        * missions_per_year
        * n_service
        * service_years
    )
    c_crew = (
        n_service
        * crew_per_aircraft
        * crew_ratio
        * crew_pay_usd_per_year
        * crew_overhead_factor
        * service_years
    )
    fleet_flight_hours = n_service * service_years * utilization_hr_per_year
    c_maint_personnel = (
        fleet_flight_hours * maintenance_hours_per_flight_hour * maintenance_rate_usd_per_hr
    )
    c_consumables = (
        fleet_flight_hours * maintenance_hours_per_flight_hour * consumable_rate_usd_per_hr
    )

    overhead = indirect_personnel_fraction + spares_fraction + depot_fraction + misc_fraction
    c_ops = (c_pol + c_crew + c_maint_personnel + c_consumables) / (1.0 - overhead)
    return {
        "n_service": n_service,
        "c_pol_usd": c_pol,
        "c_crew_usd": c_crew,
        "c_maint_personnel_usd": c_maint_personnel,
        "c_consumables_usd": c_consumables,
        "c_indirect_personnel_usd": indirect_personnel_fraction * c_ops,
        "c_spares_usd": spares_fraction * c_ops,
        "c_depot_usd": depot_fraction * c_ops,
        "c_misc_usd": misc_fraction * c_ops,
        "c_ops_usd": c_ops,
        "c_ops_per_hour_usd": c_ops / fleet_flight_hours,
    }


def roskam_life_cycle_cost(c_rdte_usd, c_acq_usd, c_ops_usd, disposal_fraction=0.01):
    """Roskam Part VIII Ch. 2/7 programme life-cycle cost.

    ``LCC = C_RDTE + C_ACQ + C_OPS + C_DISP`` with disposal a fraction of the LCC
    itself, so again a closed form.  Roskam puts disposal at roughly 1% of LCC.
    """
    lcc = (c_rdte_usd + c_acq_usd + c_ops_usd) / (1.0 - disposal_fraction)
    return {"c_lcc_usd": lcc, "c_disposal_usd": disposal_fraction * lcc}
