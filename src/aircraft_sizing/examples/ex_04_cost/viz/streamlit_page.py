"""Interactive aircraft cost trade studies.

Military is the default mode and carries the trade studies; the commercial
direct-operating-cost page is the companion.  Every figure on screen is stated in
2026 dollars, whatever dollar year the underlying CERs were fitted in.
"""

from __future__ import annotations

from dataclasses import replace

import matplotlib

matplotlib.use("Agg")  # headless backend: safe under Streamlit and AppTest threads

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import streamlit as st  # noqa: E402

from ..methods.aircraft import (  # noqa: E402
    COMMERCIAL_BASELINES,
    MILITARY_AIRCRAFT,
    REPORT_DOLLAR_YEAR,
    SCENARIOS,
    commercial_inputs_for,
)
from ..methods.config import (  # noqa: E402
    InputError,
    batch_solve_commercial,
    batch_solve_military,
    military_elasticities,
    solve_commercial,
    solve_military,
    solve_roskam,
)

__all__ = ["render"]

_MODE_KEY = "cost_mode"
_MILITARY_KEY = "cost_military_inputs"
_LOADED_MILITARY_KEY = "cost_loaded_military"
_LOADED_COMMERCIAL_KEY = "cost_loaded_commercial"
_MILITARY_BASELINE_KEY = "cost_military_baseline"
_COMMERCIAL_BASELINE_KEY = "cost_commercial_baseline"
_SCENARIO_KEY = "cost_commercial_scenario"
_STAGE_KEY = "cost_commercial_stage"

MILITARY_MODE = "Military fighter"
COMMERCIAL_MODE = "Passenger transport"

SIGNATURE_LEVELS = ("none", "A", "B", "C", "C_radome")

#: The seven ways to say "the cost", with what each one contains.  All reported in
#: 2026 dollars.  They are not interchangeable, and quoting the wrong one is the
#: commonest mistake in a student cost chart.
COST_FIGURES = (
    ("reported_recurring_unit_usd", "Recurring unit cost",
     "What the next airframe costs: production labour, materials, engines, "
     "avionics and signature treatment."),
    ("reported_avg_flyaway_usd", "Average flyaway cost",
     "Recurring plus tooling. This is what a published flyaway cost means."),
    ("reported_unit_usd", "Programme unit cost",
     "The whole programme divided by the buy, so it carries all the non-recurring "
     "design and test cost too."),
    ("reported_lcc_usd", "Life-cycle unit cost",
     "Programme unit cost plus a lifetime of fuel, crew and maintenance."),
    ("reported_nre_usd", "Total non-recurring",
     "Spent once, however many aircraft you build."),
    ("reported_program_usd", "Total programme",
     "The whole bill: development plus production."),
    ("reported_om_life_usd", "Life O&M per aircraft",
     "Fuel, crew and maintenance over the service life of one aircraft."),
)
_LABELS = {key: label for key, label, _ in COST_FIGURES}

#: Elements of the acquisition build-up, as (output variable, label).
_MILITARY_ELEMENTS = (
    ("c_eng_usd", "Engineering"),
    ("c_tool_usd", "Tooling"),
    ("c_mfg_usd", "Manufacturing"),
    ("c_qc_usd", "Quality control"),
    ("c_ds_usd", "Development support"),
    ("c_ft_usd", "Flight test"),
    ("c_mm_usd", "Manufacturing materials"),
    ("c_ep_usd", "Engine production"),
    ("c_avionics_usd", "Avionics"),
    ("c_invest_usd", "Investment"),
    ("c_stealth_program_usd", "Signature treatment"),
    ("c_software_usd", "Software"),
)

_COMMERCIAL_ELEMENTS = (
    ("c_depreciation_usd", "Depreciation"),
    ("c_battery_depreciation_usd", "Battery depreciation"),
    ("c_interest_usd", "Interest"),
    ("c_insurance_usd", "Insurance"),
    ("c_flight_crew_usd", "Flight crew"),
    ("c_cabin_crew_usd", "Cabin crew"),
    ("c_fuel_usd", "Fuel"),
    ("c_electricity_usd", "Electricity"),
    ("c_maint_usd", "Maintenance"),
    ("c_landing_usd", "Landing fees"),
    ("c_navigation_usd", "Navigation fees"),
    ("c_carbon_usd", "Carbon tax"),
)

#: Sidebar layout: (expander, [(field, label, step, format)]).  Every field listed
#: here moves at least one of COST_FIGURES.
_MILITARY_GROUPS = (
    ("Airframe", (
        ("we_lb", "Empty weight [lb]", 100.0, "%.0f"),
        ("mach_max", "Maximum Mach", 0.05, "%.2f"),
    )),
    ("Programme", (
        ("quantity", "Production quantity", 25.0, "%.0f"),
        ("flight_test_aircraft", "Flight-test aircraft", 1.0, "%.0f"),
    )),
    ("Propulsion", (
        ("thrust_max_lbf", "Max thrust per engine [lbf]", 500.0, "%.0f"),
        ("turbine_inlet_temp_r", "Turbine inlet temperature [R]", 50.0, "%.0f"),
    )),
    ("Materials (percent of structure)", (
        ("aluminium_percent", "Aluminium", 5.0, "%.0f"),
        ("carbon_fibre_percent", "Carbon fibre", 5.0, "%.0f"),
        ("fibreglass_percent", "Fibreglass", 5.0, "%.0f"),
        ("steel_percent", "Steel", 5.0, "%.0f"),
        ("titanium_percent", "Titanium", 5.0, "%.0f"),
    )),
    ("Labour rates [$/hr]", (
        ("rate_engineering_usd_per_hr", "Engineering", 5.0, "%.0f"),
        ("rate_tooling_usd_per_hr", "Tooling", 5.0, "%.0f"),
        ("rate_manufacturing_usd_per_hr", "Manufacturing", 5.0, "%.0f"),
        ("rate_quality_usd_per_hr", "Quality control", 5.0, "%.0f"),
    )),
    ("Programme economics", (
        ("avionics_factor", "Avionics factor", 0.05, "%.2f"),
        ("investment_factor", "Investment cost factor", 0.05, "%.2f"),
    )),
    ("Operations and support", (
        ("flight_hours_per_year", "Flight hours per year", 25.0, "%.0f"),
        ("maintenance_hours_per_flight_hour", "Maintenance hours per flight hour", 1.0, "%.1f"),
        ("crew_ratio", "Crew ratio", 0.1, "%.2f"),
        ("crew_hours_per_year", "Annual crew hours", 25.0, "%.0f"),
        ("fuel_usd_per_lb", "Fuel price [$/lb]", 0.05, "%.2f"),
        ("design_mission_fuel_lb", "Design mission fuel [lb]", 250.0, "%.0f"),
        ("design_mission_time_hr", "Design mission time [hr]", 0.1, "%.2f"),
        ("life_years", "Service life [years]", 1.0, "%.0f"),
    )),
    ("Software (COCOMO II)", (
        ("ksloc", "Source lines of code [KSLOC]", 250.0, "%.0f"),
        ("scale_factor_sum", "Sum of the five scale factors", 0.5, "%.2f"),
        ("effort_multiplier_product", "Product of the effort multipliers", 0.1, "%.2f"),
        ("usd_per_person_month", "Cost per person-month [$]", 1_000.0, "%.0f"),
    )),
    ("Signature (treated geometry)", (
        ("skin_wetted_area_ft2", "Skin wetted area [ft2]", 50.0, "%.0f"),
        ("treated_edge_length_ft", "Treated edge length [ft]", 5.0, "%.0f"),
        ("hinge_line_length_ft", "Hinge-line length [ft]", 5.0, "%.0f"),
        ("access_panel_perimeter_ft", "Access-panel perimeter [ft]", 5.0, "%.0f"),
        ("inlet_lip_length_ft", "Inlet lip length [ft]", 1.0, "%.1f"),
        ("inlet_duct_area_ft2", "Inlet duct area [ft2]", 10.0, "%.0f"),
        ("vertical_tail_outer_area_ft2", "Vertical tail outer area [ft2]", 10.0, "%.0f"),
        ("exhaust_area_ft2", "Exhaust-washed area [ft2]", 5.0, "%.0f"),
    )),
)

_COMMERCIAL_GROUPS = (
    ("Aircraft", (
        ("aircraft_cost_usd", "Aircraft price [$]", 5.0e5, "%.0f"),
        ("airframe_cost_usd", "Airframe price [$]", 5.0e5, "%.0f"),
        ("engine_cost_usd", "Engine price each [$]", 1.0e4, "%.0f"),
        ("motor_cost_usd", "Electric motor price each [$]", 1.0e4, "%.0f"),
        ("oew_kg", "Operating empty weight [kg]", 250.0, "%.0f"),
        ("mtom_ton", "Maximum takeoff mass [ton]", 0.5, "%.1f"),
        ("n_passengers", "Seats", 1.0, "%.0f"),
    )),
    ("Mission", (
        ("block_time_hr", "Block time [hr]", 0.1, "%.2f"),
        ("fuel_mass_kg", "Trip fuel [kg]", 25.0, "%.0f"),
        ("mission_energy_kwh", "Trip electrical energy [kWh]", 50.0, "%.0f"),
        ("thrust_per_engine_kn", "Thrust per engine [kN]", 0.5, "%.2f"),
    )),
    ("Ownership", (
        ("utilization_hr_per_year", "Utilization [hr/year]", 100.0, "%.0f"),
        ("financial_life_years", "Financial life [years]", 1.0, "%.0f"),
        ("residual_value_fraction", "Residual value fraction", 0.05, "%.2f"),
        ("annual_interest_rate", "Interest rate", 0.005, "%.3f"),
        ("insurance_premium_fraction", "Insurance premium fraction", 0.005, "%.3f"),
    )),
    ("Crew", (
        ("flight_crew_usd_per_hr", "Flight crew [$/hr]", 10.0, "%.0f"),
        ("cabin_crew_usd_per_hr", "Cabin crew [$/hr]", 1.0, "%.1f"),
        ("n_cabin_crew", "Cabin crew members", 1.0, "%.0f"),
    )),
    ("Energy, carbon and fees", (
        ("fuel_usd_per_gal", "Fuel price [$/gal]", 0.1, "%.2f"),
        ("saf_blend", "SAF blend fraction", 0.02, "%.2f"),
        ("saf_price_factor", "SAF price factor", 0.1, "%.2f"),
        ("carbon_price_usd_per_ton", "Carbon price [$/ton]", 10.0, "%.0f"),
        ("electricity_usd_per_kwh", "Electricity [$/kWh]", 0.01, "%.2f"),
        ("tans_usd_per_ton", "Landing unit rate [$/ton]", 10.0, "%.0f"),
        ("en_route_usd_per_ton_km", "En-route unit rate [$/ton km]", 5.0, "%.1f"),
    )),
)

#: Sidebar fields kept off the one-at-a-time sweep, tornado and elasticity charts.
#: The material percentages must sum to 100, so moving one alone prices D47 off an
#: invalid mix -- the Materials tab sweeps them properly, rebalanced against
#: aluminium. The investment factor is zero in the baselines, and a factor that
#: starts at zero has no meaningful percentage change to plot.
_NOT_SWEPT = (
    "aluminium_percent",
    "carbon_fibre_percent",
    "fibreglass_percent",
    "steel_percent",
    "titanium_percent",
    "investment_factor",
)

MILITARY_FIELDS = tuple(
    f for _, e in _MILITARY_GROUPS for f, _, _, _ in e if f not in _NOT_SWEPT
)
COMMERCIAL_FIELDS = tuple(f for _, e in _COMMERCIAL_GROUPS for f, _, _, _ in e)


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def _number_inputs(inputs, groups, expanded):
    """Draw the grouped number inputs and return ``{field: value}``.

    The widgets deliberately carry no ``key``: their displayed value comes from
    ``inputs`` on every run, so switching aircraft just works.
    """
    values = {}
    for title, entries in groups:
        with st.expander(title, expanded=(title == expanded)):
            for field, label, step, fmt in entries:
                values[field] = float(
                    st.number_input(
                        label, value=float(getattr(inputs, field)), step=step, format=fmt
                    )
                )
    return values


def _million(value):
    return f"${value / 1e6:,.1f}M"


def _render_table(title, caption, rows) -> None:
    st.subheader(title)
    st.caption(caption)
    st.dataframe(rows, hide_index=True, width="stretch")


def _show(figure, caption) -> None:
    st.pyplot(figure)
    st.caption(caption)
    plt.close(figure)


def _bar_figure(labels, values, xlabel, title, colour="#4878a8"):
    figure, axes = plt.subplots(figsize=(8.0, 0.32 * len(labels) + 1.8))
    axes.barh(labels, values, color=colour)
    axes.invert_yaxis()
    axes.set_xlabel(xlabel)
    axes.set_title(title)
    axes.grid(axis="x", alpha=0.3)
    figure.tight_layout()
    return figure


def _line_figure(x, series, xlabel, ylabel, title, budget=None, budget_label=""):
    figure, axes = plt.subplots(figsize=(8.0, 4.5))
    for label, values in series:
        axes.plot(x, values, marker="o", markersize=3, label=label)
    if budget is not None:
        axes.axhline(budget, color="crimson", linestyle="--", linewidth=1.5, label=budget_label)
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_title(title)
    axes.grid(alpha=0.3)
    axes.legend()
    figure.tight_layout()
    return figure


def _sweep(inputs, field, values, batch, output):
    """One-at-a-time sweep. Reuses a single problem, so it is fast enough to be live."""
    return [row[output] for row in batch(inputs, [{field: float(v)} for v in values])]


def _tornado(inputs, fields_, batch, output, spread=0.2):
    """Swing in ``output`` from a +/- ``spread`` move in every input.

    An input sitting at zero has no percentage change to take, so it is left off
    the chart -- as it is off the elasticity chart beside it.
    """
    overrides, plan = [], []
    for field in fields_:
        nominal = float(getattr(inputs, field))
        if nominal == 0.0:
            continue
        low, high = nominal * (1.0 - spread), nominal * (1.0 + spread)
        plan.append(field)
        overrides += [{field: low}, {field: nominal}, {field: high}, {field: nominal}]

    results = batch(inputs, overrides)
    base = batch(inputs, [{}])[0][output]
    rows = [
        (field, results[4 * i][output] - base, results[4 * i + 2][output] - base,
         abs(results[4 * i + 2][output] - results[4 * i][output]))
        for i, field in enumerate(plan)
    ]
    rows.sort(key=lambda row: row[3], reverse=True)
    return base, rows


def _tornado_figure(base, rows, title):
    rows = [row for row in rows if row[3] > 0.0][::-1]
    if not rows:
        return None
    labels = [row[0] for row in rows]
    figure, axes = plt.subplots(figsize=(8.0, 0.3 * len(rows) + 2.0))
    axes.barh(labels, [row[1] / 1e6 for row in rows], color="#c06050", label="-20%")
    axes.barh(labels, [row[2] / 1e6 for row in rows], color="#4878a8", label="+20%")
    axes.axvline(0.0, color="black", linewidth=1.0)
    axes.set_xlabel(f"change from baseline ({_million(base)}) [$M]")
    axes.set_title(title)
    axes.grid(axis="x", alpha=0.3)
    axes.legend()
    figure.tight_layout()
    return figure


def _figure_selector(label, keys, index, key=None):
    return st.selectbox(label, keys, index=index, format_func=_LABELS.get, key=key)


# --------------------------------------------------------------------------- #
# Military page
# --------------------------------------------------------------------------- #
def _military_sidebar(inputs, aircraft_name):
    with st.sidebar:
        st.markdown("**Aircraft**")
        name = st.selectbox(
            "Baseline", tuple(MILITARY_AIRCRAFT),
            index=tuple(MILITARY_AIRCRAFT).index(aircraft_name),
            key=_MILITARY_BASELINE_KEY,
        )
        st.caption(MILITARY_AIRCRAFT[name]["note"])
        signature_level = st.selectbox(
            "Signature level", SIGNATURE_LEVELS,
            index=SIGNATURE_LEVELS.index(inputs.signature_level),
            help="Low-observables treatment. 'none' reproduces the Brandt workbook.",
        )
        engines = st.number_input(
            "Engines per aircraft", min_value=0, max_value=6,
            value=int(round(inputs.engines_per_aircraft)), step=1,
        )
        with st.form("cost_military_controls"):
            values = _number_inputs(inputs, _MILITARY_GROUPS, "Programme")
            recompute = st.form_submit_button("Recompute", type="primary")
        reset = st.button("Reset to baseline")
    return name, signature_level, engines, recompute, reset, values


def _render_military(inputs, aircraft_name) -> None:
    results = solve_military(inputs)

    st.caption(
        f"All costs in {REPORT_DOLLAR_YEAR:.0f} dollars. The CERs work in "
        f"{inputs.model_dollar_year:.0f} dollars and are escalated by "
        f"x{results['dollar_year_scale']:.3f} for reporting."
    )
    top = st.columns(4)
    top[0].metric("Recurring unit", _million(results["reported_recurring_unit_usd"]))
    top[1].metric("Average flyaway", _million(results["reported_avg_flyaway_usd"]))
    top[2].metric("Programme unit", _million(results["reported_unit_usd"]))
    top[3].metric("Life-cycle unit", _million(results["reported_lcc_usd"]))

    tabs = st.tabs([
        "Cost summary", "Learning curve", "Sweep", "Sensitivity", "Design to cost",
        "Materials & signature", "Software", "Ownership", "DAPCA vs Roskam",
    ])

    with tabs[0]:
        _render_table(
            "Seven ways to say 'the cost'",
            f"Table: all in {REPORT_DOLLAR_YEAR:.0f} dollars. They are not interchangeable.",
            [
                {"Figure": label, "Value": _million(results[key]), "Contains": blurb}
                for key, label, blurb in COST_FIGURES
            ],
        )
        # Elements sum to c_total_base_usd, before (1 + EF).
        scale = (1.0 + inputs.escalation_factor) * results["dollar_year_scale"]
        pairs = [(label, results[key] * scale / 1e9)
                 for key, label in _MILITARY_ELEMENTS if results[key] > 0.0]
        _show(
            _bar_figure([p[0] for p in pairs], [p[1] for p in pairs],
                        f"programme cost [$B, {REPORT_DOLLAR_YEAR:.0f}]",
                        "Acquisition cost build-up"),
            "Figure: the eight DAPCA IV elements plus avionics, investment, signature "
            "and software. They sum to the total programme cost in the table above.",
        )

    with tabs[1]:
        st.markdown(
            "Unit cost falls with production quantity because the DAPCA exponents on "
            "`Q` are all less than one. The saving per extra aircraft shrinks, so there "
            "is a point past which buying more stops helping."
        )
        budget = st.number_input("Budget per aircraft [$M]", value=85.0, step=5.0)
        quantities = np.linspace(25.0, 1_500.0, 40)
        curve = batch_solve_military(inputs, [{"quantity": float(q)} for q in quantities])
        _show(
            _line_figure(
                quantities,
                [(label, [row[key] / 1e6 for row in curve]) for key, label, _ in COST_FIGURES[:3]],
                "production quantity", f"cost per aircraft [$M, {REPORT_DOLLAR_YEAR:.0f}]",
                "Learning curve", budget, f"budget ${budget:,.0f}M",
            ),
            "Figure: the three per-aircraft figures against production quantity.",
        )
        affordable = [q for q, row in zip(quantities, curve)
                      if row["reported_unit_usd"] / 1e6 <= budget]
        st.info(
            f"Programme unit cost meets the ${budget:,.0f}M budget from a buy of "
            f"{affordable[0]:,.0f} aircraft."
            if affordable
            else f"Programme unit cost never reaches ${budget:,.0f}M within 1,500 aircraft."
        )

    with tabs[2]:
        swept = st.selectbox("Parameter to sweep", MILITARY_FIELDS)
        output = _figure_selector("Cost figure", [k for k, _, _ in COST_FIGURES], 2)
        nominal = float(getattr(inputs, swept))
        lo, hi = (0.5 * nominal, 1.5 * nominal) if nominal else (0.0, 1.0)
        low, high = st.slider("Range", 0.0, float(max(3.0 * nominal, 1.0)), (float(lo), float(hi)))
        values = np.linspace(low, high, 30)
        swept_values = _sweep(inputs, swept, values, batch_solve_military, output)
        _show(
            _line_figure(values, ((_LABELS[output], [v / 1e6 for v in swept_values]),), swept,
                         f"{_LABELS[output]} [$M, {REPORT_DOLLAR_YEAR:.0f}]",
                         f"{_LABELS[output]} against {swept}"),
            "Figure: one-at-a-time sweep, re-solved at every point.",
        )

    with tabs[3]:
        st.markdown(
            "**Two views of the same question.** The tornado chart is in dollars, so it "
            "shows which parameter moves the most money. The elasticity chart is "
            "dimensionless -- 0.8 means a 10% rise in that input raises the cost by 8% "
            "-- so it puts a production quantity and a labour rate on the same axis. "
            "Both cover every sidebar parameter that can be moved on its own: the "
            "material percentages have to sum to 100, so they are swept in the "
            "Materials tab instead. Switch the cost figure and watch "
            "the ranking change: what drives the sticker price is not what drives the "
            "cost of owning the fleet."
        )
        metric = _figure_selector("Cost figure", [k for k, _, _ in COST_FIGURES[:4]], 2,
                                  key="cost_sensitivity_metric")
        left, right = st.columns(2)
        with left:
            base, rows = _tornado(inputs, MILITARY_FIELDS, batch_solve_military, metric)
            figure = _tornado_figure(base, rows, f"{_LABELS[metric]}: +/-20% on every input")
            if figure is None:
                st.info("No sidebar input moves this figure.")
            else:
                _show(figure, "Figure: tornado chart over every sidebar parameter.")
        with right:
            pairs = military_elasticities(inputs, MILITARY_FIELDS, metric)[::-1]
            if pairs:
                _show(
                    _bar_figure([p[0] for p in pairs], [p[1] for p in pairs],
                                "elasticity  d(ln cost) / d(ln input)",
                                f"{_LABELS[metric]}: elasticity", colour="#5a9367"),
                    "Figure: percentage change in cost per percentage change in input.",
                )

    with tabs[4]:
        metric = _figure_selector("Cost figure shown", [k for k, _, _ in COST_FIGURES[:4]], 2,
                                  key="cost_contour_metric")
        st.markdown(
            f"Cost as a **constraint on the design space** rather than a number reported "
            f"at the end. The contour is **{_LABELS[metric]}** in "
            f"{REPORT_DOLLAR_YEAR:.0f} dollars; the red line is the budget."
        )
        budget = st.number_input("Budget per aircraft [$M]", value=85.0, step=5.0,
                                 key="cost_contour_budget")
        weights = np.linspace(0.5 * inputs.we_lb, 1.8 * inputs.we_lb, 22)
        quantities = np.linspace(50.0, 1_200.0, 22)
        grid = np.array([
            row[metric] / 1e6 for row in batch_solve_military(inputs, [
                {"we_lb": float(w), "quantity": float(q)}
                for q in quantities for w in weights
            ])
        ]).reshape(len(quantities), len(weights))

        figure, axes = plt.subplots(figsize=(8.0, 5.0))
        filled = axes.contourf(weights, quantities, grid, levels=18, cmap="viridis")
        line = axes.contour(weights, quantities, grid, levels=[budget], colors="crimson")
        axes.clabel(line, fmt=lambda value: f"${value:,.0f}M budget")
        figure.colorbar(filled, ax=axes, label=f"{_LABELS[metric]} [$M, {REPORT_DOLLAR_YEAR:.0f}]")
        axes.set_xlabel("empty weight [lb]")
        axes.set_ylabel("production quantity")
        axes.set_title(f"Design to cost: {_LABELS[metric]}")
        figure.tight_layout()
        _show(figure, "Figure: anything above and left of the red line meets the budget.")

    with tabs[5]:
        st.markdown(
            "**Composites cost more per pound to build, and buy you fewer pounds.** "
            "Raymer's own factors for the DAPCA labour pools run 1.1-1.8 for "
            "graphite-epoxy; Roskam's `F_mat` runs to 3.0 for carbon composite. Both "
            "say a pound of composite is dearer to make, and they agree. The reason a "
            "designer picks composites anyway is the other side of the trade -- roughly "
            "5% off the empty weight -- and **this model cannot see it, because empty "
            "weight is an input here.**\n\n"
            "The Brandt workbook's own table uses 0.9 for carbon fibre, below aluminium. "
            "That nets the weight credit into the cost factor. It reproduces the sheet, "
            "but if you also reduce `we_lb` you have counted the benefit twice."
        )
        other = inputs.fibreglass_percent + inputs.steel_percent + inputs.titanium_percent
        composite = np.linspace(0.0, max(100.0 - other, 1.0), 25)
        curve = batch_solve_military(inputs, [
            {"carbon_fibre_percent": float(p), "aluminium_percent": float(100.0 - other - p)}
            for p in composite
        ])
        _show(
            _line_figure(composite, (
                ("material factor D47", [row["material_factor"] for row in curve]),
            ), "carbon fibre [% of structure]", "material factor D47",
                "Material factor against composite fraction"),
            "Figure: with the workbook's factors more composite lowers D47; with "
            "Raymer's own factors it would rise.",
        )
        levels = SIGNATURE_LEVELS
        stealth = [solve_military(replace(inputs, signature_level=lvl)) for lvl in levels]
        _render_table(
            "Low-observables treatment",
            "Table: signature level priced off the treated geometry -- the one place a "
            "geometric parameter turns straight into dollars.",
            [
                {"Level": lvl,
                 "Per airframe": f"${row['c_stealth_usd']:,.0f}",
                 "Programme unit cost": _million(row["reported_unit_usd"]),
                 "Life-cycle unit cost": _million(row["reported_lcc_usd"])}
                for lvl, row in zip(levels, stealth)
            ],
        )

    with tabs[6]:
        st.markdown(
            "COCOMO II turns source lines of code into effort, schedule and dollars. "
            "The exponent is above one, so doubling the code more than doubles the cost."
        )
        sizes = np.linspace(500.0, 20_000.0, 30)
        cost = _sweep(inputs, "ksloc", sizes, batch_solve_military, "c_software_usd")
        _show(
            _line_figure(sizes, (("software cost", [c / 1e6 for c in cost]),),
                         "code size [KSLOC]", "software development cost [$M]", "Software cost"),
            "Figure: software development cost against code size.",
        )
        columns = st.columns(3)
        columns[0].metric("Schedule", f"{results['devtime_months']:,.0f} months")
        columns[1].metric("Effort", f"{results['effort_person_months']:,.0f} person-months")
        columns[2].metric("Per aircraft", _million(results["c_software_per_aircraft_usd"]))

    with tabs[7]:
        st.markdown(
            "Acquisition price is not the decision variable people assume. Ownership "
            "share is life O&M divided by life-cycle cost."
        )
        lives = np.linspace(5.0, 40.0, 25)
        share = _sweep(inputs, "life_years", lives, batch_solve_military, "ownership_share")
        lcc = _sweep(inputs, "life_years", lives, batch_solve_military, "reported_lcc_usd")
        _show(
            _line_figure(lives, (("O&M share of life-cycle cost", share),),
                         "service life [years]", "share", "Acquisition vs ownership"),
            "Figure: the longer you keep the fleet, the less the purchase price matters.",
        )
        _show(
            _line_figure(lives, (("life-cycle unit cost", [v / 1e6 for v in lcc]),),
                         "service life [years]",
                         f"cost per aircraft [$M, {REPORT_DOLLAR_YEAR:.0f}]", "Life-cycle cost"),
            "Figure: life-cycle cost against service life.",
        )

    with tabs[8]:
        st.markdown(
            f"**{aircraft_name}, costed two ways.** DAPCA IV estimates acquisition from "
            "empty weight, speed and quantity. Roskam Part VIII covers the whole "
            "programme and exposes judgement factors DAPCA has no equivalent for: "
            "technology aggressiveness, CAD experience, material choice and "
            "low-observability. Both are published, both are credible, and they do not "
            "agree. A cost chart without a named method behind it is not a result."
        )
        roskam_inputs = MILITARY_AIRCRAFT[aircraft_name]["roskam"]
        roskam = solve_roskam(roskam_inputs)
        _render_table(
            "Unit price by method",
            f"Table: same aircraft, two textbooks, {REPORT_DOLLAR_YEAR:.0f} dollars.",
            [
                {"Method": "DAPCA IV -- recurring unit cost",
                 "Unit price": _million(results["reported_recurring_unit_usd"])},
                {"Method": "DAPCA IV -- average flyaway cost",
                 "Unit price": _million(results["reported_avg_flyaway_usd"])},
                {"Method": "DAPCA IV -- programme unit cost",
                 "Unit price": _million(results["reported_unit_usd"])},
                {"Method": "Roskam Part VIII -- acquisition per unit",
                 "Unit price": _million(roskam["c_acq_usd"] / roskam_inputs.n_production)},
                {"Method": "Roskam Part VIII -- estimated aircraft price (AEP)",
                 "Unit price": _million(roskam["aep_usd"])},
            ],
        )
        _render_table(
            "Roskam programme phases",
            "Table: Roskam covers the whole programme, so the phase split is visible.",
            [
                {"Phase": phase, "Cost": f"${roskam[key] / 1e9:,.2f}B",
                 "Share of life-cycle": f"{roskam[key] / roskam['c_roskam_lcc_usd']:.1%}"}
                for phase, key in (
                    ("RDT&E", "c_rdte_usd"),
                    ("Acquisition", "c_acq_usd"),
                    ("Operations", "c_ops_usd"),
                    ("Disposal", "c_disposal_usd"),
                )
            ],
        )


# --------------------------------------------------------------------------- #
# Commercial page
# --------------------------------------------------------------------------- #
def _commercial_sidebar(inputs, aircraft_name):
    with st.sidebar:
        st.markdown("**Aircraft**")
        name = st.selectbox(
            "Baseline", tuple(COMMERCIAL_BASELINES),
            index=tuple(COMMERCIAL_BASELINES).index(aircraft_name),
            key=_COMMERCIAL_BASELINE_KEY,
        )
        scenario = st.selectbox("Scenario", tuple(SCENARIOS), key=_SCENARIO_KEY)
        stage = st.slider("Stage length [km]", 100.0, 1_500.0, 500.0, 50.0, key=_STAGE_KEY)
        with st.form("cost_commercial_controls"):
            values = _number_inputs(inputs, _COMMERCIAL_GROUPS, "Mission")
            recompute = st.form_submit_button("Recompute", type="primary")
        reset = st.button("Reset to baseline")
    return name, scenario, stage, recompute, reset, values


def _render_commercial(inputs, aircraft_name, scenario, stage) -> None:
    results = solve_commercial(inputs)

    top = st.columns(3)
    top[0].metric("Direct operating cost", f"${results['doc_usd']:,.0f} per trip")
    top[1].metric("Unit cost", f"${results['unit_cost_usd_per_ask']:.4f} per seat-km")
    top[2].metric("Block time", f"{inputs.block_time_hr:.2f} h")

    tabs = st.tabs(["Cost breakdown", "Stage length", "Sweep", "Scenarios"])

    with tabs[0]:
        pairs = [(label, results[key]) for key, label in _COMMERCIAL_ELEMENTS if results[key] > 0.0]
        _show(
            _bar_figure([p[0] for p in pairs], [p[1] for p in pairs],
                        "cost [$ per trip]", "Direct operating cost per trip"),
            "Figure: the twelve elements of AIAA 2025-3499 Eq. 1.",
        )
        total = results["doc_usd"]
        _render_table(
            "Elements", "Table: direct operating cost by element for one trip.",
            [
                {"Element": label, "Cost": f"${results[key]:,.0f}",
                 "Share": f"{results[key] / total:.1%}"}
                for key, label in _COMMERCIAL_ELEMENTS
            ] + [{"Element": "Total", "Cost": f"${total:,.0f}", "Share": "100.0%"}],
        )

    with tabs[1]:
        st.markdown(
            "Unit cost collapses with stage length because the fixed per-trip charges "
            "are spread over a larger seat-kilometre denominator. This is why regional "
            "operations look so expensive per seat-kilometre."
        )
        stages = np.linspace(100.0, 1_500.0, 30)
        trials = [commercial_inputs_for(aircraft_name, float(s), scenario) for s in stages]
        curve = batch_solve_commercial(trials[0], [
            {"stage_length_km": t.stage_length_km, "block_time_hr": t.block_time_hr,
             "fuel_mass_kg": t.fuel_mass_kg, "thrust_per_engine_kn": t.thrust_per_engine_kn,
             "mission_energy_kwh": t.mission_energy_kwh}
            for t in trials
        ])
        _show(
            _line_figure(stages, (("unit cost", [r["unit_cost_usd_per_ask"] for r in curve]),),
                         "stage length [km]", "unit cost [$ per seat-km]",
                         f"{aircraft_name}, {scenario} scenario"),
            "Figure: unit cost against stage length.",
        )
        _show(
            _line_figure(stages, (("DOC", [r["doc_usd"] for r in curve]),),
                         "stage length [km]", "direct operating cost [$ per trip]", "Trip cost"),
            "Figure: trip cost rises with distance even as unit cost falls.",
        )

    with tabs[2]:
        swept = st.selectbox("Parameter to sweep", COMMERCIAL_FIELDS)
        output = st.selectbox("Output", ("doc_usd", "unit_cost_usd_per_ask"))
        nominal = float(getattr(inputs, swept))
        lo, hi = (0.5 * nominal, 1.5 * nominal) if nominal else (0.0, 1.0)
        low, high = st.slider("Range", 0.0, float(max(3.0 * nominal, 1.0)), (float(lo), float(hi)))
        values = np.linspace(low, high, 30)
        _show(
            _line_figure(values,
                         ((output, _sweep(inputs, swept, values, batch_solve_commercial, output)),),
                         swept, output, f"{output} against {swept}"),
            "Figure: one-at-a-time sweep.",
        )

    with tabs[3]:
        st.markdown(
            "Rising SAF blending and carbon price push conventional aircraft the wrong "
            "way over time; electrified aircraft are exempt from the carbon allowance."
        )
        _render_table(
            "Scenarios", "Table: the 2030, 2040 and 2050 assumption sets from Table 14.",
            [
                {
                    "Scenario": name,
                    "DOC [$/trip]": f"{row['doc_usd']:,.0f}",
                    "Unit cost [$/ASK]": f"{row['unit_cost_usd_per_ask']:.4f}",
                    "Fuel [$]": f"{row['c_fuel_usd']:,.0f}",
                    "Carbon [$]": f"{row['c_carbon_usd']:,.0f}",
                }
                for name, row in (
                    (s, solve_commercial(commercial_inputs_for(aircraft_name, float(stage), s)))
                    for s in SCENARIOS
                )
            ],
        )


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def render() -> None:
    """Render the aircraft cost trade-study page."""
    st.set_page_config(page_title="Aircraft cost trade studies", layout="wide")
    st.session_state.setdefault(_MILITARY_KEY, MILITARY_AIRCRAFT["F-16A"]["dapca"])
    st.session_state.setdefault(_LOADED_MILITARY_KEY, "F-16A")
    st.session_state.setdefault(_LOADED_COMMERCIAL_KEY, "ATR 72-500")

    with st.sidebar:
        st.markdown("### Cost model")
        mode = st.radio("Aircraft class", (MILITARY_MODE, COMMERCIAL_MODE), key=_MODE_KEY)
        st.divider()

    st.title("Aircraft cost trade studies")

    if mode == MILITARY_MODE:
        loaded = st.session_state.get(_LOADED_MILITARY_KEY)
        if loaded not in MILITARY_AIRCRAFT:
            loaded = "F-16A"
            st.session_state[_MILITARY_KEY] = MILITARY_AIRCRAFT[loaded]["dapca"]
            st.session_state[_LOADED_MILITARY_KEY] = loaded
        # Resolve the baseline before drawing the sidebar, so it is seeded from it.
        selected = st.session_state.get(_MILITARY_BASELINE_KEY, loaded)
        if selected in MILITARY_AIRCRAFT and selected != loaded:
            st.session_state[_MILITARY_KEY] = MILITARY_AIRCRAFT[selected]["dapca"]
            st.session_state[_LOADED_MILITARY_KEY] = loaded = selected
        inputs = st.session_state[_MILITARY_KEY]
        name, signature_level, engines, recompute, reset, values = _military_sidebar(inputs, loaded)

        try:
            if reset:
                inputs = MILITARY_AIRCRAFT[name]["dapca"]
            elif recompute:
                inputs = replace(inputs, **values)
            inputs = replace(
                inputs,
                signature_level=signature_level,
                engines_per_aircraft=float(engines),
                report_dollar_year=REPORT_DOLLAR_YEAR,
            )
            st.session_state[_MILITARY_KEY] = inputs
            _render_military(inputs, name)
        except InputError as error:
            st.error(f"{error}  Adjust the sidebar and press Recompute.")
    else:
        loaded = st.session_state.get(_LOADED_COMMERCIAL_KEY)
        if loaded not in COMMERCIAL_BASELINES:
            loaded = "ATR 72-500"
        selected = st.session_state.get(_COMMERCIAL_BASELINE_KEY, loaded)
        if selected in COMMERCIAL_BASELINES:
            loaded = selected
        st.session_state[_LOADED_COMMERCIAL_KEY] = loaded
        # Seed the sidebar from the mission being solved.
        seed_stage = float(st.session_state.get(_STAGE_KEY, 500.0))
        seed_scenario = st.session_state.get(_SCENARIO_KEY, next(iter(SCENARIOS)))
        name, scenario, stage, recompute, reset, values = _commercial_sidebar(
            commercial_inputs_for(loaded, seed_stage, seed_scenario), loaded
        )
        try:
            inputs = commercial_inputs_for(name, float(stage), scenario)
            if recompute and not reset:
                inputs = replace(inputs, **values)
            _render_commercial(inputs, name, scenario, stage)
        except InputError as error:
            st.error(f"{error}  Adjust the sidebar and press Recompute.")
