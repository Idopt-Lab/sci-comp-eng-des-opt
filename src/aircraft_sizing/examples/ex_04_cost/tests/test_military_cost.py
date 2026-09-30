"""The DAPCA IV models against ``Brandt-F16-A.xls`` sheet ``Cost``.

Every expected value below was read out of that sheet; the cell reference is given
alongside each one.  Where the sheet and the published source disagree, the
published source wins: the quality-control line is the one such case, and the
correction is applied in a single place below.
"""

import unittest
from dataclasses import replace

from aircraft_sizing.examples.ex_04_cost.methods import military as M
from aircraft_sizing.examples.ex_04_cost.methods.config import (
    MilitaryCostInputs,
    solve_military,
)

# --------------------------------------------------------------------------- #
# Inputs -- Cost!F11:F35 and the material table Cost!B41:C45
# --------------------------------------------------------------------------- #
WE_LB = 19_980.7005781593          # Cost!F11
Q = 200.0                          # Cost!F13
FTA = 3.0                          # Cost!F14
TMAX_LBF = 23_770.0                # Cost!F17
MMAX = 2.0                         # Cost!F18
TT4_R = 3_400.0                    # Cost!F19
RATES = (86.0, 88.0, 73.0, 81.0)   # Cost!F20/F21/F23/F22 -- eng, tool, mfg, qc
AF = 0.5                           # Cost!F24
ICF = 0.0                          # Cost!F25
EF = 0.16                          # Cost!F26
DMF_LB = 6_000.43227294689         # Cost!F27
DMT_HR = 1.56763275690196          # Cost!F28
AMF = 1.0                          # Cost!F29
FH_PER_YR = 500.0                  # Cost!F30
FUEL_USD_PER_LB = 0.3              # Cost!F31
CREW_RATIO = 1.1                   # Cost!F32
CREW_HR_PER_YR = 300.0             # Cost!F33
MMH_PER_FH = 15.0                  # Cost!F34
LIFE_YR = 20.0                     # Cost!F35
MATERIAL_PERCENTS = (65.0, 20.0, 0.0, 5.0, 10.0)   # Cost!B41:B45
MATERIAL_FACTORS = (1.0, 0.9, 0.9, 1.0, 1.5)       # Cost!C41:C45

# --------------------------------------------------------------------------- #
# Expected outputs -- read from the sheet
# --------------------------------------------------------------------------- #
D47 = 1.03                         # Cost!D47
V_KTS = 1_145.68047337278          # Cost!F12

# --------------------------------------------------------------------------- #
# The one place this example deliberately leaves the sheet
# --------------------------------------------------------------------------- #
# Raymer Ch. 18 publishes H_qc = 0.133 H_mfg.  The workbook applies D47 a second
# time in Cost!C56, so its QC line -- and every cell downstream of it -- is high
# by that factor.  We follow Raymer, so the QC-dependent expectations below are
# the sheet's own numbers with the spurious factor taken back out.  Everything
# else is the sheet, unmodified.  docs/ex_04_cost.md Step 7 records the delta.
QC_HOURS_WORKBOOK = 4_589_534.36394679          # Cost!C56, with the extra D47
QC_COST_WORKBOOK = 371_752_283.47969            # Cost!C63, likewise
QC_DELTA_USD = QC_COST_WORKBOOK * (1.0 - 1.0 / D47)

HOURS = {                          # Cost!C53:C56
    "h_eng_hr": 20_592_156.4802121,
    "h_tool_hr": 10_683_387.8507731,
    "h_mfg_hr": 33_502_696.2840119,
    "h_qc_hr": QC_HOURS_WORKBOOK / D47,
}
ELEMENTS = {                       # Cost!C60:C63 and C67:C70
    "c_eng_usd": 1_770_925_457.29824,
    "c_tool_usd": 940_138_130.868035,
    "c_mfg_usd": 2_445_696_828.73287,
    "c_qc_usd": QC_COST_WORKBOOK - QC_DELTA_USD,
    "c_ds_usd": 320_413_826.464056,
    "c_ft_usd": 55_788_873.4313102,
    "c_mm_usd": 800_172_930.165402,
    "c_ep_usd": 1_159_359_542.0,
}
SUBTOTAL_USD = 7_864_247_872.4396 - QC_DELTA_USD                  # Cost!C72

# How the QC correction propagates.  Avionics is AF x subtotal and the investment
# factor is zero, so the base falls by (1 + AF) x delta; the per-aircraft figures
# carry that through escalation and the buy.  Non-recurring cost is untouched:
# quality control is a recurring line, which is the cross-check on all of this.
_BASE_DELTA = (1.0 + AF) * QC_DELTA_USD
_UNIT_DELTA = (1.0 + EF) * _BASE_DELTA / Q

ROLL_UP = {
    "c_avionics_usd": 3_932_123_936.2198 - AF * QC_DELTA_USD,     # Cost!C74
    "c_invest_usd": 0.0,                                          # Cost!C76
    "c_total_base_usd": 11_796_371_808.6594 - _BASE_DELTA,        # Cost!C78
    "c_program_usd": 13_683_791_298.0449 - (1.0 + EF) * _BASE_DELTA,   # Cost!C80 (= G6)
    "c_unit_usd": 68_418_956.4902245 - _UNIT_DELTA,               # Cost!F80 (= K6)
    "c_recurring_unit_usd": 50_512_812.019467 - _UNIT_DELTA,      # Cost!J6
    "c_avg_flyaway_usd": 55_965_613.1785016 - _UNIT_DELTA,        # Cost!L6
    "c_nre_usd": 3_581_228_894.15151,                             # Cost!I6, unchanged
}
OPERATING = {
    "c_annual_fuel_usd": 574_155.418084521,     # Cost!F93
    "c_annual_crew_usd": 32_920.8,              # Cost!F100
    "c_annual_maint_usd": 635_100.0,            # Cost!F107
    "c_om_annual_usd": 1_242_176.21808452,      # Cost!F109
    "c_om_life_usd": 24_843_524.3616904,        # Cost!F111 (= F7)
}
LCC_USD = 93_262_480.8519149 - _UNIT_DELTA      # Cost!F8

LO_GEOMETRY = dict(                             # Cost!D122:D160
    skin_wetted_area_ft2=1_479.5800436568,
    inlet_lip_length_ft=11.5429484714568,
    inlet_duct_area_ft2=196.230124014765,
    treated_edge_length_ft=162.599342880023,
    vertical_tail_area_ft2=90.7324981111512,
    vertical_tail_outer_area_ft2=90.7324981111512,
    hinge_line_length_ft=83.2826512546281,
    radar_bulkhead_area_ft2=6.424,
    access_panel_perimeter_ft=45.0,
    exhaust_area_ft2=34.6288454143703,
)
LO_COST_USD = {                                 # Cost!E162/F162/G162 and L9
    "A": 9_482.48036559004,
    "B": 236_185.594461624,
    "C": 1_281_039.3425705,
    "C_radome": 1_781_039.3425705,
}

RTOL = 1.0e-9


def _dapca_costs():
    return M.dapca_costs(
        we_lb=WE_LB,
        v_max_kts=V_KTS,
        quantity=Q,
        flight_test_aircraft=FTA,
        material_factor_value=D47,
        rates=RATES,
        thrust_max_lbf=TMAX_LBF,
        mach_max=MMAX,
        turbine_inlet_temp_r=TT4_R,
        engines_total=Q,
    )


class BrandtWorkbookTests(unittest.TestCase):
    """Reproduce sheet ``Cost`` end to end."""

    def assertMatchesWorkbook(self, actual, expected, cell) -> None:
        got = float(actual)
        if expected == 0.0:
            self.assertAlmostEqual(got, 0.0, places=9, msg=cell)
        else:
            self.assertLessEqual(
                abs(got - expected),
                RTOL * abs(expected),
                msg=f"{cell}: got {got!r}, workbook {expected!r}",
            )

    def test_material_factor_and_max_velocity(self) -> None:
        self.assertMatchesWorkbook(
            M.material_factor(MATERIAL_PERCENTS, MATERIAL_FACTORS), D47, "Cost!D47"
        )
        self.assertMatchesWorkbook(M.max_velocity_kts(MMAX), V_KTS, "Cost!F12")

    def test_labour_hours(self) -> None:
        hours = M.dapca_hours(WE_LB, V_KTS, Q, D47)
        for name, expected in HOURS.items():
            with self.subTest(name):
                self.assertMatchesWorkbook(hours[name], expected, f"Cost!{name}")

    def test_cost_elements_and_subtotal(self) -> None:
        costs = _dapca_costs()
        for name, expected in ELEMENTS.items():
            with self.subTest(name):
                self.assertMatchesWorkbook(costs[name], expected, f"Cost!{name}")
        self.assertMatchesWorkbook(costs["c_subtotal_usd"], SUBTOTAL_USD, "Cost!C72")

    def test_programme_roll_up(self) -> None:
        program = M.brandt_program_cost(
            costs=_dapca_costs(),
            quantity=Q,
            avionics_factor=AF,
            investment_factor=ICF,
            escalation_factor=EF,
        )
        for name, expected in ROLL_UP.items():
            with self.subTest(name):
                self.assertMatchesWorkbook(program[name], expected, f"Cost!{name}")

    def test_operations_and_life_cycle_cost(self) -> None:
        operating = M.brandt_operating_cost(
            design_mission_fuel_lb=DMF_LB,
            design_mission_time_hr=DMT_HR,
            average_mission_factor=AMF,
            flight_hours_per_year=FH_PER_YR,
            fuel_usd_per_lb=FUEL_USD_PER_LB,
            crew_ratio=CREW_RATIO,
            crew_hours_per_year=CREW_HR_PER_YR,
            engineering_rate_usd_per_hr=RATES[0],
            maintenance_hours_per_flight_hour=MMH_PER_FH,
            manufacturing_rate_usd_per_hr=RATES[2],
            escalation_factor=EF,
            life_years=LIFE_YR,
        )
        for name, expected in OPERATING.items():
            with self.subTest(name):
                self.assertMatchesWorkbook(operating[name], expected, f"Cost!{name}")
        self.assertMatchesWorkbook(
            M.life_cycle_cost(OPERATING["c_om_life_usd"], ROLL_UP["c_unit_usd"]),
            LCC_USD,
            "Cost!F8",
        )

    def test_stealth_treatment_by_signature_level(self) -> None:
        for level, expected in LO_COST_USD.items():
            with self.subTest(level):
                self.assertMatchesWorkbook(
                    M.stealth_treatment_cost(level=level, **LO_GEOMETRY),
                    expected,
                    f"Cost!level {level}",
                )

    def test_openmdao_group_reproduces_the_workbook(self) -> None:
        # The assembled group, not just the equations -- this is what the app and
        # the docs run.  MilitaryCostInputs defaults are the workbook inputs, which
        # means signature level "none" and software priced at zero: the sheet has
        # no line for either, and reports its treatment costs separately.
        results = solve_military(MilitaryCostInputs.baseline())
        expected = dict(ROLL_UP, c_om_life_usd=OPERATING["c_om_life_usd"], c_lcc_usd=LCC_USD)
        for name, want in expected.items():
            with self.subTest(name):
                self.assertMatchesWorkbook(results[name], want, f"Cost!{name}")
        self.assertEqual(results["c_stealth_program_usd"], 0.0)
        self.assertEqual(results["c_software_usd"], 0.0)

    def test_signature_and_software_enter_the_programme_cost(self) -> None:
        # Both are switched off in the workbook baseline.  Switching them on must
        # move the programme, and each must land in the right place: signature
        # treatment is recurring (every airframe), software is not (written once).
        base = MilitaryCostInputs.baseline()
        reference = solve_military(base)

        signature = solve_military(replace(base, signature_level="C"))
        self.assertGreater(signature["c_unit_usd"], reference["c_unit_usd"])
        self.assertGreater(
            signature["c_recurring_unit_usd"], reference["c_recurring_unit_usd"]
        )

        software = solve_military(replace(base, usd_per_person_month=25_000.0))
        self.assertGreater(software["c_unit_usd"], reference["c_unit_usd"])
        self.assertGreater(software["c_nre_usd"], reference["c_nre_usd"])
        self.assertAlmostEqual(
            software["c_recurring_unit_usd"], reference["c_recurring_unit_usd"], places=6
        )

    def test_stealth_is_escalated_to_the_coefficient_dollar_year(self) -> None:
        # The treatment rates are 2005 dollars; the DAPCA subtotal is not.
        for coefficient_set in ("1999", "2012"):
            with self.subTest(coefficient_set):
                inputs = replace(
                    MilitaryCostInputs.baseline(),
                    signature_level="C",
                    coefficient_set=coefficient_set,
                )
                self.assertAlmostEqual(
                    solve_military(inputs)["c_stealth_usd"],
                    M.escalate(LO_COST_USD["C"], 2005.0, float(coefficient_set)),
                    places=4,
                )

    def test_reported_costs_are_escalated_to_one_dollar_year(self) -> None:
        # A cost is meaningless without its dollar year, so every headline figure is
        # moved to a single reporting year with Roskam's escalation factor.
        results = solve_military(MilitaryCostInputs.baseline())
        scale = results["dollar_year_scale"]
        self.assertGreater(scale, 1.0)
        for native, reported in (
            ("c_unit_usd", "reported_unit_usd"),
            ("c_avg_flyaway_usd", "reported_avg_flyaway_usd"),
            ("c_lcc_usd", "reported_lcc_usd"),
        ):
            with self.subTest(reported):
                self.assertAlmostEqual(
                    results[reported] / results[native], scale, places=9
                )


if __name__ == "__main__":
    unittest.main()
