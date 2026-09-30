"""Roskam Part VIII and COCOMO against the relations as published.

These are the models implemented in the two student senior-design programmes
supplied as references (``Gobbler-Ghost-Program-main/Cost/`` and
``ProjectSPEAR Costs/``).  Where those programmes disagree with the source they
cite, the source is what is checked here -- the published coefficient, not the
transcribed one.
"""

import unittest

from aircraft_sizing.examples.ex_04_cost.methods import military as M
from aircraft_sizing.examples.ex_04_cost.methods.config import (
    RoskamCostInputs,
    solve_roskam,
)

#: Roskam's AMPR regression, 10^(0.1936 + 0.8645*log10(W_TO)).
AMPR_AT_60000_LB = 21_101.423607

#: Roskam's cost escalation factor, CEF = 6.31752 + 0.104415*(year - 2017).
CEF_1989 = 3.393900
CEF_2017 = 6.317520
CEF_2025 = 7.152840

#: COCOMO II at a representative embedded-avionics rating set.
COCOMO_II_SCALE_FACTOR_SUM = 2.48 + 4.05 + 2.83 + 2.19 + 3.12
COCOMO_II_EFFORT_MULTIPLIER_PRODUCT = (
    1.26 * 1.14 * 1.34 * 1.15 * 1.23 * 1.29 * 1.05 * 1.00
    * 0.90 * 0.95 * 0.95 * 0.95 * 0.91 * 0.90 * 1.00 * 1.10 * 1.00
)

RTOL = 1.0e-9

ROSKAM_CASE = dict(
    w_ampr_lb=AMPR_AT_60000_LB,
    v_max_keas=965.0,
    n_rdte=8.0,
    n_static_test=2.0,
    engineering_rate_usd_per_hr=92.0 * CEF_2025 / CEF_1989,
    manufacturing_rate_usd_per_hr=51.0 * CEF_2025 / CEF_1989,
    tooling_rate_usd_per_hr=65.0 * CEF_2025 / CEF_1989,
    cef=CEF_2025,
    engine_cost_usd=10.0e6,
    engines_per_aircraft=2.0,
    avionics_cost_usd=13.25e6,
    rdte_production_rate_per_month=0.33,
    difficulty_factor=1.3,
    cad_factor=0.8,
    material_factor_roskam=1.75,
    observables_factor=1.3,
)


class PublishedRelationTests(unittest.TestCase):
    """The CER forms, checked against the equations as published."""

    def assertMatches(self, actual, expected, label) -> None:
        got = float(actual)
        self.assertLessEqual(
            abs(got - expected),
            RTOL * abs(expected),
            msg=f"{label}: got {got!r}, published {expected!r}",
        )

    def test_cost_escalation_and_ampr_weight(self) -> None:
        for year, expected in ((1989, CEF_1989), (2017, CEF_2017), (2025, CEF_2025)):
            with self.subTest(year):
                self.assertMatches(M.roskam_cef(year), expected, f"CEF({year})")
        self.assertMatches(
            M.ampr_weight_from_takeoff(60_000.0), AMPR_AT_60000_LB, "W_ampr(60000)"
        )
        # The accurate route: empty weight less the eleven bought-out groups.
        weights = (411.0, 2_512.1, 95.0, 37.0, 640.0, 300.0, 250.0, 1_800.0, 120.0, 100.0, 134.0)
        self.assertAlmostEqual(
            float(M.ampr_weight_from_components(12_674.0, weights)),
            12_674.0 - sum(weights),
            places=9,
        )

    def test_cocomo_effort_and_schedule(self) -> None:
        basic = M.cocomo_basic(1_000.0)
        self.assertMatches(basic["effort_pm"], 3.6 * 1_000.0 ** 1.2, "basic effort")
        self.assertMatches(
            basic["devtime_months"], 2.5 * float(basic["effort_pm"]) ** 0.32, "basic devtime"
        )

        second = M.cocomo_ii(
            8_000.0, COCOMO_II_SCALE_FACTOR_SUM, COCOMO_II_EFFORT_MULTIPLIER_PRODUCT
        )
        exponent = 0.91 + 0.01 * COCOMO_II_SCALE_FACTOR_SUM
        self.assertAlmostEqual(float(second["exponent"]), exponent, places=12)
        self.assertMatches(
            second["effort_pm"],
            2.94 * 8_000.0 ** exponent * COCOMO_II_EFFORT_MULTIPLIER_PRODUCT,
            "COCOMO II effort",
        )
        # The schedule exponent moves with the scale factors; 0.28 is only the
        # nominal value, reached when E == B.
        self.assertMatches(
            second["devtime_months"],
            3.67 * float(second["effort_pm"]) ** (0.28 + 0.2 * (exponent - 0.91)),
            "COCOMO II devtime",
        )
        nominal = M.cocomo_ii(8_000.0, 0.0, COCOMO_II_EFFORT_MULTIPLIER_PRODUCT)
        self.assertMatches(
            nominal["devtime_months"],
            3.67 * float(nominal["effort_pm"]) ** 0.28,
            "COCOMO II devtime at nominal scale factors",
        )

    def test_roskam_rdte_uses_the_published_coefficients(self) -> None:
        rdte = M.roskam_rdte_cost(**ROSKAM_CASE)
        # Roskam publishes 0.0396 for the RDT&E engineering man-hours; one of the
        # student programmes transcribes it as 0.03396.
        expected = (
            0.0396
            * ROSKAM_CASE["w_ampr_lb"] ** 0.791
            * ROSKAM_CASE["v_max_keas"] ** 1.526
            * ROSKAM_CASE["n_rdte"] ** 0.183
            * ROSKAM_CASE["difficulty_factor"]
            * ROSKAM_CASE["cad_factor"]
        )
        self.assertMatches(rdte["mhr_aed_hr"], expected, "MHR_aed")

        # Engines and avionics at full price on the flight-test airframes; one of the
        # student programmes adds a 0.6 factor the source does not have.
        engines_avionics = (
            (ROSKAM_CASE["engine_cost_usd"] * ROSKAM_CASE["engines_per_aircraft"]
             + ROSKAM_CASE["avionics_cost_usd"])
            * (ROSKAM_CASE["n_rdte"] - ROSKAM_CASE["n_static_test"])
        )
        others = ("c_man_rdte_usd", "c_mat_rdte_usd", "c_tool_rdte_usd")
        self.assertMatches(
            float(rdte["c_fta_usd"]) - sum(float(rdte[k]) for k in others)
            - 0.13 * float(rdte["c_man_rdte_usd"]),
            engines_avionics,
            "C_ea,r",
        )

        # Test facilities, profit and financing are fractions OF the phase total,
        # so the closed form must reproduce the requested fractions exactly.
        total = float(rdte["c_rdte_usd"])
        for key in ("c_tsf_usd", "c_pro_rdte_usd", "c_fin_rdte_usd"):
            with self.subTest(key):
                self.assertAlmostEqual(float(rdte[key]) / total, 0.1, places=12)

    def test_roskam_acquisition_and_life_cycle(self) -> None:
        rdte = M.roskam_rdte_cost(**ROSKAM_CASE)
        acquisition = M.roskam_acquisition_cost(
            w_ampr_lb=ROSKAM_CASE["w_ampr_lb"],
            v_max_keas=ROSKAM_CASE["v_max_keas"],
            n_production=500.0,
            n_rdte=ROSKAM_CASE["n_rdte"],
            rdte_costs=rdte,
            engineering_rate_usd_per_hr=ROSKAM_CASE["engineering_rate_usd_per_hr"],
            manufacturing_rate_usd_per_hr=ROSKAM_CASE["manufacturing_rate_usd_per_hr"],
            tooling_rate_usd_per_hr=ROSKAM_CASE["tooling_rate_usd_per_hr"],
            cef=ROSKAM_CASE["cef"],
            engine_cost_usd=ROSKAM_CASE["engine_cost_usd"],
            engines_per_aircraft=ROSKAM_CASE["engines_per_aircraft"],
            avionics_cost_usd=ROSKAM_CASE["avionics_cost_usd"],
            production_rate_per_month=8.0,
            difficulty_factor=ROSKAM_CASE["difficulty_factor"],
            cad_factor=ROSKAM_CASE["cad_factor"],
            material_factor_roskam=ROSKAM_CASE["material_factor_roskam"],
            operating_cost_per_hour_usd=25_000.0,
        )
        # Roskam: C_ACQ = C_MAN * (1 + F_pro).  One student programme writes
        # (1 + F_pro * C_MAN), which is not dimensionally a cost.
        self.assertMatches(
            acquisition["c_acq_usd"], float(acquisition["c_man_usd"]) * 1.1, "C_ACQ"
        )
        self.assertMatches(
            acquisition["aep_usd"],
            (float(acquisition["c_acq_usd"]) + float(rdte["c_rdte_usd"])) / 500.0,
            "AEP",
        )
        life_cycle = M.roskam_life_cycle_cost(5.0e9, 38.0e9, 28.0e9)
        self.assertAlmostEqual(
            float(life_cycle["c_disposal_usd"]) / float(life_cycle["c_lcc_usd"]),
            0.01,
            places=12,
        )

    def test_openmdao_group_matches_the_equations(self) -> None:
        # The assembled group, phase by phase, against the functions above.
        inputs = RoskamCostInputs.baseline()
        results = solve_roskam(inputs)
        self.assertMatches(
            results["w_ampr_lb"], float(M.ampr_weight_from_takeoff(inputs.w_to_lb)), "W_ampr"
        )
        self.assertMatches(
            results["cef"], float(M.roskam_cef(inputs.year_of_economics)), "CEF"
        )
        self.assertMatches(
            results["c_acq_usd"], results["c_man_usd"] * 1.1, "C_ACQ"
        )
        self.assertMatches(
            results["aep_usd"],
            (results["c_acq_usd"] + results["c_rdte_usd"]) / inputs.n_production,
            "AEP",
        )


if __name__ == "__main__":
    unittest.main()
