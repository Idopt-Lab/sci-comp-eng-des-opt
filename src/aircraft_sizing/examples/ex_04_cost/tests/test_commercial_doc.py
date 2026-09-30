"""The direct-operating-cost model against AIAA 2025-3499, Eqs. 1-15.

Espinosa-Juarez, Jouannet, Amadori & Sanchez Mata, "Comparative Analysis on
Aircraft Direct Operating Cost Models", AIAA AVIATION 2025.

Inputs are the ATR 72-500 row of Table 13 with the fixed rates of Table 15 and the
2030 scenario of Table 14.  Each element is checked against its published equation
written out independently here, so a slip in either place shows up rather than
cancelling.  The mission is stated explicitly because the paper does not publish the
block-speed or fuel-burn model behind its own figures.
"""

import math
import unittest

from aircraft_sizing.examples.ex_04_cost.methods import commercial as C
from aircraft_sizing.examples.ex_04_cost.methods.aircraft import commercial_inputs_for
from aircraft_sizing.examples.ex_04_cost.methods.config import (
    CommercialCostInputs,
    solve_commercial,
)

# --- Table 13, ATR 72-500 --------------------------------------------------- #
AIRCRAFT_COST = 20.1e6
AIRFRAME_COST = 16.9e6
ENGINE_COST = 1.6e6
N_ENGINES = 2.0
RESIDUAL_VALUE = 0.20
FINANCIAL_LIFE_YR = 14.0
UTILIZATION_HR_PER_YR = 1_700.0
INTEREST_RATE = 0.05
LOAN_YEARS = 14.0
INSURANCE_PREMIUM = 0.036
N_CABIN_CREW = 2.0
OEW_KG = 12_400.0
MTOM_TON = 22.0
N_PAX = 70.0

# --- Table 15, fixed rates -------------------------------------------------- #
FLIGHT_CREW_USD_PER_HR = 286.0
CABIN_CREW_USD_PER_HR = 39.1
FUEL_USD_PER_GAL = 2.12
ELECTRICITY_USD_PER_KWH = 0.23
TANS_USD_PER_TON = 208.0
EN_ROUTE_USD_PER_TON_KM = 95.3

# --- Table 14, 2030 scenario ------------------------------------------------ #
SAF_BLEND = 0.06
SAF_PRICE_FACTOR = 3.0
CARBON_PRICE_USD_PER_TON = 70.0

# --- the mission this test fixes -------------------------------------------- #
# The ATR 72-500's 1,000 km design mission, on its published *block* speed of
# 420 km/h.  An earlier draft turned Table 13's cruise Mach into a block speed of
# 674 km/h and came out 40% low; see docs/ex_04_cost.md Step 6.
STAGE_LENGTH_KM = 1_000.0
BLOCK_TIME_HR = 2.730952380952381
FUEL_MASS_KG = 1_720.5
THRUST_PER_ENGINE_KN = 10.231765815324167
PAYMENTS_PER_YEAR = 12.0

PLACES = 9


class PaperEquationTests(unittest.TestCase):
    """Each element against the equation as printed in the paper."""

    def test_ownership_equations_2_to_5(self) -> None:
        depreciation = C.depreciation_cost(
            airframe_cost_usd=AIRFRAME_COST,
            engine_cost_usd=ENGINE_COST,
            n_engines=N_ENGINES,
            motor_cost_usd=0.0,
            n_motors=0.0,
            residual_value_fraction=RESIDUAL_VALUE,
            financial_life_years=FINANCIAL_LIFE_YR,
            utilization_hr_per_year=UTILIZATION_HR_PER_YR,
            block_time_hr=BLOCK_TIME_HR,
        )
        base = AIRFRAME_COST + 1.125 * ENGINE_COST * N_ENGINES
        self.assertAlmostEqual(                                              # Eq. 2
            float(depreciation),
            base * (1.0 - RESIDUAL_VALUE)
            / (FINANCIAL_LIFE_YR * UTILIZATION_HR_PER_YR) * BLOCK_TIME_HR,
            places=PLACES,
        )

        self.assertAlmostEqual(                                              # Eq. 3
            float(C.battery_depreciation_cost(1.0e6, 0.225, 1_000.0, 1.26)),
            1.0e6 * (1.0 - 0.225) / (1_000.0 * (-0.139 * 1.26 + 1.155)),
            places=PLACES,
        )

        monthly = INTEREST_RATE / PAYMENTS_PER_YEAR
        n = PAYMENTS_PER_YEAR * LOAN_YEARS
        growth = (1.0 + monthly) ** n
        payment = AIRCRAFT_COST * monthly * growth / (growth - 1.0)
        self.assertAlmostEqual(                                              # Eq. 4
            float(
                C.interest_cost(
                    AIRCRAFT_COST, INTEREST_RATE, LOAN_YEARS, PAYMENTS_PER_YEAR,
                    UTILIZATION_HR_PER_YR, BLOCK_TIME_HR,
                )
            ),
            (payment * n - AIRCRAFT_COST)
            / (LOAN_YEARS * UTILIZATION_HR_PER_YR) * BLOCK_TIME_HR,
            places=PLACES,
        )

        self.assertAlmostEqual(                                              # Eq. 5
            float(
                C.insurance_cost(
                    AIRCRAFT_COST, INSURANCE_PREMIUM, 0.0,
                    UTILIZATION_HR_PER_YR, BLOCK_TIME_HR,
                )
            ),
            AIRCRAFT_COST * INSURANCE_PREMIUM / UTILIZATION_HR_PER_YR * BLOCK_TIME_HR,
            places=PLACES,
        )

    def test_crew_equations_6_and_7(self) -> None:
        self.assertAlmostEqual(                                              # Eq. 6
            float(C.flight_crew_cost(FLIGHT_CREW_USD_PER_HR, BLOCK_TIME_HR)),
            FLIGHT_CREW_USD_PER_HR * BLOCK_TIME_HR,
            places=PLACES,
        )
        # Eq. 7 as printed: the international allowance is an additive +1.75 USD/h,
        # not a 1.75x multiplier, even though the prose reads like the latter.
        self.assertAlmostEqual(
            float(C.cabin_crew_cost(CABIN_CREW_USD_PER_HR, 1.0, N_CABIN_CREW, BLOCK_TIME_HR)),
            (CABIN_CREW_USD_PER_HR + 1.75) * N_CABIN_CREW * BLOCK_TIME_HR,
            places=PLACES,
        )

    def test_energy_and_carbon_equations_8_9_and_15(self) -> None:
        self.assertAlmostEqual(                                              # Eq. 8
            float(
                C.fuel_cost(
                    FUEL_MASS_KG, C.RHO_JET_A1_KG_PER_GAL, FUEL_USD_PER_GAL,
                    SAF_BLEND, SAF_PRICE_FACTOR,
                )
            ),
            FUEL_MASS_KG / C.RHO_JET_A1_KG_PER_GAL * FUEL_USD_PER_GAL
            * (1.0 + SAF_BLEND * (SAF_PRICE_FACTOR - 1.0)),
            places=PLACES,
        )
        self.assertAlmostEqual(                                              # Eq. 9
            float(C.electricity_cost(500.0, ELECTRICITY_USD_PER_KWH)), 115.0, places=PLACES
        )
        self.assertAlmostEqual(                                             # Eq. 15
            float(C.carbon_tax_cost(FUEL_MASS_KG, CARBON_PRICE_USD_PER_TON, SAF_BLEND, 1.0)),
            CARBON_PRICE_USD_PER_TON * (3.16 * FUEL_MASS_KG / 1000.0) * (1.0 - SAF_BLEND),
            places=PLACES,
        )

    def test_maintenance_equations_10_to_12(self) -> None:
        self.assertAlmostEqual(                                             # Eq. 11
            float(C.airframe_maintenance_cost(OEW_KG, BLOCK_TIME_HR)),
            (15.6 + 3.65 * OEW_KG / 1000.0) * BLOCK_TIME_HR * (322.3 / 148.2),
            places=PLACES,
        )
        self.assertAlmostEqual(                                             # Eq. 12
            float(C.engine_maintenance_cost(THRUST_PER_ENGINE_KN, BLOCK_TIME_HR)),
            0.106 * (14.71 * THRUST_PER_ENGINE_KN + 30.5 * BLOCK_TIME_HR + 10.6)
            * BLOCK_TIME_HR * (322.3 / 218.1),
            places=PLACES,
        )
        total = C.maintenance_cost(                                         # Eq. 10
            OEW_KG, THRUST_PER_ENGINE_KN, N_ENGINES, 0.0, 0.0, BLOCK_TIME_HR
        )
        self.assertAlmostEqual(
            float(total["c_maint_usd"]),
            float(total["c_maint_airframe_usd"])
            + N_ENGINES * float(total["c_maint_per_engine_usd"]),
            places=PLACES,
        )
        # Propeller thrust is nominal power over cruise speed (Eq. 12 note).
        self.assertAlmostEqual(
            float(C.propeller_equivalent_thrust_kn(1_000.0, 200.0)), 5.0, places=12
        )

    def test_fee_equations_13_and_14(self) -> None:
        self.assertAlmostEqual(                                             # Eq. 13
            float(C.landing_fee(TANS_USD_PER_TON, MTOM_TON)),
            TANS_USD_PER_TON * (MTOM_TON / 50.0) ** 0.7,
            places=PLACES,
        )
        self.assertAlmostEqual(                                             # Eq. 14
            float(C.navigation_fee(EN_ROUTE_USD_PER_TON_KM, STAGE_LENGTH_KM, MTOM_TON)),
            EN_ROUTE_USD_PER_TON_KM * (STAGE_LENGTH_KM / 100.0) * math.sqrt(MTOM_TON / 50.0),
            places=PLACES,
        )

    def test_baseline_is_the_derived_mission(self) -> None:
        # One mission per aircraft: the dataclass defaults must be what
        # commercial_inputs_for derives for the same aircraft and stage, or the two
        # entry points price different aeroplanes.
        baseline = CommercialCostInputs.baseline()
        derived = commercial_inputs_for("ATR 72-500", STAGE_LENGTH_KM, "2030")
        for field in ("block_time_hr", "fuel_mass_kg", "thrust_per_engine_kn",
                      "stage_length_km", "mission_energy_kwh"):
            with self.subTest(field):
                self.assertAlmostEqual(
                    getattr(baseline, field), getattr(derived, field), places=9
                )
        self.assertAlmostEqual(baseline.block_time_hr, BLOCK_TIME_HR, places=12)

    def test_scenarios_carry_the_table_14_c_rate(self) -> None:
        for scenario, c_rate in (("2030", 1.26), ("2040", 2.0), ("2050", 2.0)):
            with self.subTest(scenario):
                inputs = commercial_inputs_for("ATR 72-500", STAGE_LENGTH_KM, scenario)
                self.assertEqual(inputs.charge_c_rate, c_rate)

    def test_openmdao_group_sums_equation_1(self) -> None:
        # The assembled group: twelve elements, their total, and the unit cost.
        # CommercialCostInputs defaults are the ATR 72-500 case above.
        results = solve_commercial(CommercialCostInputs.baseline())
        self.assertEqual(len(C.DOC_ELEMENTS), 12)
        self.assertAlmostEqual(                                              # Eq. 1
            results["doc_usd"],
            sum(results[name] for name in C.DOC_ELEMENTS),
            places=6,
        )
        self.assertAlmostEqual(
            results["unit_cost_usd_per_ask"],
            results["doc_usd"] / (N_PAX * STAGE_LENGTH_KM),
            places=12,
        )


if __name__ == "__main__":
    unittest.main()
