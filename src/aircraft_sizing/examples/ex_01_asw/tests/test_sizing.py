"""Tests for the OpenMDAO + JAX ASW sizing model.

These lock the Raymer 3.6 baseline numbers, confirm every nonlinear solver reaches
the same takeoff gross weight, and check that the analytic JAX gradients agree with
a finite-difference of the whole solve.
"""

import unittest

from aircraft_sizing.examples.ex_01_asw.methods.atmosphere import (
    MAXIMUM_ALTITUDE_FT,
    MINIMUM_ALTITUDE_FT,
    standard_atmosphere,
)
from aircraft_sizing.examples.ex_01_asw.methods.config import (
    ASWSizingInputs,
    SizingDivergedError,
    derived_quantities,
    input_sensitivity_sweep,
    mission_fractions,
    segment_ratios,
    solve,
)
from aircraft_sizing.examples.ex_01_asw.methods.components import CallCounter
from aircraft_sizing.examples.ex_01_asw.methods.group import (
    OPT_FORMULATIONS,
    SOLVER_CHOICES,
    build_asw_optimization_problem,
    build_asw_problem,
)

import numpy as np

BASELINE_TOGW_LB = 57_615.87

#: Altitude above which the ICAO standard atmosphere holds temperature -- and so
#: the speed of sound -- constant, which flattens every altitude trade in this model.
TROPOPAUSE_FT = 36_089.0


class AtmosphereTests(unittest.TestCase):
    """The ICAO standard atmosphere, in the US customary units the model uses."""

    def test_sea_level_matches_the_standard_atmosphere(self) -> None:
        sea_level = standard_atmosphere(0.0)
        self.assertAlmostEqual(sea_level.speed_of_sound_ft_per_s, 1116.45, places=2)
        self.assertAlmostEqual(sea_level.density_slug_per_ft3, 0.00237689, places=8)
        self.assertAlmostEqual(sea_level.temperature_rankine, 518.67, places=2)
        self.assertAlmostEqual(sea_level.pressure_lb_per_ft2, 2116.22, places=2)

    def test_cruise_altitude_matches_the_standard_atmosphere(self) -> None:
        cruise = standard_atmosphere(30_000.0)
        self.assertAlmostEqual(cruise.speed_of_sound_ft_per_s, 994.85, places=2)
        self.assertAlmostEqual(cruise.density_slug_per_ft3, 0.00089069, places=8)
        self.assertAlmostEqual(cruise.temperature_rankine, 411.84, places=2)
        self.assertAlmostEqual(cruise.pressure_lb_per_ft2, 629.67, places=2)

    def test_properties_fall_through_the_troposphere(self) -> None:
        states = [standard_atmosphere(h) for h in (0.0, 10_000.0, 20_000.0, 30_000.0)]
        for lower, upper in zip(states, states[1:]):
            self.assertLess(upper.speed_of_sound_ft_per_s, lower.speed_of_sound_ft_per_s)
            self.assertLess(upper.density_slug_per_ft3, lower.density_slug_per_ft3)
            self.assertLess(upper.temperature_rankine, lower.temperature_rankine)
            self.assertLess(upper.pressure_lb_per_ft2, lower.pressure_lb_per_ft2)

    def test_temperature_and_speed_of_sound_plateau_in_the_stratosphere(self) -> None:
        # Above the tropopause the ICAO atmosphere is isothermal, so a stops falling
        # even though pressure and density keep dropping.  Every altitude trade in
        # this model therefore goes flat up there.
        lower = standard_atmosphere(TROPOPAUSE_FT + 1_000.0)
        upper = standard_atmosphere(TROPOPAUSE_FT + 9_000.0)
        self.assertAlmostEqual(upper.temperature_rankine, lower.temperature_rankine, places=6)
        self.assertAlmostEqual(
            upper.speed_of_sound_ft_per_s, lower.speed_of_sound_ft_per_s, places=6
        )
        self.assertLess(upper.pressure_lb_per_ft2, lower.pressure_lb_per_ft2)

    def test_dynamic_pressure_uses_the_local_density(self) -> None:
        cruise = standard_atmosphere(30_000.0)
        expected = 0.5 * cruise.density_slug_per_ft3 * 596.91 ** 2
        self.assertAlmostEqual(cruise.dynamic_pressure_lb_per_ft2(596.91), expected, places=9)

    def test_altitude_outside_the_standard_atmosphere_raises(self) -> None:
        for altitude in (MINIMUM_ALTITUDE_FT - 1.0, MAXIMUM_ALTITUDE_FT + 1.0, float("nan")):
            with self.assertRaises(ValueError):
                standard_atmosphere(altitude)

    def test_inputs_reject_an_unflyable_altitude(self) -> None:
        with self.assertRaises(ValueError):
            ASWSizingInputs(cruise_altitude_ft=500_000.0)


class BaselinePhysicsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.inputs = ASWSizingInputs.baseline()

    def test_derived_quantities_match_worked_example(self) -> None:
        derived = derived_quantities(self.inputs)
        self.assertAlmostEqual(derived["wetted_aspect_ratio"], 7.0 / 5.5, places=6)
        self.assertAlmostEqual(derived["lift_to_drag_max"], 15.7941, places=3)
        self.assertAlmostEqual(derived["cruise_lift_to_drag"], 13.6777, places=3)
        self.assertAlmostEqual(derived["loiter_lift_to_drag"], 15.7941, places=3)
        self.assertAlmostEqual(derived["cruise_speed_ft_per_s"], 596.91, places=2)
        # Speed of sound is the ISA value at 30,000 ft, not the hand-entered 994.8 the
        # textbook rounds to; that 0.05 ft/s is the whole reason the numbers below moved.
        self.assertAlmostEqual(derived["speed_of_sound_ft_per_s"], 994.85, places=2)

    def test_segment_ratios_match_worked_example(self) -> None:
        ratios = segment_ratios(self.inputs)
        self.assertAlmostEqual(ratios.warmup_takeoff, 0.970, places=6)
        self.assertAlmostEqual(ratios.climb, 0.985, places=6)
        self.assertAlmostEqual(ratios.landing, 0.995, places=6)
        self.assertAlmostEqual(ratios.outbound_cruise, 0.8564, places=3)
        self.assertAlmostEqual(ratios.on_station_loiter, 0.9268, places=3)
        self.assertAlmostEqual(ratios.prelanding_loiter, 0.9916, places=3)
        self.assertAlmostEqual(ratios.return_cruise, ratios.outbound_cruise, places=9)

    def test_mission_fractions_match_worked_example(self) -> None:
        fractions = mission_fractions(self.inputs)
        self.assertAlmostEqual(fractions["mission_weight_fraction"], 0.6408, places=3)
        self.assertAlmostEqual(fractions["reserve_and_trapped_multiplier"], 1.06, places=6)
        self.assertAlmostEqual(fractions["fuel_weight_fraction"], 0.3808, places=3)


class SolveTests(unittest.TestCase):
    def setUp(self) -> None:
        self.inputs = ASWSizingInputs.baseline()

    def test_baseline_solution_matches_worked_example(self) -> None:
        result = solve(self.inputs)
        self.assertAlmostEqual(result.final_takeoff_gross_weight_lb, BASELINE_TOGW_LB, delta=1.0)
        self.assertAlmostEqual(result.final_empty_weight_fraction, 0.432, places=2)
        self.assertAlmostEqual(result.fixed_weight_lb, 10_800.0, places=6)
        # The closed-form fixed-point history reproduces Raymer's 7-iteration table.
        self.assertEqual(result.iteration_count, 7)
        self.assertLessEqual(abs(result.iteration_history[-1].difference_lb), 1.0)

    def test_all_solvers_reach_the_same_takeoff_weight(self) -> None:
        for solver in SOLVER_CHOICES:
            weight = solve(self.inputs, solver=solver).final_takeoff_gross_weight_lb
            self.assertAlmostEqual(weight, BASELINE_TOGW_LB, delta=1e-2, msg=f"solver={solver}")

    def test_convergence_from_multiple_initial_guesses(self) -> None:
        weights = [
            solve(self.inputs, initial_guess=guess).final_takeoff_gross_weight_lb
            for guess in (30_000.0, 50_000.0, 80_000.0)
        ]
        for weight in weights[1:]:
            self.assertAlmostEqual(weight, weights[0], places=4)

    def test_finite_difference_derivative_mode_also_converges(self) -> None:
        # Newton driven by OpenMDAO finite-difference partials (no JAX) must land
        # on the same answer as the analytic run -- just less efficiently.
        result = solve(self.inputs, solver="newton", deriv="fd")
        self.assertAlmostEqual(result.final_takeoff_gross_weight_lb, BASELINE_TOGW_LB, delta=1.0)

    def test_composite_structure_reduces_takeoff_weight(self) -> None:
        metal = solve(ASWSizingInputs(structure_material="metal"))
        composite = solve(ASWSizingInputs(structure_material="composite"))
        self.assertLess(
            composite.final_takeoff_gross_weight_lb,
            metal.final_takeoff_gross_weight_lb,
        )

    def test_input_sensitivity_sweep_is_monotonic_in_range(self) -> None:
        sweep = input_sensitivity_sweep(
            self.inputs, "cruise_range_one_way_nm", (1_400.0, 1_500.0, 1_600.0)
        )
        weights = [result.final_takeoff_gross_weight_lb for _, result in sweep]
        self.assertTrue(weights[0] < weights[1] < weights[2])

    def test_range_trade_grows_strongly_nonlinearly(self) -> None:
        # The app's range trade spans 1,000-2,000 nm precisely because the response
        # is convex, not linear: doubling the range nearly doubles the aircraft, and
        # the marginal cost per nm more than doubles across the band.
        sweep = input_sensitivity_sweep(
            self.inputs, "cruise_range_one_way_nm", (1_000.0, 1_500.0, 2_000.0)
        )
        low, middle, high = (result.final_takeoff_gross_weight_lb for _, result in sweep)
        self.assertAlmostEqual(low, 42_800.7, delta=1.0)
        self.assertAlmostEqual(middle, BASELINE_TOGW_LB, delta=1.0)
        self.assertAlmostEqual(high, 82_215.0, delta=1.0)
        # A straight line would have a zero second difference; this one is ~9,800 lb.
        self.assertGreater(high - 2.0 * middle + low, 5_000.0)
        # The second half of the band costs well over 1.5x the first half.
        self.assertGreater((high - middle) / (middle - low), 1.5)

    def test_sweep_over_the_full_app_band_stays_convex(self) -> None:
        values = tuple(1_000.0 + 125.0 * i for i in range(9))
        sweep = input_sensitivity_sweep(self.inputs, "cruise_range_one_way_nm", values)
        weights = [result.final_takeoff_gross_weight_lb for _, result in sweep]
        steps = [b - a for a, b in zip(weights, weights[1:])]
        for earlier, later in zip(steps, steps[1:]):
            self.assertGreater(later, earlier)

    def test_cruise_altitude_changes_the_sized_aircraft(self) -> None:
        # The regression this guards: cruise_altitude_ft used to be declared and never
        # read, so 5,000 ft and 30,000 ft returned byte-identical weights.
        low = solve(self.inputs.with_value("cruise_altitude_ft", 5_000.0))
        high = solve(self.inputs.with_value("cruise_altitude_ft", 30_000.0))
        self.assertNotAlmostEqual(
            low.final_takeoff_gross_weight_lb, high.final_takeoff_gross_weight_lb, places=0
        )
        self.assertGreater(low.speed_of_sound_ft_per_s, high.speed_of_sound_ft_per_s)
        self.assertGreater(low.air_density_slug_per_ft3, high.air_density_slug_per_ft3)

    def test_climbing_through_the_troposphere_costs_takeoff_weight(self) -> None:
        # Colder air aloft means a lower speed of sound, a slower Mach-0.6 cruise, and
        # so a worse Breguet range ratio -- the aircraft grows with cruise altitude.
        sweep = input_sensitivity_sweep(
            self.inputs, "cruise_altitude_ft", (0.0, 10_000.0, 20_000.0, 30_000.0)
        )
        weights = [result.final_takeoff_gross_weight_lb for _, result in sweep]
        for lighter, heavier in zip(weights, weights[1:]):
            self.assertGreater(heavier, lighter)

    def test_altitude_trade_goes_flat_above_the_tropopause(self) -> None:
        # Isothermal air above ~36,089 ft freezes the speed of sound, and with it the
        # only atmospheric property this model consumes.
        lower = solve(self.inputs.with_value("cruise_altitude_ft", TROPOPAUSE_FT + 1_000.0))
        upper = solve(self.inputs.with_value("cruise_altitude_ft", TROPOPAUSE_FT + 9_000.0))
        self.assertAlmostEqual(
            upper.final_takeoff_gross_weight_lb,
            lower.final_takeoff_gross_weight_lb,
            places=4,
        )
        # Density still falls, so the reported cruise dynamic pressure keeps dropping.
        self.assertLess(
            upper.cruise_dynamic_pressure_lb_per_ft2,
            lower.cruise_dynamic_pressure_lb_per_ft2,
        )

    def test_range_beyond_model_validity_reports_divergence(self) -> None:
        # Past ~3,000 nm the fixed point runs away from the default 50,000 lb start.
        # That must surface as a SizingDivergedError, never as a NaN weight, a
        # complex W_TO ** b, or a plausible-looking number.
        with self.assertRaises(SizingDivergedError):
            solve(self.inputs.with_value("cruise_range_one_way_nm", 3_500.0))

    def test_iteration_cap_the_solver_cannot_meet_reports_divergence(self) -> None:
        with self.assertRaises(SizingDivergedError):
            solve(self.inputs, max_iterations=3)


class GradientTests(unittest.TestCase):
    """Analytic JAX total derivatives must match a finite-difference of solve()."""

    def setUp(self) -> None:
        self.inputs = ASWSizingInputs.baseline()

    def _analytic_total(self, wrt_path: str) -> float:
        prob = build_asw_problem(self.inputs.params(), self.inputs.input_values(), solver="nlbgs")
        prob.run_model()
        totals = prob.compute_totals(of=["sizing.takeoff_gross_weight"], wrt=[wrt_path])
        return float(totals["sizing.takeoff_gross_weight", wrt_path][0, 0])

    def _central_difference(self, input_name: str, base_value: float, step: float) -> float:
        high = solve(self.inputs.with_value(input_name, base_value + step)).final_takeoff_gross_weight_lb
        low = solve(self.inputs.with_value(input_name, base_value - step)).final_takeoff_gross_weight_lb
        return (high - low) / (2.0 * step)

    def test_dwto_dwing_aspect_ratio(self) -> None:
        analytic = self._analytic_total("aero.wing_aspect_ratio")
        finite = self._central_difference("wing_aspect_ratio", 7.0, 1e-3)
        self.assertAlmostEqual(analytic / finite, 1.0, places=5)

    def test_dwto_dcruise_range(self) -> None:
        # The analytic derivative is per foot; convert the per-nm FD to per-foot.
        analytic = self._analytic_total("mission.range_ft")
        finite_per_nm = self._central_difference("cruise_range_one_way_nm", 1_500.0, 1e-2)
        finite_per_ft = finite_per_nm / 6076.0
        self.assertAlmostEqual(analytic / finite_per_ft, 1.0, places=5)


class ValidationTests(unittest.TestCase):
    def test_invalid_material_raises(self) -> None:
        with self.assertRaises(ValueError):
            ASWSizingInputs(structure_material="titanium")

    def test_nonpositive_range_raises(self) -> None:
        with self.assertRaises(ValueError):
            ASWSizingInputs(cruise_range_one_way_nm=0.0)

    def test_out_of_range_segment_ratio_raises(self) -> None:
        with self.assertRaises(ValueError):
            ASWSizingInputs(climb_weight_ratio=1.5)


class OptimizationFormulationTests(unittest.TestCase):
    """The sizing loop recast as an optimization (Lesson 3) recovers the baseline W_TO."""

    def test_both_formulations_and_derivs_reach_baseline(self) -> None:
        inputs = ASWSizingInputs.baseline()
        for formulation in OPT_FORMULATIONS:
            for deriv in ("jax", "fd"):
                with self.subTest(formulation=formulation, deriv=deriv):
                    counter = CallCounter()
                    prob = build_asw_optimization_problem(
                        inputs.params(), inputs.input_values(),
                        formulation=formulation, deriv=deriv, counter=counter,
                    )
                    with np.errstate(divide="ignore", invalid="ignore"):
                        prob.run_driver()
                    w_to = float(prob.get_val("dv.takeoff_gross_weight")[0])
                    self.assertAlmostEqual(w_to, BASELINE_TOGW_LB, delta=1e-2)
                    # The optimizer actually drove the discipline evaluations.
                    self.assertGreater(counter.total("compute"), 0)

    def test_min_weight_satisfies_residual_constraint(self) -> None:
        inputs = ASWSizingInputs.baseline()
        prob = build_asw_optimization_problem(
            inputs.params(), inputs.input_values(), formulation="min_weight", deriv="jax"
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            prob.run_driver()
        residual = float(prob.get_val("sizing.sizing_residual")[0])
        self.assertGreaterEqual(residual, -1e-6)  # R >= 0 constraint honored

    def test_unknown_formulation_raises(self) -> None:
        inputs = ASWSizingInputs.baseline()
        with self.assertRaises(ValueError):
            build_asw_optimization_problem(
                inputs.params(), inputs.input_values(), formulation="not_a_formulation"
            )


if __name__ == "__main__":
    unittest.main()
