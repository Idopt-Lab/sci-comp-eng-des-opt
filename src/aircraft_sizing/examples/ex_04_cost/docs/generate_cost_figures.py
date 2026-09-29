"""Generate the committed figures for the cost example.

Run from the repository root::

    python src/aircraft_sizing/examples/ex_04_cost/docs/generate_cost_figures.py

Writes PNGs into ``docs/assets/images/`` so the narrative renders on GitHub
without anyone re-running the models.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[5]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from aircraft_sizing.examples.ex_04_cost.methods.aircraft import (  # noqa: E402
    COMMERCIAL_BASELINES,
    MILITARY_AIRCRAFT,
    PUBLISHED_FLYAWAY,
    SCENARIOS,
    VALIDATION_AIRCRAFT,
    commercial_inputs_for,
    escalate_published,
)
from aircraft_sizing.examples.ex_04_cost.methods.config import (  # noqa: E402
    batch_solve_commercial,
    batch_solve_military,
    solve_commercial,
    solve_military,
)

OUTPUT_DIR = Path(__file__).resolve().parent / "assets" / "images"


def _save(figure, name: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / name
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    print(f"wrote {path.relative_to(REPO_ROOT)}")


def cost_breakdown() -> None:
    """Where the money goes on the F-16A baseline."""
    results = solve_military(MILITARY_AIRCRAFT["F-16A"]["dapca"])
    elements = [
        ("Engineering", "c_eng_usd"), ("Tooling", "c_tool_usd"),
        ("Manufacturing", "c_mfg_usd"), ("Quality control", "c_qc_usd"),
        ("Development support", "c_ds_usd"), ("Flight test", "c_ft_usd"),
        ("Materials", "c_mm_usd"), ("Engine production", "c_ep_usd"),
        ("Avionics", "c_avionics_usd"),
    ]
    labels = [label for label, _ in elements]
    values = [results[key] / 1e9 for _, key in elements]

    figure, axes = plt.subplots(figsize=(8.0, 4.5))
    axes.barh(labels, values, color="#4878a8")
    axes.invert_yaxis()
    axes.set_xlabel("programme cost [$B, 1999 dollars, before escalation]")
    axes.set_title("F-16A acquisition cost build-up (DAPCA IV, 200 aircraft)")
    axes.grid(axis="x", alpha=0.3)
    _save(figure, "military_cost_breakdown.png")


def learning_curve() -> None:
    """Unit cost against production quantity, with the three per-aircraft figures."""
    inputs = MILITARY_AIRCRAFT["F-16A"]["dapca"]
    quantities = np.linspace(25.0, 1500.0, 60)
    curve = batch_solve_military(
        inputs,
        [{"quantity": float(q)} for q in quantities],
    )

    figure, axes = plt.subplots(figsize=(8.0, 4.5))
    axes.plot(quantities, [r["reported_unit_usd"] / 1e6 for r in curve], label="programme unit cost")
    axes.plot(quantities, [r["reported_avg_flyaway_usd"] / 1e6 for r in curve], label="average flyaway")
    axes.plot(quantities, [r["reported_recurring_unit_usd"] / 1e6 for r in curve], label="recurring unit")
    axes.axvline(200.0, color="grey", linestyle=":", label="workbook quantity (200)")
    axes.set_xlabel("production quantity")
    axes.set_ylabel("cost per aircraft [$M, 2026 dollars]")
    axes.set_title("The learning curve, and three numbers people all call 'unit cost'")
    axes.grid(alpha=0.3)
    axes.legend()
    _save(figure, "military_learning_curve.png")


def design_to_cost() -> None:
    """Unit cost over empty weight and quantity, with a budget contour."""
    inputs = MILITARY_AIRCRAFT["F-16A"]["dapca"]
    weights = np.linspace(10_000.0, 36_000.0, 30)
    quantities = np.linspace(50.0, 1200.0, 30)
    overrides = [
        {"we_lb": float(w), "quantity": float(q)}
        for q in quantities
        for w in weights
    ]
    grid = np.array(
        [r["reported_unit_usd"] / 1e6 for r in batch_solve_military(inputs, overrides)]
    ).reshape(len(quantities), len(weights))

    figure, axes = plt.subplots(figsize=(8.0, 5.0))
    filled = axes.contourf(weights, quantities, grid, levels=20, cmap="viridis")
    line = axes.contour(weights, quantities, grid, levels=[85.0], colors="crimson")
    axes.clabel(line, fmt=lambda value: f"${value:,.0f}M")
    figure.colorbar(filled, ax=axes, label="programme unit cost [$M, 2026 dollars]")
    axes.set_xlabel("empty weight [lb]")
    axes.set_ylabel("production quantity")
    axes.set_title("Design to cost: what can I afford?")
    _save(figure, "military_design_to_cost.png")


def published_comparison() -> None:
    """Model flyaway cost against published values, at the quantity actually built."""
    names, ratios = [], []
    for name, (published, year, quantity, engines) in PUBLISHED_FLYAWAY.items():
        results = batch_solve_military(
            VALIDATION_AIRCRAFT[name],
            [{"quantity": quantity, "engines_per_aircraft": float(engines)}],
        )[0]
        reference = escalate_published(published, year, 2006)
        names.append(name)
        ratios.append(results["c_avg_flyaway_usd"] / reference)

    order = np.argsort(ratios)
    figure, axes = plt.subplots(figsize=(8.0, 4.0))
    axes.barh([names[i] for i in order], [ratios[i] for i in order], color="#4878a8")
    axes.axvline(1.0, color="black", linewidth=1.2, label="published value")
    axes.axvspan(0.8, 1.2, color="green", alpha=0.12, label="within 20%")
    axes.set_xlabel("model average flyaway / published flyaway (2006 dollars)")
    axes.set_title("DAPCA IV against five real aircraft")
    axes.grid(axis="x", alpha=0.3)
    axes.legend()
    _save(figure, "military_published_comparison.png")


def commercial_unit_cost() -> None:
    """Unit cost against stage length for every Table-13 aircraft."""
    stages = np.linspace(100.0, 1500.0, 30)
    figure, axes = plt.subplots(figsize=(8.0, 5.0))
    for name in COMMERCIAL_BASELINES:
        trials = [commercial_inputs_for(name, float(s), "2030") for s in stages]
        curve = batch_solve_commercial(
            trials[0],
            [
                {
                    "stage_length_km": t.stage_length_km,
                    "block_time_hr": t.block_time_hr,
                    "fuel_mass_kg": t.fuel_mass_kg,
                    "thrust_per_engine_kn": t.thrust_per_engine_kn,
                    "mission_energy_kwh": t.mission_energy_kwh,
                }
                for t in trials
            ],
        )
        axes.plot(stages, [r["unit_cost_usd_per_ask"] for r in curve], label=name)
    axes.set_xlabel("stage length [km]")
    axes.set_ylabel("unit cost [$ per available seat kilometre]")
    axes.set_title("Direct operating cost, 2030 scenario")
    axes.grid(alpha=0.3)
    axes.legend(fontsize=8)
    _save(figure, "commercial_unit_cost.png")


def commercial_scenarios() -> None:
    """What rising SAF blending and carbon price do to a conventional aircraft."""
    figure, axes = plt.subplots(figsize=(8.0, 4.5))
    width = 0.35
    positions = np.arange(len(SCENARIOS))
    for offset, name in enumerate(tuple(COMMERCIAL_BASELINES)):
        values = [
            solve_commercial(commercial_inputs_for(name, 500.0, scenario))["doc_usd"]
            for scenario in SCENARIOS
        ]
        axes.bar(positions + offset * width, values, width, label=name)
    axes.set_xticks(positions + width / 2.0)
    axes.set_xticklabels(tuple(SCENARIOS))
    axes.set_ylabel("direct operating cost [$ per 500 km trip]")
    axes.set_title("Conventional aircraft get dearer as carbon is priced in")
    axes.grid(axis="y", alpha=0.3)
    axes.legend()
    _save(figure, "commercial_scenarios.png")


def main() -> int:
    cost_breakdown()
    learning_curve()
    design_to_cost()
    published_comparison()
    commercial_unit_cost()
    commercial_scenarios()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
