# Third-Party References

## ElectricAircraftDesignExample_AIAA2026 (git submodule)

- **What:** Open-source companion code (Jupyter notebook + Python modules) for the paper
  *"Conceptual Design of Electrified Aircraft: A Practical Guide for Engineering Students
  and Practitioners"* (de Vries et al., **AIAA 2026-4690**, AIAA AVIATION 2026).
- **Upstream:** <https://github.com/EAT-AD-TC/ElectricAircraftDesignExample_AIAA2026>
- **Pinned to:** tag `v1.0` (commit `b3af7f2`).
- **Provenance:** A non-profit initiative of the AIAA Electrified Aircraft Technologies (EAT)
  and Aircraft Design (AD) Technical Committees. © 2026 the paper's authors.
- **Why it's here:** Referenced by
  [`ex_03_electrified_aircraft_sizing.md`](../src/aircraft_sizing/examples/ex_03_e19_hybrid_electric/docs/ex_03_electrified_aircraft_sizing.md)
  for equation-to-code cross-references and to let students run the E-19 worked example.

### How it is included

It is a **git submodule** — this repository stores only a *pointer* (commit SHA in
`.gitmodules` + a gitlink), **not** a copy of the upstream source. To fetch it:

```bash
git submodule update --init references/ElectricAircraftDesignExample_AIAA2026
```

### ⚠ License status

The upstream repository ships **no `LICENSE` file**, so formal reuse terms are not granted.
It is included here **by reference only** (submodule pointer) for non-commercial educational
use, with attribution. Before redistributing the code, vendoring a copy into this repo's
history, or publishing derivatives, **confirm reuse terms with the authors / AIAA EAT-TC**.

The accompanying course document restates equations and factual data from the paper in our
own words with citation; it does not reproduce the paper's prose or figures.

## AircraftCost (local only, not tracked)

- **What:** the source material behind
  [`ex_04_cost`](../src/aircraft_sizing/examples/ex_04_cost/docs/ex_04_cost.md) — a
  textbook design workbook, a conference paper, and two student senior-design
  programmes.
- **Why it is not in the repository:** none of it is ours to redistribute. The
  directory is listed in `.gitignore`; the example cites it and reproduces its
  *numbers* as test constants with cell and equation references, but not its files.

### Contents and provenance

| Item | What it is | Terms |
|---|---|---|
| `Brandt-F16-A.xls` | Design workbook accompanying Brandt et al., *Introduction to Aeronautics: A Design Perspective* (AIAA). Sheet `Cost` is the authority for the F-16A DAPCA IV build-up, the O&M roll-up and the low-observables adder. | © the authors / AIAA. Educational use. |
| `F16BrandtCost.m`, `.md` | A MATLAB port of that `Cost` sheet, with the workbook's validation targets. | Derived from the workbook above. |
| `espinosa-juárez-et-al-2025-…pdf` | Espinosa-Juárez, Jouannet, Amadori & Sánchez Mata, "Comparative Analysis on Aircraft Direct Operating Cost Models", AIAA AVIATION 2025, [10.2514/6.2025-3499](https://doi.org/10.2514/6.2025-3499). | © 2025 AIAA. Obtain your own copy. |
| `Gobbler-Ghost-Program-main/` | Student senior-design programme (AOE 4065, Virginia Tech, Fall 2025). DAPCA IV, Roskam Part VIII and COCOMO II. | Student work. Not redistributed. |
| `ProjectSPEAR Costs/` | Student senior-design programme. Roskam Part VIII with a component-based AMPR weight and basic COCOMO. | Student work. Not redistributed. |

### How the example uses them

The **published sources are the authority**. Where a student programme disagrees
with the textbook it cites, `ex_04_cost` implements the textbook; the divergences and
what each one costs are tabulated in
[`docs/ex_04_cost.md`](../src/aircraft_sizing/examples/ex_04_cost/docs/ex_04_cost.md).
Student code is a comparison case, never a regression target — the test suite checks
the models against the published sources only.

Constants read out of `Brandt-F16-A.xls` carry their cell reference
(`Cost!F11`, `Cost!C80`, …) so any value can be traced back to the sheet.
