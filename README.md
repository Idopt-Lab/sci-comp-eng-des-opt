# Scientific Computing Methods for Engineering Design Optimization

This repository collects worked examples, reference notes, and interactive aircraft-sizing materials for the course. It ships an installable Python package, `sci-comp-eng-des-opt` (defined in `pyproject.toml`), plus a Streamlit app for the ASW (Anti-Submarine Warfare) fixed-point sizing example.

## Course information

**Term:** Fall 2026 &middot; **Format:** Graduate independent study (project-based) &middot; **Instructor:** Darshan Sarojini (<sdarshan@vt.edu>), Virginia Tech

Scientific-computing methods for engineering design. The course is for graduate
students whose research involves multidisciplinary design analysis and optimization
(MDAO), uncertainty quantification, surrogate modeling, and computational design
workflows. **Each student defines a project tied to their own graduate research** in
consultation with the instructor; a sequence of assignments builds toward it. The
objective is to formulate, implement, and evaluate a modern computational design
study using research-relevant tools and high-performance computing.

**Over the term you will:**

- Formulate a design study from your own research as a formal MDO problem and
  describe its structure with an **N2 diagram**.
- Identify and characterize sources of **uncertainty** and technology **"k-factors"**
  in that problem.
- Use **agentic AI workflows** to write design-study scripts in **Google JAX** and
  **NASA's OpenMDAO**.
- Generate data by running MDAO scripts on **VT-ARC** high-performance computing
  resources.
- Fit surrogate models with **NASA's Surrogate Modeling Toolbox (SMT)**.
- Perform uncertainty quantification on those surrogates with **UM-Bridge**, and
  interpret the results.

**Format & meetings.** Each topic is introduced in a **75-minute group lecture every
other week**, run **hybrid** — a room on campus for in-person students and Zoom for
those attending online. Between lectures, each student has a **one-hour individual
meeting** with the instructor to discuss how the topic applies to their own research.

**Assignments & grading.** After each topic you complete an **individual assignment**
that applies the topic to your research, so the assignments accumulate into your final
project. Assessment is entirely project-based: work is judged on the clarity of the
problem formulation, appropriate use of computational methods, the correctness and
reproducibility of scripts and workflows, the quality of the generated data and
surrogate models, and the interpretation of the uncertainty-quantification results.

**Reference text.** Joaquim R. R. A. Martins and Andrew Ning, *Engineering Design
Optimization*, Cambridge University Press — free online at
<https://mdobook.github.io/>. Chapter numbers in the schedule refer to this book.
The aircraft examples in this repo follow Daniel P. Raymer, *Aircraft Design: A
Conceptual Approach*.

**Prerequisites & tools.** Working Python, plus graduate-level familiarity with
your own research domain. The hands-on work uses this repository's
`eng-des-opt-course` environment (OpenMDAO, JAX, Streamlit); see the
[Quickstart](#quickstart-run-the-asw-streamlit-example) below. Additional tools
introduced during the term: SMT, UM-Bridge, and VT-ARC HPC.

## Course schedule — Fall 2026

The whole group meets **every other week**, starting the **week of August 31** (no
meeting the week of Nov 23, Thanksgiving). Most sessions cover a methods topic; two
are reserved for presentations. The project pipeline (rightmost milestone column)
advances in parallel, paced to build toward the final project. Intervening weeks are
for project work, the runnable [lessons](lessons/), and individual check-ins as
arranged. Chapter numbers refer to Martins & Ning.

| # | Week of | Topic | Project milestone | Reference & repo |
|---|---|---|---|---|
| 1 | Aug 31 | Course kickoff; **Design Structure Matrices** — N2 and XDSM; scoping your project; tour of the worked examples | Draft **project problem statement** (objective, design variables, constraints) | Ch. 1–2, 13.2–13.3; [Lesson 1 — DSM](lessons/lesson_01_dsm/docs/lesson_01_dsm.md); [ASW app](#quickstart-run-the-asw-streamlit-example), [ex_01](src/aircraft_sizing/examples/ex_01_asw/docs/ex_01_asw_sizing.md), [ex_02](src/aircraft_sizing/examples/ex_02_electrified_aircraft/docs/ex_02_electrified_aircraft.md), [ex_03](src/aircraft_sizing/examples/ex_03_e19_hybrid_electric/docs/ex_03_electrified_aircraft_sizing.md) |
| 2 | Sep 14 | **Iterative methods** — fixed-point (Gauss–Seidel), Newton, Broyden; convergence and cost | **N2 diagram** of your MDO problem | Ch. 3, App. B–C; [Lesson 2 — Iterative methods](lessons/lesson_02_iterative_methods/asw/docs/lesson_02_iterative_methods.md) |
| 3 | Sep 28 | **Gradient-based optimization & computing derivatives** — line search, BFGS, SQP; finite difference, complex step, JAX AD, adjoint | First runnable **JAX + OpenMDAO** script (agentic AI workflow) | Ch. 4–6; [Lesson 3 — Gradient-based optimization](lessons/lesson_03_gradient_based_optimization/docs/lesson_03_gradient_based_optimization.md); [ex_05 — C172 trim](src/aircraft_sizing/examples/ex_05_c172_trim/docs/ex_05_c172_trim.md) |
| 4 | Oct 12 | **Mid-semester presentations** — project-idea pitch | **Pitch**: problem statement, N2, and planned approach (incl. uncertainty sources / k-factors) | — |
| 5 | Oct 26 | **Surrogate-based optimization & optimization under uncertainty** — surrogate modeling with SMT; UQ with UM-Bridge; robust / reliability-based design | Design of experiments + **data set on VT-ARC HPC** | Ch. 10, 12 |
| 6 | Nov 9 | **Gradient-free & multi-objective optimization** — Nelder–Mead, genetic algorithms, particle swarm; Pareto fronts, weighted-sum and ε-constraint | Fitted **SMT surrogate** + **UM-Bridge UQ results** | Ch. 7, 9; [ex_05 — C172 trim](src/aircraft_sizing/examples/ex_05_c172_trim/docs/ex_05_c172_trim.md) (pymoo GA solver) |
| 7 | Dec 7 | **Final presentations** | **Final report + presentation** | — |

## Quickstart: run the ASW Streamlit example

Follow these steps to create a Conda environment, install the package into it, and launch the interactive ASW app with sliders for the mission parameters.

### 0. Prerequisites

- [Miniconda or Anaconda](https://docs.conda.io/en/latest/miniconda.html) (provides the `conda` command)
- [Git](https://git-scm.com/)

### 1. Get the code

```bash
git clone https://github.com/<your-account>/sci-comp-eng-des-opt.git
cd sci-comp-eng-des-opt
```

The E-19 electrified-aircraft example (ex_03) references a companion repository tracked as a git submodule. It is **not** required to run the ASW app, but if you want it, fetch it once with:

```bash
git submodule update --init references/ElectricAircraftDesignExample_AIAA2026
```

> Note: `conda`/`pip` do **not** download git submodules — only `git` does, via the command above (or `git clone --recurse-submodules`).

### 2. Create the Conda environment

From the repository root:

```bash
conda env create -f environment.yml --yes
```

`environment.yml` creates an environment named **`eng-des-opt-course`** (Python 3.12) and installs this package in editable mode with the extra derivative backends for the ex_05 C172 trim example (`-e .[trim,csdl]` — CasADi, Warp, pymoo, pandas, and CSDL-alpha), so the environment picks up local source edits automatically. Warp reaches an NVIDIA GPU on native Windows through its bundled CUDA runtime (just the driver is needed); JAX stays on CPU on Windows. To add the extras to an existing install: `python -m pip install -e ".[trim,csdl]"`.

### 3. Activate the environment

```bash
conda activate eng-des-opt-course
```

### 4. Launch the Streamlit app

From the repository root, with the environment active:

```bash
python -m streamlit run apps/streamlit/app.py
```

Streamlit prints a **Local URL** (default <http://localhost:8501>) and usually opens it in your browser automatically. If it doesn't, open that URL yourself. Press `Ctrl+C` in the terminal to stop the server.

### 5. Use the app

The page has a **control sidebar on the left** and **results on the right**. Results recompute only when you click **Recompute** — so you can move several controls, then apply them all at once.

**Sidebar controls** (grouped in expanders):

- **Mission — sliders.** Drag to change the four headline mission parameters: one-way cruise range, cruise altitude, mission-equipment weight, and on-station loiter endurance.
- **Propulsion — number fields.** Cruise and loiter TSFC.
- **Aerodynamics — number fields.** Wing aspect ratio and wetted-area ratio S_wet/S_ref. Maximum L/D is *computed* from these (Raymer Eq. 3.12), not entered.
- **Structure — radio.** Switch between **metal** and **composite** (composite scales the empty-weight fraction by 0.95).
- **Solver controls — number fields.** Initial TOGW guess, convergence tolerance, and maximum iterations.
- **One-at-a-time sensitivity sweep.** Pick the swept input — cruise range, cruise altitude, mission-equipment weight, aspect ratio, or cruise TSFC — then set the **absolute band** to sweep (a two-ended slider) and the number of samples. The default is a **1,000–2,000 nm range trade**, wide enough to show that TOGW grows strongly nonlinearly with range.

Cruise altitude drives the ICAO standard atmosphere — speed of sound, density, temperature, and pressure — through the [`ambiance`](https://pypi.org/project/ambiance/) package, so it is computed, not entered. Everything else (cruise Mach, crew weight, prelanding loiter, historical segment ratios, and the fuel/empty-weight allowances) stays fixed at the Raymer 3.6 baseline.

**Buttons:**

- **Recompute** — re-solve using the current sidebar values.
- **Reset baseline** — restore the approved baseline inputs and re-solve.

**Results** (top metrics + four tabs):

- **Metrics row:** Final TOGW, fuel weight, empty weight, and iteration count.
- **Overview:** mission-profile table and the sizing XDSM diagram.
- **Intermediate calculations:** derived cruise/aero quantities, mission segment ratios, mission and fuel fractions, and the final mass breakdown.
- **Convergence:** the fixed-point convergence plot and the iteration-by-iteration table.
- **Sensitivity & sanity checks:** the one-at-a-time TOGW sensitivity plot and table, plus automated sanity checks.

**Try it:** open the **Sensitivity** tab on the default 1,000–2,000 nm range trade. TOGW goes from **42,801 lb to 82,215 lb** — doubling the range nearly doubles the aircraft, and each extra 250 nm costs more than the last, because more fuel means a heavier aircraft which then burns more fuel. Then drag the **One-way cruise range** slider and click **Recompute** to move the operating point along that curve, or switch the swept input to see how each assumption drives the final weight. Click **Reset baseline** to return to the defaults.

> Past roughly **3,000 nm** this Class I method stops having an answer: fuel plus empty weight consume the whole aircraft. The app reports that as a warning and keeps the last valid result rather than showing a meaningless number.

## Maintaining the environment

If you change package metadata or dependencies, refresh the editable install (with the environment active):

```bash
python -m pip install -e .
```

## Canonical course documents

### Example narratives

- **ex_01 —** [ASW fixed-point sizing narrative](src/aircraft_sizing/examples/ex_01_asw/docs/ex_01_asw_sizing.md) (fuel-based, from Raymer)
- **ex_02 —** [Electrified aircraft — worked examples](src/aircraft_sizing/examples/ex_02_electrified_aircraft/docs/ex_02_electrified_aircraft.md) (AIAA Short Course), a set of isolated examples:
  - [Range equation](src/aircraft_sizing/examples/ex_02_electrified_aircraft/docs/01_range_equation.md)
  - [Reference airplane](src/aircraft_sizing/examples/ex_02_electrified_aircraft/docs/02_reference_airplane.md)
  - [Electric drive train (EDT) SOA energy](src/aircraft_sizing/examples/ex_02_electrified_aircraft/docs/03_edt_soa_energy.md)
  - [Reduced-order battery sizing](src/aircraft_sizing/examples/ex_02_electrified_aircraft/docs/04_battery_sizing.md)
- **ex_03 —** [Electrified aircraft (E-19) sizing narrative](src/aircraft_sizing/examples/ex_03_e19_hybrid_electric/docs/ex_03_electrified_aircraft_sizing.md) (hybrid-electric, from de Vries et al., AIAA 2026-4690)
- **ex_05 —** [Cessna-172 six-DOF trim as optimization](src/aircraft_sizing/examples/ex_05_c172_trim/docs/ex_05_c172_trim.md) (five derivative backends — numpy/JAX/CasADi/CSDL/Warp — plus a pymoo GA)

### Course lessons

- [Lesson 1 — Design Structure Matrices (DSM), N2, and XDSM](lessons/lesson_01_dsm/docs/lesson_01_dsm.md)
- [Lesson 2 — Iterative methods: fixed-point vs Newton vs Broyden, gradients vs finite difference](lessons/lesson_02_iterative_methods/asw/docs/lesson_02_iterative_methods.md)
- [Lesson 3 — Gradient-based optimization & computing derivatives (analytic / FD / complex-step / AD)](lessons/lesson_03_gradient_based_optimization/docs/lesson_03_gradient_based_optimization.md)

### Reference notes

- [Electrified-aircraft equations](src/aircraft_sizing/examples/ex_02_electrified_aircraft/docs/electrified_aircraft_equations.md)

## Deploying to Streamlit Community Cloud

Set the app entry point to `apps/streamlit/app.py` and keep the repository-root `requirements.txt` as the install source — it installs this package for deployment. Use the app path above as the launch target.
