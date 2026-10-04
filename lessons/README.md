# Course lessons

Hands-on lessons in scientific-computing methods for engineering design
optimization. Every lesson uses the same shared model — the OpenMDAO + JAX ASW
conceptual-sizing example in
[`src/aircraft_sizing/examples/ex_01_asw`](../src/aircraft_sizing/examples/ex_01_asw) —
so the ideas build on one concrete artifact instead of a new toy each time.

Run all commands from the repository root with the `eng-des-opt-course` environment
active (`conda activate eng-des-opt-course`; see the top-level [README](../README.md)).

## Lessons

1. **[Design Structure Matrices (DSM), N2, and XDSM](lesson_01_dsm/docs/lesson_01_dsm.md)**
   — read a model's coupling structure; generate OpenMDAO's automatic N2 and a
   hand-authored pyXDSM, then meet the other XDSM shapes (optimizer, DOE, bi-level,
   `stack=True`).
   ```bash
   python lessons/lesson_01_dsm/generate_n2.py     # -> outputs/asw_n2.html
   python lessons/lesson_01_dsm/generate_xdsm.py   # -> outputs/asw_xdsm.pdf/.png
   python lessons/lesson_01_dsm/generate_shapes.py # -> outputs/shape_*.pdf/.png
   ```

2. **[Iterative methods](lesson_02_iterative_methods/asw/docs/lesson_02_iterative_methods.md)**
   — fixed point vs Newton vs Broyden, counting iterations and
   function/derivative evaluations; analytic (JAX) gradients vs finite difference.
   ```bash
   python lessons/lesson_02_iterative_methods/asw/compare_solvers.py
   # -> outputs/solver_comparison.md, outputs/convergence.png
   ```

3. **[Gradient-based optimization & computing derivatives](lesson_03_gradient_based_optimization/docs/lesson_03_gradient_based_optimization.md)**
   — wrap the Sellar MDA in SLSQP (KKT, iteration vs evaluation counts); analytic vs
   finite-difference vs complex-step vs JAX derivatives; automatic differentiation
   from scratch; and the ASW sizing loop recast as an optimization.
   ```bash
   python lessons/lesson_03_gradient_based_optimization/sellar/01_sellar_optimization.py
   python lessons/lesson_03_gradient_based_optimization/sellar/02_sellar_derivatives.py
   python lessons/lesson_03_gradient_based_optimization/autodiff/03_computational_graph.py
   python lessons/lesson_03_gradient_based_optimization/asw/04_asw_as_optimization.py
   python lessons/lesson_03_gradient_based_optimization/c172/05_c172_trim.py  # capstone: ex_05 C172 trim, 5 backends
   python lessons/lesson_03_gradient_based_optimization/generate_xdsm.py  # -> outputs/*.pdf/.png
   ```

Each lesson has a `docs/` narrative, runnable scripts, and committed `outputs/` so
the figures render on GitHub without re-running anything. Rendering the XDSM to
PDF/PNG additionally needs `pdflatex` and `pdftoppm` on PATH.
