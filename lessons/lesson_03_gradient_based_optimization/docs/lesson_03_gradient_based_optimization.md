# Lesson 3 — Gradient-based optimization & computing derivatives

Week 3 of the course. Having built the Sellar MDA (Lesson 2) and counted solver
iterations, we now **wrap the analysis in an optimizer** and study the one ingredient
that makes gradient-based optimization work at scale: **derivatives**. The lesson
culminates in a real engineering problem — a Cessna-172 six-DOF trim (`ex_05`) solved
with five derivative backends (finite difference, JAX, CasADi, CSDL, and Warp on the
GPU) — tying the derivative ideas back to a concrete sizing-adjacent model.

Every script is runnable and writes committed `outputs/` (PNG/MD) so the figures
render on GitHub without re-running. Run from the repository root with the
`eng-des-opt-course` environment active.

## Scripts, in order

| # | Script | What it teaches | Run |
|---|---|---|---|
| 1 | `sellar/01_sellar_optimization.py` | NLP formulation; wrap the Sellar MDA in SLSQP; iteration history; counts (iterations ≪ model/discipline evals); KKT at the optimum | `python lessons/lesson_03_gradient_based_optimization/sellar/01_sellar_optimization.py` |
| 2 | `sellar/02_sellar_derivatives.py` | partials vs totals; coupled totals by hand (direct & adjoint) vs `compute_totals`; why `DirectSolver` not `LinearRunOnce`; analytic/FD/complex-step/JAX derivatives; FD error & solver-tolerance floor (full math in [`sellar/derivations.md`](../sellar/derivations.md)) | `python lessons/lesson_03_gradient_based_optimization/sellar/02_sellar_derivatives.py` |
| 3 | `autodiff/03_computational_graph.py` | automatic differentiation from scratch: a `Var` tape, the computational graph, forward (dual numbers) vs reverse (adjoint) mode, then the same in JAX | `python lessons/lesson_03_gradient_based_optimization/autodiff/03_computational_graph.py` |
| 4 | `asw/04_asw_as_optimization.py` | the ASW sizing loop recast as an optimization (optimizer replaces the solver); compared against the Lesson-2 nonlinear solvers | `python lessons/lesson_03_gradient_based_optimization/asw/04_asw_as_optimization.py` |
| 5 | `c172/05_c172_trim.py` | **capstone** — a Cessna-172 six-DOF trim solved five ways (finite-difference / JAX / CasADi / CSDL / Warp-on-GPU); every exact backend walks the identical optimizer path. Drives [`ex_05_c172_trim`](../../../src/aircraft_sizing/examples/ex_05_c172_trim/docs/ex_05_c172_trim.md) | `python lessons/lesson_03_gradient_based_optimization/c172/05_c172_trim.py` |

Diagrams: `python lessons/lesson_03_gradient_based_optimization/generate_xdsm.py`
renders `outputs/sellar_mdf_opt.*` and `outputs/asw_opt.*` (needs `pdflatex` +
`pdftoppm`). Script 4 drives the shared `ex_01_asw` model through its new
`build_asw_optimization_problem` factory (added in `src/aircraft_sizing/examples/ex_01_asw`).

## The general nonlinear program (`fmincon` form)

```
min_x   f(x)
s.t.    c(x)    <= 0      nonlinear inequality
        c_eq(x)  = 0      nonlinear equality
        A x     <= b      linear inequality
        A_eq x   = b_eq   linear equality
        lb <= x <= ub     bounds
```

## Terminology cheat sheet

- **Design variables (DVs)** `x` — what the optimizer chooses. **Bounds** `lb, ub` — box limits on each DV.
- **Objective** `f` — the scalar being minimized. **Feasible set** — points satisfying all constraints.
- **Constraints** — nonlinear/linear, inequality (`<= 0`) or equality (`= 0`). An **active** constraint holds with equality at the optimum.
- **Lagrangian** `L = f + Σ λ_i g_i`; **Lagrange multipliers** `λ_i ≥ 0` price each constraint.
- **KKT conditions** — first-order optimality: stationarity (`∇L = 0`), primal/dual feasibility, complementary slackness (`λ_i g_i = 0`).
- **Iteration** — one optimizer step; **function evaluation** — one model solve (one iteration costs several). **Derivative evaluation** — one gradient/Jacobian.
- **Tolerance** — convergence target; **exit flag** — why the solver stopped (success, max-iter, infeasible, …).
- **Algorithms**: interior-point, SQP (sequential quadratic programming), active-set, trust-region-reflective.

## `fmincon` ↔ `scipy.optimize.minimize` ↔ OpenMDAO

| Concept | MATLAB `fmincon` | `scipy.optimize.minimize` | OpenMDAO |
|---|---|---|---|
| design variables | `x0`, `lb`, `ub` | `x0`, `bounds=` | `add_design_var(name, lower=, upper=, ref=)` |
| objective | `fun` | `fun` | `add_objective(name, ref=)` |
| nonlinear inequality | `nonlcon` → `c <= 0` | `constraints` (`'ineq'`) | `add_constraint(name, upper=0)` / `lower=` |
| nonlinear equality | `nonlcon` → `ceq = 0` | `constraints` (`'eq'`) | `add_constraint(name, equals=0)` |
| linear constraints | `A,b,Aeq,beq` | `LinearConstraint` | `add_constraint(..., linear=True)` |
| algorithm | `'sqp'`, `'interior-point'` | `method='SLSQP'`, `'trust-constr'` | `ScipyOptimizeDriver(optimizer='SLSQP')`, `pyOptSparseDriver` |
| scaling | `TypicalX` | — | `ref=` on DV/objective/constraint |

## Key numbers (acceptance)

- Sellar optimum: **f\* = 3.183394**, `z = (1.977639, 0)`, `x = 0`, `y1 = 3.16`, `y2 = 3.755278`.
- g1 = `3.16 − y1` is **active** (≈0, λ1 > 0); g2 = `y2 − 24` is **inactive** (≈ −20.2).
- Analytic, finite-difference (forward/central), complex-step, and JAX derivatives all
  reach the same optimum; the computational-graph example gives
  `∇f(2,5) = (5.5, 2 − cos 5)`.
- ASW-as-optimization: both formulations (min-weight, residual-least-squares) and all
  four Lesson-2 solvers close on the same takeoff weight, **W_TO = 57,615.87 lb**.
- C172 trim (M = 0.1): **θ = 9.505408°, δ_e = −9.581727°, ω = 1823.38 RPM**; all five
  derivative backends (numpy-FD / JAX / CasADi / CSDL / Warp) reach it, and the four
  exact backends take identical iteration counts.
