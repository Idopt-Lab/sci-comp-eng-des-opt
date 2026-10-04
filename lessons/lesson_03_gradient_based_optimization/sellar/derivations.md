# Sellar derivatives — the math behind `02_sellar_derivatives.py`

Companion to [`02_sellar_derivatives.py`](02_sellar_derivatives.py). It works through
(3) the **analytic partials** that `check_partials` verifies, and (2) the **coupled
totals** computed by hand with the direct and adjoint methods — including *how*
OpenMDAO turns partials into totals and *why* `LinearRunOnce` gives the wrong total
across a feedback loop while `DirectSolver` gives the right one.

## The Sellar problem

Two coupled disciplines, an objective, and two constraints:

$$y_1 = z_1^2 + z_2 + x - 0.2\,y_2, \qquad y_2 = \sqrt{y_1} + z_1 + z_2,$$

$$f = x^2 + z_2 + y_1 + e^{-y_2}, \qquad g_1 = 3.16 - y_1 \le 0, \qquad g_2 = y_2 - 24 \le 0.$$

Design variables $x,\ z_1,\ z_2$; coupling variables (states) $y_1,\ y_2$. The two
disciplines form a loop ($y_2$ feeds $d_1$, $y_1$ feeds $d_2$), so for any
$(x,z_1,z_2)$ the states come from a **multidisciplinary analysis (MDA)** — the
fixed-point / Newton solve of Lesson 2.

---

## (3) Analytic partials

A **partial** derivative is *local* to one component: hold that component's inputs
independent and differentiate its outputs. These are what `compute_partials` returns
and what `check_partials` compares against a finite-difference / complex-step
reference.

**Discipline 1**, $y_1 = z_1^2 + z_2 + x - 0.2\,y_2$:

$$\frac{\partial y_1}{\partial z_1} = 2z_1, \quad
  \frac{\partial y_1}{\partial z_2} = 1, \quad
  \frac{\partial y_1}{\partial x} = 1, \quad
  \frac{\partial y_1}{\partial y_2} = -0.2 .$$

**Discipline 2**, $y_2 = y_1^{1/2} + z_1 + z_2$:

$$\frac{\partial y_2}{\partial z_1} = 1, \quad
  \frac{\partial y_2}{\partial z_2} = 1, \quad
  \frac{\partial y_2}{\partial y_1} = \tfrac{1}{2}\,y_1^{-1/2} .$$

**Objective**, $f = x^2 + z_2 + y_1 + e^{-y_2}$:

$$\frac{\partial f}{\partial x} = 2x, \quad
  \frac{\partial f}{\partial z_2} = 1, \quad
  \frac{\partial f}{\partial y_1} = 1, \quad
  \frac{\partial f}{\partial y_2} = -e^{-y_2}, \quad
  \frac{\partial f}{\partial z_1} = 0 .$$

**Constraints**: $\partial g_1/\partial y_1 = -1$ and $\partial g_2/\partial y_2 = 1$.

`check_partials` confirms every one of these to a relative error below $10^{-6}$ (the
script asserts it). These partials are the *only* calculus in the problem; everything
below assembles them into totals with linear algebra.

---

## (2) Coupled totals: how OpenMDAO turns partials into totals

### Residual form

Write the MDA as residuals that vanish at the converged state. With states
$u = (y_1, y_2)$ and (say) design variable $z_1$:

$$\mathcal{R}_1 = y_1 - \big(z_1^2 + z_2 + x - 0.2\,y_2\big), \qquad
  \mathcal{R}_2 = y_2 - \big(\sqrt{y_1} + z_1 + z_2\big),$$

and the converged state satisfies $\mathcal{R}(u, x) = 0$.

### The unified derivatives equation

Differentiate $\mathcal{R}(u, z_1) = 0$ at the converged point (implicit function
theorem):

$$\frac{\partial \mathcal{R}}{\partial u}\,\frac{\mathrm{d}u}{\mathrm{d}z_1}
   + \frac{\partial \mathcal{R}}{\partial z_1} = 0
   \;\;\Longrightarrow\;\;
   \frac{\mathrm{d}u}{\mathrm{d}z_1}
   = -\Big(\frac{\partial \mathcal{R}}{\partial u}\Big)^{-1}
      \frac{\partial \mathcal{R}}{\partial z_1}.$$

The total derivative of the objective then chains the explicit and implicit parts:

$$\frac{\mathrm{d}f}{\mathrm{d}z_1}
   = \frac{\partial f}{\partial z_1}
   + \frac{\partial f}{\partial u}\,\frac{\mathrm{d}u}{\mathrm{d}z_1}
   = \frac{\partial f}{\partial z_1}
   - \frac{\partial f}{\partial u}
     \Big(\frac{\partial \mathcal{R}}{\partial u}\Big)^{-1}
     \frac{\partial \mathcal{R}}{\partial z_1}.$$

This is OpenMDAO's **unified derivatives equation**: it needs only the *partials*
($\partial\mathcal{R}/\partial u$, $\partial\mathcal{R}/\partial z_1$,
$\partial f/\partial u$, $\partial f/\partial z_1$) plus one **linear solve** with the
Jacobian $\partial\mathcal{R}/\partial u$.

### The Jacobian blocks (built from the partials above)

$$\frac{\partial \mathcal{R}}{\partial u} =
  \begin{bmatrix}
    \dfrac{\partial \mathcal{R}_1}{\partial y_1} & \dfrac{\partial \mathcal{R}_1}{\partial y_2}\\[2mm]
    \dfrac{\partial \mathcal{R}_2}{\partial y_1} & \dfrac{\partial \mathcal{R}_2}{\partial y_2}
  \end{bmatrix}
  =
  \begin{bmatrix} 1 & 0.2 \\ -\tfrac{1}{2}y_1^{-1/2} & 1 \end{bmatrix},
  \qquad
  \frac{\partial \mathcal{R}}{\partial z_1} =
  \begin{bmatrix} -2z_1 \\ -1 \end{bmatrix}.$$

The **off-diagonal** entries are the feedback coupling: $\partial\mathcal{R}_1/\partial y_2 = +0.2$
(because $y_2$ enters $d_1$ with coefficient $-0.2$) and
$\partial\mathcal{R}_2/\partial y_1 = -\tfrac12 y_1^{-1/2}$. Also
$\partial f/\partial u = [\,1,\ -e^{-y_2}\,]$ and $\partial f/\partial z_1 = 0$.

These are exactly the matrices assembled in `hand_coupled_totals(...)`.

### Direct vs. adjoint — two ways to do the one linear solve

**Direct** (solve for $\mathrm{d}u/\mathrm{d}z_1$, one solve *per design variable*):

$$\frac{\partial \mathcal{R}}{\partial u}\,\phi = -\frac{\partial \mathcal{R}}{\partial z_1},
  \qquad
  \frac{\mathrm{d}f}{\mathrm{d}z_1} = \frac{\partial f}{\partial z_1} + \frac{\partial f}{\partial u}\cdot\phi .$$

**Adjoint** (solve for $\psi$, one solve *per output / function*):

$$\Big(\frac{\partial \mathcal{R}}{\partial u}\Big)^{\!\top}\psi = \Big(\frac{\partial f}{\partial u}\Big)^{\!\top},
  \qquad
  \frac{\mathrm{d}f}{\mathrm{d}z_1} = \frac{\partial f}{\partial z_1} - \psi\cdot\frac{\partial \mathcal{R}}{\partial z_1}.$$

Both return the identical total. **Direct costs one linear solve per design variable;
adjoint costs one per function.** So adjoint wins when there are many design variables
and few outputs (the usual optimization case), and direct wins the opposite shape —
the same forward-vs-reverse trade-off as the AD modes in `03_computational_graph.py`.

### Worked numbers (at $z = (5, 2)$, $x = 1$)

The MDA converges to $y_1 \approx 25.588$, $y_2 \approx 12.059$, so
$y_1^{-1/2} \approx 0.1977$ and $e^{-y_2} \approx 5.8\times10^{-6}$. All four routes
agree (the script asserts this):

| method | $\mathrm{d}f/\mathrm{d}z_1$ |
|---|---|
| hand, direct | **9.610011** |
| hand, adjoint | **9.610011** |
| OpenMDAO (`DirectSolver`) | **9.610011** |
| central-FD reference | **9.610011** |
| OpenMDAO (`LinearRunOnce`) | 9.999988  ❌ |

---

## Why `LinearRunOnce` is wrong and `DirectSolver` is right

OpenMDAO solves the unified derivatives equation with whatever **linear solver** the
group carries. The quality of the total depends entirely on how accurately that solver
inverts the coupled Jacobian $\partial\mathcal{R}/\partial u$.

* **`DirectSolver`** LU-factorizes the full $2\times2$ block and applies the exact
  inverse — one factorization gives the exact $\mathrm{d}u/\mathrm{d}z_1$, hence the
  exact total.
* **`LinearRunOnce`** makes a *single* block Gauss–Seidel sweep. One sweep inverts a
  *triangular* (feed-forward) system exactly, but across a feedback loop it drops the
  back-coupling, so the total is wrong — exactly as a single NLBGS sweep fails to
  solve the nonlinear MDA.

**Concretely.** The linear system for $\mathrm{d}u/\mathrm{d}z_1 = (\dot y_1, \dot y_2)$ is

$$\dot y_1 + 0.2\,\dot y_2 = 2z_1 = 10, \qquad -\tfrac12 y_1^{-1/2}\,\dot y_1 + \dot y_2 = 1.$$

A single forward sweep takes $\dot y_2^{\text{old}} = 0$, so from the first equation
$\dot y_1 = 10$ (it **ignores** the $0.2\,\dot y_2$ feedback), then
$\dot y_2 = 1 + \tfrac12 y_1^{-1/2}(10) \approx 1.988$. With
$\partial f/\partial u = [1,\,-e^{-y_2}]$ that gives

$$\frac{\mathrm{d}f}{\mathrm{d}z_1}\Big|_{\text{one sweep}} \approx 1\cdot 10 - 5.8\times10^{-6}\cdot 1.988 \approx 9.99999,$$

the wrong `LinearRunOnce` value. Solving the coupled system exactly instead gives
$\dot y_1 = 9.610$ (the feedback correction $-0.2\,\dot y_2$ pulls $10 \to 9.61$), hence
$\mathrm{d}f/\mathrm{d}z_1 = 9.610$. The entire error is the dropped feedback term.

**Rule of thumb.** The *linear* solver used for derivatives must match the coupling of
the *nonlinear* model: feed-forward models are fine with `LinearRunOnce`; any feedback
loop needs `DirectSolver` (or an iterative `LinearBlockGS`/Krylov solver driven to
convergence). This is why `build_problem(..., linear_solver="direct")` is the correct
choice for the Sellar MDA.
