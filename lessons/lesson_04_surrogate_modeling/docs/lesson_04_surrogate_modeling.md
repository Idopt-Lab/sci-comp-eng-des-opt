# Lesson 4 — Surrogate modeling

In Lessons 2–3 the Sellar disciplines were cheap analytic
functions we could call as often as we liked. Real analyses (CFD, FEA, a full
mission sim) are *expensive*, so we often can't afford to call them inside an
optimizer. The fix: run the expensive analysis a fixed number of times at a
**designed** set of points, fit a cheap **surrogate** (a.k.a. metamodel / response
surface) to that data, and optimize through the surrogate instead.

We keep the exact same Sellar problem and the exact same Lesson-3 optimizer — only
the discipline internals change (analytic → surrogate). The lesson answers three
questions: **where do I sample (DoE)?**, **what do I fit (surrogate family)?**, and
**how do I know it's any good (goodness-of-fit)?**

Every script is runnable and writes committed `outputs/` (PNG/MD) so the figures
render on GitHub without re-running. Run from the repository root with the
`eng-des-opt-course` environment active.

## Scripts, in order

| # | Script | What it teaches | Run |
|---|---|---|---|
| — | `sellar_functions.py` | the two disciplines as plain-numpy functions (the "truth" the DoE samples) + their sampling boxes | `python lessons/lesson_04_surrogate_modeling/sellar_functions.py` |
| 1 | `01_doe_sampling.py` | DoE techniques — full-factorial vs. random vs. LHS; space-filling; the curse of dimensionality | `python lessons/lesson_04_surrogate_modeling/01_doe_sampling.py` |
| 2 | `02_fit_surrogates.py` | fit QP / RBF / KRG with SMT; judge fit with held-out R²/RMSE/MAE, parity plots, k-fold CV, and a learning curve | `python lessons/lesson_04_surrogate_modeling/02_fit_surrogates.py` |
| 3 | `03_surrogate_mda_opt.py` | drop trained surrogates into the Lesson-3 MDA+SLSQP and recover the optimum; exact surrogate gradients; the validity-domain caveat | `python lessons/lesson_04_surrogate_modeling/03_surrogate_mda_opt.py` |

Supporting module `surrogate_component.py` holds `SmtSurrogateComp`, an OpenMDAO
discipline with the same interface as the analytic `SellarDis1/2` whose response
comes from a trained SMT model.

Diagrams: `python lessons/lesson_04_surrogate_modeling/generate_xdsm.py` renders
`outputs/sellar_doe_sampling.*` (the offline DOE-driver sampling step) and
`outputs/sellar_surrogate_mdf.*` (the Lesson-3 MDF with `METAMODEL` boxes in place
of the disciplines). Needs `pdflatex` + `pdftoppm`.

## The surrogate workflow

```
DoE  ->  evaluate the true discipline at the sample points  ->  fit a surrogate
     ->  check the fit (held-out error / CV)  ->  optimize through the surrogate
```

The DoE samples each discipline *separately* over its input box:

- Discipline 1: inputs `[z1, z2, x, y2]` → `y1`, over `z1∈[-10,10], z2∈[0,10], x∈[0,10], y2∈[0,24]`
- Discipline 2: inputs `[z1, z2, y1]` → `y2`, over `z1∈[-10,10], z2∈[0,10], y1∈[0,40]`

Inputs are scaled to `[0,1]` before fitting. Kriging rescales internally and does
not care, but distance-based **RBF is useless on raw inputs** (negative R²) and
excellent on scaled ones — normalize before fitting a distance-based surrogate.

## Judging a surrogate: goodness-of-fit

A surrogate that reproduces its *training* points proves nothing — an interpolator
(RBF, KRG) hits them exactly by construction. What matters is accuracy where it has
**never been evaluated**. So `02_fit_surrogates.py` draws an independent **held-out
test set** (300 LHS points) and scores the predictions $\hat{y}$ against the truth $y$
there. Let $N$ be the number of test points and $\bar{y}$ the mean of $y$.

### Error metrics

$$
\mathrm{RMSE} = \sqrt{\frac{1}{N}\sum_{i=1}^{N}\left(\hat{y}_i - y_i\right)^2}
\qquad\qquad
\mathrm{MAE} = \frac{1}{N}\sum_{i=1}^{N}\left|\hat{y}_i - y_i\right|
$$

$$
R^2 = 1 - \frac{SS_{\text{res}}}{SS_{\text{tot}}}, \qquad
SS_{\text{res}} = \sum_{i=1}^{N}\left(\hat{y}_i - y_i\right)^2, \qquad
SS_{\text{tot}} = \sum_{i=1}^{N}\left(y_i - \bar{y}\right)^2
$$

- **RMSE** (root-mean-square error) and **MAE** (mean absolute error) are *absolute*:
  $0$ is perfect and they carry the units of the output, so whether a value is "small"
  depends on the output's own scale. RMSE squares the errors, so it punishes a few
  large misses harder than MAE does.
- **$R^2$** (coefficient of determination) is *scale-free*: $1$ is perfect, $0$ is no
  better than always predicting the mean $\bar{y}$, and a **negative** $R^2$ means the
  surrogate is *worse* than that constant guess — a red flag, not a rounding artifact
  (this is exactly what un-normalized RBF scores here).

### k-fold cross-validation

A single train/test split can be lucky or unlucky, especially with few points.
**k-fold cross-validation** removes that luck: split the training data into `k` equal
folds (`k = 5` here); for each fold, train on the other `k-1` folds and score on the
held-out fold; rotate so **every point is predicted exactly once by a model that never
saw it**. The reported CV score is the mean (± std) of $R^2$ across the folds. When the
CV score matches the held-out test score, the fit is stable rather than an artifact of
one particular split.

### Learning curve

How much data is "enough"? A **learning curve** re-fits the surrogate at growing
training-set sizes (`n = 8, 16, 32, 64, 128, 256, 512`) and plots the held-out error
versus `n` on **log–log axes**. If the error falls and then **flattens**, the DoE is
already dense enough; if it keeps dropping at the largest `n`, more samples would still
pay off. Here the response is smooth and **noise-free**, so there is no irreducible
error floor: KRG keeps improving — roughly a power law, with a visible
diminishing-returns bend around `n ≈ 64–128` — reaching RMSE ≈ 1.7e-3 at `n = 512`,
while RBF's stiffer kernel stays well above it throughout. A true plateau instead shows
up when the data are **noisy** or the surrogate has a fixed **bias**. We use a log RMSE
axis because $R^2$ saturates near 1 on these smooth functions and hides the trend.

### The output files, and how to read them

- **`outputs/surrogate_parity.png`** — a 2×3 grid of *parity* (predicted-vs-true)
  plots: rows are the two disciplines, columns are QP / RBF / KRG, each titled with its
  held-out R². Read it by eye: points hugging the dashed 45° line = accurate; a tilt or
  bend away from the line = systematic **bias** (watch QP on D2's `sqrt` term); fuzzy
  vertical scatter about the line = random error.
- **`outputs/surrogate_cv.png`** — *left*: 5-fold CV R² as bars per model and
  discipline (note the zoomed `[0.9, 1.0]` axis, so small bar differences are real);
  *right*: the D2 learning curve — test RMSE vs. training size for RBF and KRG on
  **log–log axes** (sizes `n = 8 … 512`), where the downhill slope is "more data →
  less error."
- **`outputs/surrogate_metrics.md`** — the generated table of R² / RMSE / MAE and the
  5-fold CV R² for every model and discipline, plus the learning-curve numbers behind
  the right-hand plot.

## Terminology cheat sheet

- **Design of Experiments (DoE)** — the chosen set of sample points.
  **Full factorial** = regular grid (`levels**n_inputs` runs — explodes with
  dimension). **Random / Monte Carlo** = i.i.d. uniform draws (clumps, gaps).
  **Latin Hypercube (LHS)** = stratified, space-filling, one sample per bin per
  axis; the default for surrogate training.
- **Surrogate / metamodel / response surface** — a cheap function fit to data that
  approximates an expensive analysis. **QP** = quadratic polynomial (global,
  low-flexibility, biased on non-polynomial data). **RBF** = radial basis
  functions (interpolator from distance kernels). **KRG** = Kriging / Gaussian
  process (interpolator + a predicted-variance error estimate).
- **Goodness-of-fit** — R², RMSE, MAE, k-fold cross-validation, and learning curves;
  see [Judging a surrogate](#judging-a-surrogate-goodness-of-fit) above for the
  formulae and how to read each figure.
- **Validity domain** — a surrogate is only trustworthy *inside* the box it was
  trained on; outside it extrapolates and its "optimum" is meaningless.

## Key numbers (acceptance)

- Fit quality (held-out, dense DoE, 300 test points): QP fits the quadratic D1 to
  machine precision (RMSE ≈ 3e-14); KRG reaches **R² ≈ 1** on both disciplines;
  RBF needs normalized inputs (then R² ≈ 0.997–0.9998).
- Learning curve on D2 (KRG): test RMSE falls from ≈ 2.4e-1 at `n=8` to ≈ 1.7e-3
  at `n=512` — a smooth, noise-free response keeps rewarding more data (no irreducible
  floor; just diminishing returns).
- Surrogate-driven optimum matches the analytic Lesson-3 optimum
  **f\* = 3.183394** to ≈ 3.5e-4, at `z ≈ (1.978, 0)`, `x ≈ 0`, with `g1` active
  (`y1 = 3.16`) — **using the surrogate's exact `predict_derivatives` gradients**.
  Finite-differencing the surrogate instead sends SLSQP to the wrong point
  (`f ≈ 4.34`): don't FD a noisy model when it can hand you an exact Jacobian.
- The surrogate MDA is converged to a tolerance (`1e-6`) **matched to the surrogate's
  accuracy**, not the `1e-10` the exact analytic model reaches. A coupled system of
  approximations can only be made as self-consistent as the approximations are
  accurate; demanding more just makes the fixed-point solver stall at its residual
  floor and print spurious "failed to converge" warnings.
