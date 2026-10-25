# Lesson 5 - Polynomial chaos (UQPCE) vs. Monte Carlo (generated)

Both evaluated at the deterministic optimum z = [1.9776, 0.0000], x = 0.0000.

**PCE used 60 model runs; Monte Carlo used 100,000.**

| Output | PCE mean (analytic) | MC mean | PCE std (analytic) | MC std | PCE 95% CI | MC 95% CI |
|---|---|---|---|---|---|---|
| obj | 3.1840 | 3.1847 | 0.3873 | 0.3882 | [2.459, 3.977] | [2.459, 3.978] |
| con1 | -0.0003 | -0.0003 | 0.2238 | 0.2236 | [-0.438, 0.436] | [-0.438, 0.435] |
| con2 | -20.2461 | -20.2455 | 0.1487 | 0.1482 | [-20.530, -19.957] | [-20.530, -19.957] |

- Probability of violating the active constraint `const1` (y1 < 3.16): PCE **50.0%** vs Monte Carlo **50.0%**.
- PCE mean/variance come straight from the 60-run polynomial coefficients; the CI and
  violation probability come from sampling the *cheap* fitted polynomial.
- Same accuracy as Monte Carlo at ~1666x fewer true model evaluations - the reason PCE
  is affordable *inside* an optimizer (next script).