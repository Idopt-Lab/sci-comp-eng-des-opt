# Lesson 4 - Surrogate vs. analytic Sellar optimum (generated)

| Quantity | Surrogate | Analytic | abs. error |
|---|---|---|---|
| obj | 3.183040 | 3.183394 | 3.54e-04 |
| z1 | 1.978387 | 1.977639 | 7.48e-04 |
| z2 | 0.000000 | 0.000000 | 1.13e-14 |
| x | 0.000000 | 0.000000 | 8.98e-15 |
| y1 | 3.160000 | 3.160000 | 1.10e-07 |
| y2 | 3.770521 | 3.755278 | 1.52e-02 |

- Discipline 1 surrogate evaluations: 36
- Discipline 2 surrogate evaluations: 36

The optimizer runs the identical Lesson-3 MDF setup; only the discipline internals changed
(analytic -> Kriging). The optimum matches the analytic solution because the surrogates are
accurate across the trajectory, which stays inside the training box.