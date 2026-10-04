# Lesson 3 - Sellar optimization by derivative source (generated)

Discipline calls = total d1+d2 `compute` evaluations over the whole optimization. Analytic and JAX partials add *no* extra model calls (the Jacobian is formed in closed form); complex-step and forward finite difference perturb each input once per gradient, and central finite difference perturbs it twice -- so the model-call cost climbs even though the SLSQP path (iters, gradient evaluations) is identical.

| Derivative source | f* | |dx*| vs analytic | iters | gradient evals | discipline calls |
|---|---|---|---|---|---|
| analytic | 3.183394 | 0.00e+00 | 7 | 6 | 126 |
| jax | 3.183394 | 0.00e+00 | 7 | 6 | 126 |
| cs | 3.183394 | 0.00e+00 | 7 | 6 | 168 |
| fd-forward | 3.183394 | 7.67e-13 | 7 | 6 | 168 |
| fd-central | 3.183394 | 9.37e-14 | 7 | 6 | 210 |
