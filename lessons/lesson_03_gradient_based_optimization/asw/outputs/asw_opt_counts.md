`analytic evals` uses exact JAX partials (no finite-difference perturbations); `FD evals` finite-differences every required derivative, so the extra calls (`FD evals` − `analytic evals`) are exactly the model calls spent building derivatives by finite difference. The optimizers stay cheap even with FD because there is a **single design variable** (W_TO) and OpenMDAO's relevance analysis finite-differences only the two components on the W_TO → residual path (Structures, SizingResidual); the upstream disciplines do not depend on W_TO and are never perturbed. The nonlinear solvers re-linearize the full coupled model every iteration, so their FD cost is far higher.

| method | iterations | analytic evals | FD evals | FD overhead | W_TO (lb) |
|---|---|---|---|---|---|
| `solver:nlbgs` | 17 | 85 | 85 | +0 | 57,615.87 |
| `solver:nlbgs_aitken` | 7 | 35 | 35 | +0 | 57,615.87 |
| `solver:newton` | 7 | 110 | 243 | +133 | 57,615.87 |
| `solver:broyden` | 11 | 285 | 361 | +76 | 57,615.87 |
| `opt:min_weight` | 5 | 13 | 25 | +12 | 57,615.87 |
| `opt:residual_ls` | 7 | 17 | 32 | +15 | 57,615.87 |
