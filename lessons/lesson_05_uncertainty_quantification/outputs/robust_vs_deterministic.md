# Lesson 5 - Robust vs. deterministic optimum (generated)

| Quantity | Deterministic | Robust |
|---|---|---|
| z1 | 1.9776 | 2.0962 |
| z2 | 0.0000 | 0.0000 |
| x | 0.0000 | 0.0617 |
| E[obj] | 3.1851 | 3.6781 |
| mean const1 | -0.0006 | -0.4949 |
| **P(g1 violated)** | **50.0%** | **2.5%** |

- Baseline is the canonical Sellar optimum (`obj = 3.1834`), which holds `g1` exactly
  and so is infeasible ~50% of the time under uncertainty.
- The robust reformulation (`min E[obj]` s.t. `CI_upper(g) <= 0`) raises `y1` (mostly via `z1`)
  to hold `g1` across the scatter: the violation probability drops from **50%** to **2.5%**.
- The price is a higher expected objective (3.19 -> 3.68): **reliability costs performance**.
- Statistics come from UQPCE's order-2 polynomial chaos on 60 collocation points, evaluated
  inside every optimizer iteration.