# Lesson 5 - Monte Carlo forward UQ (generated)

Deterministic optimum (uncertain params fixed at their means = classic Sellar): z = [1.9776, 0.0000], x = 0.0000, obj = 3.183394.

Propagated with 100,000 Monte Carlo samples at that design:

| Output | mean | std | 95% CI | P(violation) |
|---|---|---|---|---|
| obj | 3.1824 | 0.3866 | [2.452, 3.966] | - |
| const1 = 3.16 - y1 | 0.0006 | 0.2231 | - | 50.2% |
| const2 = y2 - 24 | -20.2470 | 0.1483 | - | 0.0% |

## Sobol sensitivity of obj

| input | first-order S_i | total S_Ti |
|---|---|---|
| a0 | 0.275 | 0.331 |
| a1 | 0.004 | 0.003 |
| a2 | 0.007 | 0.003 |
| a3 | 0.702 | 0.664 |

The deterministic optimum puts the active constraint `const1` right on its boundary, so under
uncertainty it is violated **50%** of the time - a design that is optimal on paper but
unreliable in practice. `const2` keeps a wide margin and never binds. This motivates the
robust reformulation in `04_robust_optimization.py`.