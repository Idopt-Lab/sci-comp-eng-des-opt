# Lesson 4 - Surrogate goodness-of-fit (generated)

Held-out test set of 300 independent LHS points; inputs scaled to [0, 1].

| Discipline | Model | R^2 | RMSE | MAE | 5-fold CV R^2 |
|---|---|---|---|---|---|
| d1 | QP | 1.00000 | 2.685e-14 | 2.135e-14 | 1.00000 +/- 0.0e+00 |
| d1 | RBF | 0.99749 | 1.504e+00 | 1.014e+00 | 0.96798 +/- 2.2e-03 |
| d1 | KRG | 1.00000 | 1.895e-05 | 1.334e-05 | 1.00000 +/- 2.6e-12 |
| d2 | QP | 0.99960 | 1.323e-01 | 9.859e-02 | 0.99883 +/- 1.2e-03 |
| d2 | RBF | 0.99976 | 1.026e-01 | 5.988e-02 | 0.99883 +/- 7.8e-04 |
| d2 | KRG | 0.99999 | 1.616e-02 | 7.426e-03 | 0.99987 +/- 2.2e-04 |

## Learning curve on D2 (held-out RMSE vs. training size)

| n | RBF | KRG |
|---|---|---|
| 8 | 1.380e+00 | 2.388e-01 |
| 16 | 6.797e-01 | 1.154e-01 |
| 32 | 1.266e-01 | 2.231e-02 |
| 64 | 8.794e-02 | 1.158e-02 |
| 128 | 9.443e-02 | 1.091e-02 |
| 256 | 3.258e-02 | 2.824e-03 |
| 512 | 2.344e-02 | 1.712e-03 |

- QP fits the quadratic D1 to machine precision but keeps a small bias on D2's `sqrt` term.
- RBF needs its inputs normalized (unscaled it fails); scaled, RBF and KRG interpolate
  both disciplines well.
- Test RMSE grows as the training set shrinks (learning curve) - the quantitative face of
  "not enough data."