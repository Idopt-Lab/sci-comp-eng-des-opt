"""Cessna-172 six-degree-of-freedom trim as a gradient-based optimization.

The example trims a C172 at one or more Mach numbers by driving the six rigid-body
accelerations to zero, and solves the resulting nonlinear least-squares problem with
five interchangeable derivative backends (numpy, JAX, CasADi, CSDL-alpha, Warp) plus
a gradient-free pymoo GA.  It is the worked example for the "gradient-based
optimization & computing derivatives" and "gradient-free optimization" weeks.

Ported from ``darshansarojini/mdo-jax-examples`` (``jax/c172``) with two documented
corrections -- see ``docs/ex_05_c172_trim.md``.
"""
