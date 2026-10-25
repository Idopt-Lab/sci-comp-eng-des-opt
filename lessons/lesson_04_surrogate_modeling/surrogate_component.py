"""A drop-in Sellar discipline backed by a trained SMT surrogate.

The payoff of Lesson 4 is that a surrogate can stand in for an analysis *without
changing anything around it*. ``SmtSurrogateComp`` has the exact same
input/output interface as the analytic ``SellarDis1`` / ``SellarDis2`` components
from Lessons 2-3, so it slots straight into the same ``cycle`` group, the same
``NonlinearBlockGS`` + ``DirectSolver``, and the same SLSQP driver (see
``03_surrogate_mda_opt.py``). The only difference is that ``compute`` calls the
surrogate's ``predict_values`` instead of evaluating the Sellar equation.

The surrogates were trained on inputs scaled to the unit box (see
``02_fit_surrogates.py``), so this component applies the *same* normalization -
using the discipline's ``xlimits`` - before calling ``predict_values``.

Partials come from the surrogate's own analytic derivatives
(``predict_derivatives(x, kx)``), **not** finite differences. This is the Lesson-3
derivative story resurfacing: finite-differencing a Kriging model gives the
optimizer noisy gradients (its mean has tiny ripples between training points) and
SLSQP then stalls at the wrong point. A Gaussian process is smooth and hands back
an exact Jacobian for free, so we use it; the chain rule just divides by the input
span introduced by normalization.
"""

from __future__ import annotations

import numpy as np
import openmdao.api as om

# Column order each discipline's surrogate was trained on (see sellar_functions).
_D1_COLS = ("z1", "z2", "x", "y2")  # -> y1
_D2_COLS = ("z1", "z2", "y1")       # -> y2


class SmtSurrogateComp(om.ExplicitComponent):
    """Sellar discipline whose response comes from a trained SMT surrogate.

    Options
    -------
    model : smt.surrogate_models.*
        A *trained* SMT surrogate (``set_training_values`` + ``train`` already run).
    kind : {"d1", "d2"}
        Which discipline this stands in for. ``d1`` reads ``z, x, y2`` and writes
        ``y1``; ``d2`` reads ``z, y1`` and writes ``y2``.
    xlimits : (n_inputs, 2) array
        Input bounds the surrogate was trained on (same column order as above),
        used to scale inputs to ``[0, 1]`` before prediction.
    counter : CallCounter, optional
        Shared tally of discipline evaluations (same object used in Lesson 3), so
        we can compare surrogate-MDA call counts against the analytic run.
    """

    def initialize(self):
        self.options.declare("model", recordable=False)
        self.options.declare("kind", values=("d1", "d2"))
        self.options.declare("xlimits")
        self.options.declare("counter", default=None, recordable=False)

    def setup(self):
        self.add_input("z", val=np.zeros(2))  # [z1, z2] shared design variables
        if self.options["kind"] == "d1":
            self.add_input("x", val=0.0)
            self.add_input("y2", val=1.0)     # coupling IN (from D2)
            self.add_output("y1", val=1.0)    # coupling OUT (to D2)
        else:
            self.add_input("y1", val=1.0)     # coupling IN (from D1)
            self.add_output("y2", val=1.0)    # coupling OUT (to D1)

    def setup_partials(self):
        if self.options["kind"] == "d1":
            self.declare_partials("y1", ["z", "x", "y2"])
        else:
            self.declare_partials("y2", ["z", "y1"])

    def _span(self) -> np.ndarray:
        xl = self.options["xlimits"]
        return xl[:, 1] - xl[:, 0]

    def _scaled_row(self, raw: list[float]) -> np.ndarray:
        lo, hi = self.options["xlimits"][:, 0], self.options["xlimits"][:, 1]
        return ((np.array(raw) - lo) / (hi - lo)).reshape(1, -1)

    def compute(self, inputs, outputs):
        counter = self.options["counter"]
        model = self.options["model"]
        z1, z2 = inputs["z"]
        if self.options["kind"] == "d1":
            if counter is not None:
                counter.tick("Dis1", "compute")
            x_row = self._scaled_row([z1, z2, inputs["x"][0], inputs["y2"][0]])
            outputs["y1"] = model.predict_values(x_row)[0, 0]
        else:
            if counter is not None:
                counter.tick("Dis2", "compute")
            x_row = self._scaled_row([z1, z2, inputs["y1"][0]])
            outputs["y2"] = model.predict_values(x_row)[0, 0]

    def compute_partials(self, inputs, partials):
        counter = self.options["counter"]
        model = self.options["model"]
        span = self._span()
        z1, z2 = inputs["z"]
        if self.options["kind"] == "d1":
            if counter is not None:
                counter.tick("Dis1", "partials")
            xn = self._scaled_row([z1, z2, inputs["x"][0], inputs["y2"][0]])
            # d y1 / d (scaled input k), then chain-rule back to the raw input.
            d = np.array([model.predict_derivatives(xn, k)[0, 0] for k in range(4)]) / span
            partials["y1", "z"] = d[:2].reshape(1, 2)
            partials["y1", "x"] = d[2]
            partials["y1", "y2"] = d[3]
        else:
            if counter is not None:
                counter.tick("Dis2", "partials")
            xn = self._scaled_row([z1, z2, inputs["y1"][0]])
            d = np.array([model.predict_derivatives(xn, k)[0, 0] for k in range(3)]) / span
            partials["y2", "z"] = d[:2].reshape(1, 2)
            partials["y2", "y1"] = d[2]
