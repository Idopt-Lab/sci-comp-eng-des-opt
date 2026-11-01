"""Smoke + acceptance tests for the Lesson 4 surrogate-modeling scripts.

Each teaching script self-asserts its key numbers, so executing it and checking the
exit code guards the whole lesson. The XDSM script is allowed to "fail cleanly"
when ``pdflatex`` is unavailable. We additionally rebuild a surrogate-driven Sellar
optimize inline, so the acceptance number (f* ~= 3.1834) is tested directly.
"""

from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np

LESSON_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
for p in (LESSON_DIR, REPO_ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

SCRIPTS = [
    LESSON_DIR / "sellar_functions.py",
    LESSON_DIR / "01_doe_sampling.py",
    LESSON_DIR / "02_fit_surrogates.py",
    LESSON_DIR / "03_surrogate_mda_opt.py",
]


class ScriptsRunCleanTests(unittest.TestCase):
    def test_scripts_exit_zero(self) -> None:
        env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
        for script in SCRIPTS:
            with self.subTest(script=script.name):
                result = subprocess.run(
                    [sys.executable, str(script)],
                    capture_output=True, text=True, encoding="utf-8", errors="replace",
                    env=env, timeout=600,
                )
                self.assertEqual(result.returncode, 0,
                                 msg=f"{script.name} failed:\n{result.stderr[-2000:]}")

    def test_xdsm_exits_zero_or_reports_missing_toolchain(self) -> None:
        env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
        result = subprocess.run(
            [sys.executable, str(LESSON_DIR / "generate_xdsm.py")],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            env=env, timeout=300,
        )
        if result.returncode != 0:
            self.assertIn("Missing prerequisite", result.stderr,
                          msg=f"unexpected XDSM failure:\n{result.stderr[-2000:]}")


class SurrogateInlineTests(unittest.TestCase):
    """Independent surrogate fit + optimize, so acceptance numbers are tested directly."""

    def _train(self, disc_key: str, n: int):
        from smt.sampling_methods import LHS
        from smt.surrogate_models import KRG

        from sellar_functions import DISCIPLINES

        disc = DISCIPLINES[disc_key]
        lo, hi = disc["xlimits"][:, 0], disc["xlimits"][:, 1]
        x_raw = LHS(xlimits=disc["xlimits"], criterion="ese", seed=1)(n)
        model = KRG(print_global=False)
        model.set_training_values((x_raw - lo) / (hi - lo), disc["func"](x_raw))
        model.train()
        return model

    def test_surrogate_fit_is_accurate(self) -> None:
        from smt.sampling_methods import LHS

        from sellar_functions import DISCIPLINES

        model = self._train("d1", 80)
        disc = DISCIPLINES["d1"]
        lo, hi = disc["xlimits"][:, 0], disc["xlimits"][:, 1]
        x_raw = LHS(xlimits=disc["xlimits"], criterion="ese", seed=7)(200)
        y_true = disc["func"](x_raw).ravel()
        y_pred = model.predict_values((x_raw - lo) / (hi - lo)).ravel()
        ss_res = float(np.sum((y_pred - y_true) ** 2))
        ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
        self.assertGreater(1.0 - ss_res / ss_tot, 0.99)

    def test_surrogate_optimum(self) -> None:
        import openmdao.api as om

        from sellar_functions import DISCIPLINES
        from surrogate_component import SmtSurrogateComp

        models = {"d1": self._train("d1", 80), "d2": self._train("d2", 50)}
        prob = om.Problem(reports=False)
        cycle = prob.model.add_subsystem("cycle", om.Group(), promotes=["*"])
        cycle.add_subsystem(
            "d1", SmtSurrogateComp(model=models["d1"], kind="d1", xlimits=DISCIPLINES["d1"]["xlimits"]),
            promotes_inputs=["x", "z", "y2"], promotes_outputs=["y1"])
        cycle.add_subsystem(
            "d2", SmtSurrogateComp(model=models["d2"], kind="d2", xlimits=DISCIPLINES["d2"]["xlimits"]),
            promotes_inputs=["z", "y1"], promotes_outputs=["y2"])
        cycle.nonlinear_solver = om.NonlinearBlockGS(maxiter=50, atol=1e-6, rtol=1e-6)
        cycle.linear_solver = om.DirectSolver()
        prob.model.add_subsystem("obj_cmp", om.ExecComp("obj = x**2 + z[1] + y1 + exp(-y2)",
                                                        z=np.zeros(2), x=0.0),
                                 promotes=["x", "z", "y1", "y2", "obj"])
        prob.model.add_subsystem("con1", om.ExecComp("c1 = 3.16 - y1"), promotes=["c1", "y1"])
        prob.model.add_subsystem("con2", om.ExecComp("c2 = y2 - 24.0"), promotes=["c2", "y2"])
        prob.model.set_input_defaults("x", 1.0)
        prob.model.set_input_defaults("z", np.array([5.0, 2.0]))
        prob.driver = om.ScipyOptimizeDriver(optimizer="SLSQP", tol=1e-9)
        prob.model.add_design_var("z", lower=np.array([-10.0, 0.0]), upper=np.array([10.0, 10.0]))
        prob.model.add_design_var("x", lower=0.0, upper=10.0)
        prob.model.add_objective("obj")
        prob.model.add_constraint("c1", upper=0.0)
        prob.model.add_constraint("c2", upper=0.0)
        prob.setup()
        prob.run_driver()
        self.assertAlmostEqual(float(prob.get_val("obj")[0]), 3.183394, places=2)


if __name__ == "__main__":
    unittest.main()
