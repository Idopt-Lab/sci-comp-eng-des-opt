"""Smoke tests for the Lesson 3 scripts.

Each teaching script self-asserts its key numbers, so executing it and checking the
exit code guards the whole lesson.  We additionally re-run the core Sellar optimize
inline and assert the optimum, independent of the scripts.
"""

from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np
import openmdao.api as om

LESSON_DIR = Path(__file__).resolve().parents[1]
SCRIPTS = [
    LESSON_DIR / "sellar" / "01_sellar_optimization.py",
    LESSON_DIR / "sellar" / "02_sellar_derivatives.py",
    LESSON_DIR / "autodiff" / "03_computational_graph.py",
    LESSON_DIR / "asw" / "04_asw_as_optimization.py",
    LESSON_DIR / "c172" / "05_c172_trim.py",
]


class ScriptsRunCleanTests(unittest.TestCase):
    def test_scripts_exit_zero(self) -> None:
        # Force UTF-8 end-to-end: some scripts print Unicode math, which a legacy
        # Windows codepage would otherwise fail to decode from the captured pipe.
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


class SellarOptimumInlineTests(unittest.TestCase):
    """An independent Sellar optimize, so the acceptance number is tested directly."""

    def test_optimum(self) -> None:
        class Dis1(om.ExplicitComponent):
            def setup(self):
                self.add_input("z", val=np.zeros(2))
                self.add_input("x", val=0.0)
                self.add_input("y2", val=1.0)
                self.add_output("y1", val=1.0)
                self.declare_partials("*", "*", method="fd")

            def compute(self, inputs, outputs):
                z1, z2 = inputs["z"]
                outputs["y1"] = z1**2 + z2 + inputs["x"] - 0.2 * inputs["y2"]

        class Dis2(om.ExplicitComponent):
            def setup(self):
                self.add_input("z", val=np.zeros(2))
                self.add_input("y1", val=1.0)
                self.add_output("y2", val=1.0)
                self.declare_partials("*", "*", method="fd")

            def compute(self, inputs, outputs):
                z1, z2 = inputs["z"]
                y1 = inputs["y1"]
                if y1.real < 0.0:
                    y1 *= -1
                outputs["y2"] = y1**0.5 + z1 + z2

        prob = om.Problem(reports=False)
        cycle = prob.model.add_subsystem("cycle", om.Group(), promotes=["*"])
        cycle.add_subsystem("d1", Dis1(), promotes_inputs=["x", "z", "y2"], promotes_outputs=["y1"])
        cycle.add_subsystem("d2", Dis2(), promotes_inputs=["z", "y1"], promotes_outputs=["y2"])
        cycle.nonlinear_solver = om.NonlinearBlockGS()
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
        self.assertAlmostEqual(float(prob.get_val("obj")[0]), 3.183394, places=4)


if __name__ == "__main__":
    unittest.main()
