"""Smoke + acceptance tests for the Lesson 5 uncertainty-quantification scripts.

Scripts 1-2 (distributions, Monte Carlo) need only numpy/scipy and always run.
Scripts 3-4 (polynomial chaos, robust optimization) need the optional ``uqpce``
package, so those tests are gated behind an import check. The XDSM script is allowed
to "fail cleanly" when ``pdflatex`` is unavailable.
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

try:
    import uqpce  # noqa: F401
    UQPCE_AVAILABLE = True
except Exception:
    UQPCE_AVAILABLE = False

MC_SCRIPTS = [LESSON_DIR / "sellar_uncertain.py",
              LESSON_DIR / "01_distributions.py",
              LESSON_DIR / "02_forward_uq_montecarlo.py"]
UQPCE_SCRIPTS = [LESSON_DIR / "03_pce_uqpce.py",
                 LESSON_DIR / "04_robust_optimization.py"]


def _run(script: Path):
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    return subprocess.run(
        [sys.executable, str(script)],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        env=env, timeout=900,
    )


class ScriptsRunCleanTests(unittest.TestCase):
    def test_mc_scripts_exit_zero(self) -> None:
        for script in MC_SCRIPTS:
            with self.subTest(script=script.name):
                result = _run(script)
                self.assertEqual(result.returncode, 0,
                                 msg=f"{script.name} failed:\n{result.stderr[-2000:]}")

    @unittest.skipUnless(UQPCE_AVAILABLE, "uqpce not installed")
    def test_uqpce_scripts_exit_zero(self) -> None:
        for script in UQPCE_SCRIPTS:
            with self.subTest(script=script.name):
                result = _run(script)
                self.assertEqual(result.returncode, 0,
                                 msg=f"{script.name} failed:\n{result.stderr[-2000:]}")

    def test_xdsm_exits_zero_or_reports_missing_toolchain(self) -> None:
        result = _run(LESSON_DIR / "generate_xdsm.py")
        if result.returncode != 0:
            self.assertIn("Missing prerequisite", result.stderr,
                          msg=f"unexpected XDSM failure:\n{result.stderr[-2000:]}")


class ForwardUQInlineTests(unittest.TestCase):
    """Independent checks of the headline numbers (no uqpce needed)."""

    def test_deterministic_optimum_matches_canonical_and_is_unreliable(self) -> None:
        from sellar_uncertain import deterministic_optimum, sample_uncertain, solve_mda

        det = deterministic_optimum()
        self.assertTrue(det["success"])
        # Mean-value problem is the canonical Sellar: f* = 3.183394.
        self.assertAlmostEqual(det["obj"], 3.183394, places=3)
        rng = np.random.default_rng(0)
        resp = solve_mda(det["z"], det["x"], **sample_uncertain(50_000, rng))
        viol1 = float((resp["con1"] > 0).mean())   # g1 (y1 >= 3.16) violated when con1 > 0
        self.assertGreater(viol1, 0.4)   # deterministic optimum ~50% infeasible
        self.assertLess(viol1, 0.6)


@unittest.skipUnless(UQPCE_AVAILABLE, "uqpce not installed")
class PceInlineTests(unittest.TestCase):
    def test_pce_matches_monte_carlo(self) -> None:
        import contextlib
        import io

        from sellar_uncertain import (
            build_pce, deterministic_optimum, ensure_run_matrix, sample_uncertain, solve_mda,
        )

        det = deterministic_optimum()
        Xt = np.loadtxt(ensure_run_matrix(), ndmin=2)
        train = solve_mda(det["z"], det["x"], Xt[:, 0], Xt[:, 1], Xt[:, 2], Xt[:, 3])
        pce = build_pce()
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            pce.fit(Xt, train["obj"])
        pce_mean = float(np.real(pce._mean[0]))

        rng = np.random.default_rng(0)
        mc_obj = solve_mda(det["z"], det["x"], **sample_uncertain(100_000, rng))["obj"]
        self.assertLess(abs(pce_mean - mc_obj.mean()) / abs(mc_obj.mean()), 0.02)


if __name__ == "__main__":
    unittest.main()
