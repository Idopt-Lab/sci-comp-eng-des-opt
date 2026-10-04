"""Lesson 3: render the XDSM diagrams for gradient-based optimization.

Produces two diagrams in ``outputs/``:

* ``sellar_mdf_opt`` -- the Sellar MDF architecture: an optimizer wrapped around an
  MDA solver that drives the two coupled disciplines, feeding objective/constraints
  back to the optimizer;
* ``asw_opt`` -- the ASW sizing loop recast as an optimization: the optimizer takes
  the place of the solver, choosing ``W_TO`` to close the sizing residual.

Rendering needs ``pdflatex`` (TeX Live/MiKTeX) and, for the PNG, ``pdftoppm``
(poppler) on PATH; if they are missing the script prints a message and exits
non-zero, like ``lessons/lesson_01_dsm/generate_shapes.py``.

Run from the repository root (with the ``eng-des-opt-course`` environment active)::

    python lessons/lesson_03_gradient_based_optimization/generate_xdsm.py
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
REPO_ROOT = Path(__file__).resolve().parents[2]


class MissingPrerequisiteError(RuntimeError):
    """Raised when a rendering prerequisite is unavailable."""


def _pyxdsm_api():
    try:
        from pyxdsm.XDSM import FUNC, LEFT, OPT, RIGHT, SOLVER, XDSM
    except ModuleNotFoundError as exc:  # pragma: no cover - environment guard
        raise MissingPrerequisiteError(
            "Missing prerequisite: Python package 'pyXDSM' is not installed."
        ) from exc
    return {"XDSM": XDSM, "OPT": OPT, "SOLVER": SOLVER, "FUNC": FUNC, "LEFT": LEFT, "RIGHT": RIGHT}


def build_sellar_mdf(api):
    """Sellar MDF: OPT -> MDA solver -> d1/d2 -> objective/constraints -> OPT."""
    XDSM, OPT, SOLVER, FUNC, RIGHT = (
        api["XDSM"], api["OPT"], api["SOLVER"], api["FUNC"], api["RIGHT"]
    )
    x = XDSM(use_sfmath=False)
    x.add_system("opt", OPT, (r"\text{Optimizer}",))
    x.add_system("solver", SOLVER, (r"\text{MDA}", r"\text{(NLBGS)}"))
    x.add_system("d1", FUNC, (r"\text{Discipline 1}",))
    x.add_system("d2", FUNC, (r"\text{Discipline 2}",))
    x.add_system("f", FUNC, (r"\text{Objective}", r"\text{\& constraints}"))

    x.add_input("opt", r"z^{(0)},\ x^{(0)}")
    x.connect("opt", "d1", r"z,\ x")
    x.connect("opt", "d2", r"z")
    x.connect("solver", "d1", r"y_2")
    x.connect("d1", "d2", r"y_1")
    x.connect("d2", "solver", r"y_2")
    x.connect("d1", "f", r"y_1")
    x.connect("d2", "f", r"y_2")
    x.connect("opt", "f", r"z,\ x")
    x.connect("f", "opt", r"f,\ g_1,\ g_2")
    x.add_output("opt", r"z^{*},\ x^{*}", side=RIGHT)

    x.add_process(["opt", "solver", "d1", "d2", "solver", "f", "opt"], arrow=True)
    return x


def build_asw_opt(api):
    """ASW as optimization: the optimizer replaces the solver and closes R(W_TO)=0."""
    XDSM, OPT, FUNC, RIGHT = api["XDSM"], api["OPT"], api["FUNC"], api["RIGHT"]
    x = XDSM(use_sfmath=False)
    x.add_system("opt", OPT, (r"\text{Optimizer}",))
    x.add_system("disc", FUNC, (r"\text{Aero / prop /}", r"\text{mission / structures}"))
    x.add_system("res", FUNC, (r"\text{Sizing residual}", r"R(W_{TO})"))

    x.add_input("opt", r"W_{TO}^{(0)}")
    x.connect("opt", "disc", r"W_{TO}")
    x.connect("disc", "res", r"W_f/W_{TO},\ W_e/W_{TO}")
    x.connect("res", "opt", r"f,\ R")
    x.add_output("opt", r"W_{TO}^{*}", side=RIGHT)

    x.add_process(["opt", "disc", "res", "opt"], arrow=True)
    return x


DIAGRAMS = [("sellar_mdf_opt", build_sellar_mdf), ("asw_opt", build_asw_opt)]


def _require_executable(name: str, purpose: str) -> str:
    executable = shutil.which(name)
    if executable is None:
        raise MissingPrerequisiteError(
            f"Missing prerequisite: executable '{name}' is required for {purpose}."
        )
    return executable


def _cleanup(stem: Path) -> None:
    for suffix in (".aux", ".fdb_latexmk", ".fls", ".log", ".tex", ".tikz"):
        artifact = stem.with_suffix(suffix)
        if artifact.exists():
            artifact.unlink()


def _render_one(basename: str, builder) -> list[Path]:
    xdsm = builder(_pyxdsm_api())
    xdsm.write(basename, build=False, cleanup=True, outdir=str(OUTPUT_DIR))

    tex_path = OUTPUT_DIR / f"{basename}.tex"
    _require_executable("pdflatex", "pyXDSM PDF rendering")
    subprocess.run(
        ["pdflatex", "-halt-on-error", "-interaction=nonstopmode", tex_path.name],
        cwd=tex_path.parent, check=True,
    )

    written = [OUTPUT_DIR / f"{basename}.pdf"]
    pdftoppm = shutil.which("pdftoppm")
    if pdftoppm is not None:
        png_prefix = OUTPUT_DIR / basename
        subprocess.run(
            [pdftoppm, "-png", "-r", "200", "-singlefile", str(written[0]), str(png_prefix)],
            check=True,
        )
        written.append(png_prefix.with_suffix(".png"))

    _cleanup(tex_path.with_suffix(""))
    return written


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    try:
        for basename, builder in DIAGRAMS:
            for path in _render_one(basename, builder):
                print(f"Wrote {path.relative_to(REPO_ROOT)}")
    except MissingPrerequisiteError as exc:
        print(exc, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
