"""Lesson 4: render the XDSM diagrams for surrogate-based optimization.

Produces two diagrams in ``outputs/``:

* ``sellar_doe_sampling`` -- the *offline* data-gathering step: a DOE driver
  samples a Sellar discipline (stacked, once per sample) over its input box and
  collects the input/response pairs that become the surrogate's training data;
* ``sellar_surrogate_mdf`` -- the Lesson-3 Sellar MDF architecture with the two
  analysis boxes replaced by ``METAMODEL`` (surrogate) boxes inside the MDA loop,
  so the optimizer drives ``hat{y}_1`` / ``hat{y}_2`` instead of the true
  disciplines.

Rendering needs ``pdflatex`` (TeX Live/MiKTeX) and, for the PNG, ``pdftoppm``
(poppler) on PATH; if they are missing the script prints a message and exits
non-zero, like the other lessons' XDSM scripts.

Run from the repository root (with the ``eng-des-opt-course`` environment active)::

    python lessons/lesson_04_surrogate_modeling/generate_xdsm.py
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
        from pyxdsm.XDSM import DOE, FUNC, METAMODEL, OPT, RIGHT, SOLVER, XDSM
    except ModuleNotFoundError as exc:  # pragma: no cover - environment guard
        raise MissingPrerequisiteError(
            "Missing prerequisite: Python package 'pyXDSM' is not installed."
        ) from exc
    return {"XDSM": XDSM, "OPT": OPT, "SOLVER": SOLVER, "DOE": DOE,
            "FUNC": FUNC, "METAMODEL": METAMODEL, "RIGHT": RIGHT}


def build_doe_sampling(api):
    """DOE driver sampling one discipline to build its surrogate training data."""
    XDSM, DOE, FUNC, RIGHT = api["XDSM"], api["DOE"], api["FUNC"], api["RIGHT"]
    x = XDSM(use_sfmath=False)
    x.add_system("doe", DOE, (r"\text{DOE driver}", r"\text{(LHS)}"))
    x.add_system("disc", FUNC, (r"\text{Sellar discipline}",), stack=True)

    x.add_input("doe", r"x_L,\ x_U,\ N")
    x.connect("doe", "disc", r"x^{(i)}", stack=True)
    x.connect("disc", "doe", r"y^{(i)}", stack=True)
    x.add_output("doe", r"\{x^{(i)},\ y^{(i)}\}\ \text{(training data)}", side=RIGHT)

    x.add_process(["doe", "disc", "doe"], arrow=True)
    return x


def build_surrogate_mdf(api):
    """Sellar MDF with METAMODEL surrogates standing in for the disciplines."""
    XDSM, OPT, SOLVER, METAMODEL, FUNC, RIGHT = (
        api["XDSM"], api["OPT"], api["SOLVER"], api["METAMODEL"], api["FUNC"], api["RIGHT"]
    )
    x = XDSM(use_sfmath=False)
    x.add_system("opt", OPT, (r"\text{Optimizer}",))
    x.add_system("solver", SOLVER, (r"\text{MDA}", r"\text{(NLBGS)}"))
    x.add_system("d1", METAMODEL, (r"\text{Surrogate}", r"\hat{y}_1"))
    x.add_system("d2", METAMODEL, (r"\text{Surrogate}", r"\hat{y}_2"))
    x.add_system("f", FUNC, (r"\text{Objective}", r"\text{\& constraints}"))

    x.add_input("opt", r"z^{(0)},\ x^{(0)}")
    x.add_input("d1", r"\text{training data}")
    x.add_input("d2", r"\text{training data}")
    x.connect("opt", "d1", r"z,\ x")
    x.connect("opt", "d2", r"z")
    x.connect("solver", "d1", r"y_2")
    x.connect("d1", "d2", r"\hat{y}_1")
    x.connect("d2", "solver", r"\hat{y}_2")
    x.connect("d1", "f", r"\hat{y}_1")
    x.connect("d2", "f", r"\hat{y}_2")
    x.connect("opt", "f", r"z,\ x")
    x.connect("f", "opt", r"f,\ g_1,\ g_2")
    x.add_output("opt", r"z^{*},\ x^{*}", side=RIGHT)

    x.add_process(["opt", "solver", "d1", "d2", "solver", "f", "opt"], arrow=True)
    return x


DIAGRAMS = [("sellar_doe_sampling", build_doe_sampling),
            ("sellar_surrogate_mdf", build_surrogate_mdf)]


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
