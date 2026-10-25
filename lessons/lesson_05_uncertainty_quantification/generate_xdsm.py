"""Lesson 5: render the XDSM diagrams for uncertainty quantification.

Produces two diagrams in ``outputs/``:

* ``sellar_forward_uq`` -- forward propagation: a UQ sampler (Monte Carlo or the
  PCE collocation design) evaluates the stacked uncertain Sellar MDA across draws
  of ``a0..a3`` and a statistics box rolls the responses up into ``E[.]`` and
  ``sigma[.]``;
* ``sellar_robust_opt`` -- the robust architecture: an optimizer wrapped around a
  UQPCE group (which hides the MDA + polynomial chaos) whose feedback edge carries
  the *statistical* objective/constraints (``E[obj]``, ``CI_upper(g)``).

Rendering needs ``pdflatex`` (TeX Live/MiKTeX) and, for the PNG, ``pdftoppm``
(poppler) on PATH; if they are missing the script prints a message and exits
non-zero, like the other lessons' XDSM scripts.

Run from the repository root (with the ``eng-des-opt-course`` environment active)::

    python lessons/lesson_05_uncertainty_quantification/generate_xdsm.py
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
        from pyxdsm.XDSM import DOE, FUNC, GROUP, OPT, RIGHT, XDSM
    except ModuleNotFoundError as exc:  # pragma: no cover - environment guard
        raise MissingPrerequisiteError(
            "Missing prerequisite: Python package 'pyXDSM' is not installed."
        ) from exc
    return {"XDSM": XDSM, "OPT": OPT, "DOE": DOE, "FUNC": FUNC, "GROUP": GROUP, "RIGHT": RIGHT}


def build_forward_uq(api):
    """Forward UQ: sampler -> stacked uncertain MDA -> statistics."""
    XDSM, DOE, GROUP, FUNC, RIGHT = (
        api["XDSM"], api["DOE"], api["GROUP"], api["FUNC"], api["RIGHT"]
    )
    x = XDSM(use_sfmath=False)
    x.add_system("uq", DOE, (r"\text{UQ sampler}", r"\text{(MC / PCE)}"))
    x.add_system("mda", GROUP, (r"\text{Sellar MDA}",), stack=True)
    x.add_system("stats", FUNC, (r"\text{Statistics}",))

    x.add_input("uq", r"p(a_0),\ \ldots,\ p(a_3)")
    x.add_input("mda", r"z,\ x")
    x.connect("uq", "mda", r"a^{(i)}", stack=True)
    x.connect("mda", "stats", r"\mathrm{obj}^{(i)},\ g^{(i)}", stack=True)
    x.add_output("stats", r"\mathbb{E}[\cdot],\ \sigma[\cdot]", side=RIGHT)

    x.add_process(["uq", "mda", "stats"], arrow=True)
    return x


def build_robust_opt(api):
    """Robust optimization: OPT around a UQPCE group over statistical obj/constraints."""
    XDSM, OPT, GROUP, RIGHT = api["XDSM"], api["OPT"], api["GROUP"], api["RIGHT"]
    x = XDSM(use_sfmath=False)
    x.add_system("opt", OPT, (r"\text{Optimizer}",))
    x.add_system("uqpce", GROUP, (r"\text{UQPCE group}", r"\text{(MDA + PCE)}"))

    x.add_input("opt", r"z^{(0)},\ x^{(0)}")
    x.add_input("uqpce", r"p(a_0),\ \ldots,\ p(a_3)")
    x.connect("opt", "uqpce", r"z,\ x")
    x.connect("uqpce", "opt", r"\mathbb{E}[\mathrm{obj}],\ \mathrm{CI}_{\mathrm{up}}(g_1,\ g_2)")
    x.add_output("opt", r"z^{*},\ x^{*}", side=RIGHT)

    x.add_process(["opt", "uqpce", "opt"], arrow=True)
    return x


DIAGRAMS = [("sellar_forward_uq", build_forward_uq),
            ("sellar_robust_opt", build_robust_opt)]


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
