"""Render the C172 trim XDSM: an optimizer driving the analysis (no solver).

Unlike the ASW sizing diagram (where a nonlinear solver closes a feedback loop), trim
is posed directly as an optimization: the optimizer owns the design variables
(theta, delta_e, omega), which feed the aerodynamics, propulsion, and gravity
disciplines; their forces and moments feed the 6-DOF equations of motion, whose
residual becomes the objective returned to the optimizer.  This mirrors the
"optimizer replaces the solver" idea from lesson 3.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

DIAGRAM_BASENAME = "c172_trim_xdsm"
SOURCE_DIR = Path(__file__).resolve().parent
EXAMPLE_DIR = SOURCE_DIR.parents[1]  # .../examples/ex_05_c172_trim


def _find_repo_root() -> Path:
    for candidate in (SOURCE_DIR, *SOURCE_DIR.parents):
        if (candidate / "pyproject.toml").exists():
            return candidate
    raise RuntimeError(f"Could not locate repository root from {SOURCE_DIR}.")


REPO_ROOT = _find_repo_root()
PNG_OUTPUT_DIR = EXAMPLE_DIR / "docs" / "assets" / "images"
PDF_OUTPUT_DIR = EXAMPLE_DIR / "docs" / "assets" / "reference"
SCRIPT_COMMAND = f"python {Path(__file__).resolve().relative_to(REPO_ROOT)}"


class MissingPrerequisiteError(RuntimeError):
    """Raised when rendering prerequisites are unavailable."""


def _pyxdsm_api():
    try:
        from pyxdsm.XDSM import FUNC, OPT, RIGHT, XDSM
    except ModuleNotFoundError as exc:
        raise MissingPrerequisiteError(
            "Missing prerequisite: Python package 'pyXDSM' is not installed in the active "
            f"environment. Install it, then rerun `{SCRIPT_COMMAND}` from the repository root."
        ) from exc

    return XDSM, OPT, FUNC, RIGHT


def _require_executable(name: str, purpose: str) -> str:
    executable = shutil.which(name)
    if executable is None:
        raise MissingPrerequisiteError(
            f"Missing prerequisite: executable '{name}' is required for {purpose}. "
            f"After installing it, rerun `{SCRIPT_COMMAND}`."
        )
    return executable


def build_xdsm():
    """The trim optimization XDSM: OPT -> {aero, prop, gravity} -> EOM -> objective."""
    XDSM, OPT, FUNC, RIGHT = _pyxdsm_api()

    xdsm = XDSM(use_sfmath=False)

    xdsm.add_system("opt", OPT, (r"\text{Optimizer}", r"\text{(SLSQP)}"))
    xdsm.add_system("aero", FUNC, (r"\text{Aerodynamics}",))
    xdsm.add_system("prop", FUNC, (r"\text{Propulsion}",))
    xdsm.add_system("gravity", FUNC, (r"\text{Gravity}",))
    xdsm.add_system("eom", FUNC, (r"\text{6-DOF EOM}",))
    xdsm.add_system("obj", FUNC, (r"\text{Objective}", r"\tfrac{1}{2}\|\tilde r\|^2"))

    xdsm.add_input("opt", r"\theta_0,\ \delta_{e,0},\ \omega_0")
    xdsm.add_input("aero", r"M,\ S,\ \bar c")
    xdsm.add_input("prop", r"M,\ R,\ \rho")
    xdsm.add_input("gravity", r"m,\ g")

    # Optimizer drives the three disciplines with the design variables.
    xdsm.connect("opt", "aero", r"\theta,\ \delta_e")
    xdsm.connect("opt", "prop", r"\omega")
    xdsm.connect("opt", "gravity", r"\theta")

    # Disciplines feed forces/moments into the equations of motion.
    xdsm.connect("aero", "eom", r"F_a,\ M_a")
    xdsm.connect("prop", "eom", r"F_p")
    xdsm.connect("gravity", "eom", r"F_i")

    # EOM -> residual -> objective -> back to the optimizer.
    xdsm.connect("eom", "obj", r"\dot u, \dot w, \dot q")
    xdsm.connect("obj", "opt", r"f,\ \nabla f")

    xdsm.add_output("opt", r"\theta^*,\ \delta_e^*,\ \omega^*", side=RIGHT)

    xdsm.add_process(["opt", "aero", "eom", "obj", "opt"], arrow=True)
    xdsm.add_process(["opt", "prop", "eom"], arrow=True)
    xdsm.add_process(["opt", "gravity", "eom"], arrow=True)

    return xdsm


def _cleanup_latex_artifacts(stem: Path) -> None:
    for suffix in (".aux", ".fdb_latexmk", ".fls", ".log"):
        artifact = stem.with_suffix(suffix)
        if artifact.exists():
            artifact.unlink()


def _copy_output(source_path: Path, destination_dir: Path) -> Path:
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination_path = destination_dir / source_path.name
    shutil.copy2(source_path, destination_path)
    return destination_path


def _render_pdf(tex_path: Path) -> Path:
    _require_executable("pdflatex", "pyXDSM PDF rendering")
    subprocess.run(
        ["pdflatex", "-halt-on-error", "-interaction=nonstopmode", tex_path.name],
        cwd=tex_path.parent,
        check=True,
    )
    pdf_path = _copy_output(tex_path.with_suffix(".pdf"), PDF_OUTPUT_DIR)
    source_pdf_path = tex_path.with_suffix(".pdf")
    if source_pdf_path.exists():
        source_pdf_path.unlink()
    _cleanup_latex_artifacts(tex_path.with_suffix(""))
    return pdf_path


def _render_png(pdf_path: Path) -> Path | None:
    pdftoppm = shutil.which("pdftoppm")
    if pdftoppm is None:
        return None
    PNG_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    png_prefix = PNG_OUTPUT_DIR / pdf_path.with_suffix("").name
    subprocess.run(
        [pdftoppm, "-png", "-singlefile", str(pdf_path), str(png_prefix)], check=True
    )
    return png_prefix.with_suffix(".png")


def render() -> list[Path]:
    """Render the diagram into the documentation asset paths."""
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    PDF_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PNG_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    xdsm = build_xdsm()
    xdsm.write(DIAGRAM_BASENAME, build=False, cleanup=True, outdir=str(SOURCE_DIR))

    tex_path = SOURCE_DIR / f"{DIAGRAM_BASENAME}.tex"
    tikz_path = SOURCE_DIR / f"{DIAGRAM_BASENAME}.tikz"
    pdf_path = _render_pdf(tex_path)
    png_path = _render_png(pdf_path)

    outputs = [tikz_path, tex_path, pdf_path]
    if png_path is not None:
        outputs.append(png_path)
    return outputs


def _format_relative(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def main() -> int:
    try:
        outputs = render()
    except MissingPrerequisiteError as exc:
        print(exc, file=sys.stderr)
        print(f"Render command: {SCRIPT_COMMAND}", file=sys.stderr)
        return 1

    print("Rendered:", ", ".join(_format_relative(path) for path in outputs))
    png_path = PNG_OUTPUT_DIR / f"{DIAGRAM_BASENAME}.png"
    if png_path.exists():
        print(f"Markdown embed: ![C172 trim XDSM]({_format_relative(png_path)})")
    else:
        pdf_path = PDF_OUTPUT_DIR / f"{DIAGRAM_BASENAME}.pdf"
        print(f"Markdown link: [C172 trim XDSM]({_format_relative(pdf_path)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
