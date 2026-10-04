"""Matplotlib figures for the C172 trim study.

Each function takes a pandas ``DataFrame`` (or a :class:`~..methods.solve.TrimSolution`)
and writes one PNG.  Columns expected by the study frames are documented on each
function.  Importing matplotlib with the non-interactive Agg backend keeps these
usable in headless study runs.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _source_label(row) -> str:
    return row["source"] if row["deriv"] in ("exact", "none") else f"{row['source']}:{row['deriv']}"


def plot_trim_schedule(solution, path: Path) -> Path:
    """Trim pitch, elevator, and RPM versus Mach for one multi-node solution."""
    mach = solution.mach
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
    axes[0].plot(mach, solution.theta_deg, "o-")
    axes[0].set_ylabel(r"$\theta$ (deg)")
    axes[1].plot(mach, solution.delta_e_deg, "s-", color="C1")
    axes[1].set_ylabel(r"$\delta_e$ (deg)")
    axes[2].plot(mach, solution.omega_rpm, "^-", color="C2")
    axes[2].set_ylabel(r"$\omega$ (RPM)")
    for ax in axes:
        ax.set_xlabel("Mach")
        ax.grid(True, alpha=0.3)
    fig.suptitle("C172 trim schedule vs. airspeed")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


# Distinct markers so that series which coincide exactly (e.g. every exact-gradient
# backend shares the same model-evaluation count) show as concentric rings instead of
# hiding behind whichever line is drawn last.
_MARKERS = ("o", "s", "^", "D", "v", "P", "X", "*")


def _plot_series(ax, x, y, label, index):
    """loglog plot with an open marker whose size shrinks with ``index`` (nested rings)."""
    ax.loglog(x, y, linestyle="-", linewidth=1.3, alpha=0.85,
              marker=_MARKERS[index % len(_MARKERS)],
              markersize=max(10 - 1.7 * index, 3.0), fillstyle="none",
              markeredgewidth=1.4, label=label)


def plot_scaling(df, path: Path, y: str = "solve_s", ylabel: str = "solve time (s)") -> Path:
    """``y`` versus number of design variables, one line per backend+deriv.

    Expects columns: ``n_dv, source, deriv, <y>``.  Exact-gradient backends share an
    identical ``n_model_evals`` curve (the optimizer path is the same however the
    derivatives are produced), so they are drawn with nested open markers to stay
    distinguishable where they overlap.
    """
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    groups = sorted(df.groupby(["source", "deriv"]), key=lambda kv: str(kv[0]))
    for i, ((source, deriv), group) in enumerate(groups):
        group = group.sort_values("n_dv")
        label = source if deriv in ("exact", "none") else f"{source}:{deriv}"
        _plot_series(ax, group["n_dv"], group[y], label, i)
    ax.set_xlabel("number of design variables (3N)")
    ax.set_ylabel(ylabel)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_gradient_per_call(df, path: Path) -> Path:
    """Per-call gradient time versus problem size.  Columns: ``n_dv, backend, per_call_ms``."""
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    groups = sorted(df.groupby("backend"), key=lambda kv: str(kv[0]))
    for i, (backend, group) in enumerate(groups):
        group = group.sort_values("n_dv")
        device = group["device"].iloc[0] if "device" in group else ""
        label = f"{backend} ({device})" if device else backend
        _plot_series(ax, group["n_dv"], group["per_call_ms"], label, i)
    ax.set_xlabel("number of design variables (3N)")
    ax.set_ylabel("per-call gradient time (ms)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path
