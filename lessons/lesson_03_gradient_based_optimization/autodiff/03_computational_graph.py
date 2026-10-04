r"""Lesson 3 - Step 3: automatic differentiation from scratch, then in JAX.

The script is in two clearly separated parts:

  PART A - the textbook example (Griewank & Walther 2008; Baydin et al. 2018)
      f(x1, x2) = ln(x1) + x1 * x2 - sin(x2),   at (x1, x2) = (2, 5)
    We build the computational graph with a ~40-line ``Var`` tape, differentiate it
    by hand in forward (dual numbers) and reverse (adjoint) mode, confirm JAX agrees,
    and *visualize* the same expression three ways:
      - our own networkx graph of the tape,
      - CasADi's ``export_graph`` (its recent expression-graph exporter), and
      - the JAX ``jaxpr`` drawn as a primitive graph.

  PART B - a Sellar discipline
      y1 = z1^2 + z2 + x - 0.2*y2
    We differentiate it with the same ``Var`` tape and tabulate tape-vs-analytic.

  Then a short demonstration of *why* reverse mode suits many-input problems and
  forward mode suits many-output problems, by counting sweeps on two tiny problems.

Run from the repository root (with the ``eng-des-opt-course`` environment active)::

    python lessons/lesson_03_gradient_based_optimization/autodiff/03_computational_graph.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

# The teaching output uses Unicode math (superscripts, minus sign, arrows); make sure
# it prints on a legacy-codepage Windows console too.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):  # pragma: no cover
    pass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
REPO_ROOT = Path(__file__).resolve().parents[3]

_INPUT_COLOR = "#34658b"  # blue  (independent variables)
_OP_COLOR = "#b00000"     # red   (operations)


# --------------------------------------------------------------------------- #
# Pretty-printing helpers
# --------------------------------------------------------------------------- #
def banner(title: str) -> None:
    rule = "=" * 76
    print(f"\n{rule}\n  {title}\n{rule}")


def print_table(headers, rows, right=()) -> None:
    """Print an aligned text table. ``right`` is a set of column indices to right-justify."""
    cells = [[str(c) for c in row] for row in rows]
    widths = [len(str(h)) for h in headers]
    for row in cells:
        for i, c in enumerate(row):
            widths[i] = max(widths[i], len(c))

    def line(values):
        out = []
        for i, v in enumerate(values):
            out.append(v.rjust(widths[i]) if i in right else v.ljust(widths[i]))
        return "  " + "   ".join(out)

    print(line([str(h) for h in headers]))
    print("  " + "   ".join("-" * w for w in widths))
    for row in cells:
        print(line(row))


# --------------------------------------------------------------------------- #
# A ~40-line reverse-mode autodiff "tape": each node records its value, its parents,
# and the local partial dnode/dparent for each parent.
# --------------------------------------------------------------------------- #
class Var:
    _counter = 0

    def __init__(self, value, parents=(), label=None):
        self.value = float(value)
        self.parents = list(parents)  # list of (parent_Var, local_partial)
        self.grad = 0.0
        self.label = label or f"v{Var._counter}"
        Var._counter += 1

    @staticmethod
    def _coerce(other):
        return other if isinstance(other, Var) else Var(other, label=f"c({other})")

    def __add__(self, other):
        o = self._coerce(other)
        return Var(self.value + o.value, [(self, 1.0), (o, 1.0)])

    def __sub__(self, other):
        o = self._coerce(other)
        return Var(self.value - o.value, [(self, 1.0), (o, -1.0)])

    def __mul__(self, other):
        o = self._coerce(other)
        return Var(self.value * o.value, [(self, o.value), (o, self.value)])

    __radd__ = __add__
    __rmul__ = __mul__

    def __rsub__(self, other):
        return self._coerce(other).__sub__(self)


def vlog(v: Var) -> Var:
    v = Var._coerce(v)
    return Var(math.log(v.value), [(v, 1.0 / v.value)])


def vsin(v: Var) -> Var:
    v = Var._coerce(v)
    return Var(math.sin(v.value), [(v, math.cos(v.value))])


def topological_order(output: Var) -> list[Var]:
    order, seen = [], set()

    def visit(node: Var) -> None:
        if id(node) in seen:
            return
        seen.add(id(node))
        for parent, _ in node.parents:
            visit(parent)
        order.append(node)

    visit(output)
    return order


def reverse_mode(output: Var) -> None:
    """Seed the output adjoint to 1 and accumulate adjoints backward (one sweep)."""
    order = topological_order(output)
    for node in order:
        node.grad = 0.0
    output.grad = 1.0
    for node in reversed(order):
        for parent, local in node.parents:
            parent.grad += node.grad * local


# --------------------------------------------------------------------------- #
# Forward mode via dual numbers: carry (value, derivative) through every op.
# --------------------------------------------------------------------------- #
class Dual:
    def __init__(self, value, deriv=0.0):
        self.value = float(value)
        self.deriv = float(deriv)

    @staticmethod
    def _coerce(other):
        return other if isinstance(other, Dual) else Dual(other, 0.0)

    def __add__(self, other):
        o = self._coerce(other)
        return Dual(self.value + o.value, self.deriv + o.deriv)

    def __sub__(self, other):
        o = self._coerce(other)
        return Dual(self.value - o.value, self.deriv - o.deriv)

    def __mul__(self, other):
        o = self._coerce(other)
        return Dual(self.value * o.value, self.deriv * o.value + self.value * o.deriv)

    __radd__ = __add__
    __rmul__ = __mul__

    def __rsub__(self, other):
        return self._coerce(other).__sub__(self)


def dlog(d: Dual) -> Dual:
    d = Dual._coerce(d)
    return Dual(math.log(d.value), d.deriv / d.value)


def dsin(d: Dual) -> Dual:
    d = Dual._coerce(d)
    return Dual(math.sin(d.value), d.deriv * math.cos(d.value))


# --------------------------------------------------------------------------- #
# Shared drawing: lay a DAG out in topological layers and render it.
# --------------------------------------------------------------------------- #
def _layered_positions(g: nx.DiGraph) -> dict:
    try:
        generations = list(nx.topological_generations(g))
    except nx.NetworkXError:
        return nx.spring_layout(g, seed=1)
    pos = {}
    for depth, layer in enumerate(generations):
        layer = list(layer)
        for i, node in enumerate(layer):
            pos[node] = (depth, (len(layer) - 1) / 2.0 - i)
    return pos


def draw_dag(g: nx.DiGraph, outfile: Path, title: str, labels: dict, colors: dict) -> None:
    pos = _layered_positions(g)
    node_colors = [colors.get(n, _OP_COLOR) for n in g.nodes]
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    nx.draw_networkx_edges(g, pos, ax=ax, arrows=True, arrowsize=14,
                           edge_color="#34658b", width=1.4, node_size=2000)
    nx.draw_networkx_nodes(g, pos, ax=ax, node_color=node_colors, node_size=2000,
                           edgecolors="black")
    nx.draw_networkx_labels(g, pos, ax=ax, labels=labels, font_size=10,
                            font_color="white", font_weight="bold")
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(outfile, dpi=130)
    plt.close(fig)


# =========================================================================== #
# PART A - the textbook example
# =========================================================================== #
def evaluation_trace(x1: float, x2: float):
    vm1, v0 = x1, x2
    v1 = math.log(vm1)
    v2 = vm1 * v0
    v3 = math.sin(v0)
    v4 = v1 + v2
    v5 = v4 - v3
    return [
        ("v-1", "x1", vm1), ("v0", "x2", v0),
        ("v1", "ln(v-1)", v1), ("v2", "v-1 * v0", v2), ("v3", "sin(v0)", v3),
        ("v4", "v1 + v2", v4), ("v5", "v4 - v3  (= f)", v5),
    ]


def textbook_graph(outfile: Path) -> None:
    """Our own tape, drawn as a networkx DAG."""
    g = nx.DiGraph()
    labels, colors = {}, {}
    nodes = {
        "x1": ("x1", _INPUT_COLOR), "x2": ("x2", _INPUT_COLOR),
        "ln": ("ln", _OP_COLOR), "mul": ("×", _OP_COLOR), "sin": ("sin", _OP_COLOR),
        "add": ("+", _OP_COLOR), "sub": ("−  (f)", _OP_COLOR),
    }
    for n, (lab, col) in nodes.items():
        g.add_node(n)
        labels[n], colors[n] = lab, col
    g.add_edges_from([("x1", "ln"), ("x1", "mul"), ("x2", "mul"), ("x2", "sin"),
                      ("ln", "add"), ("mul", "add"), ("add", "sub"), ("sin", "sub")])
    draw_dag(g, outfile, "Computational graph (our Var tape):  f = ln(x1) + x1·x2 − sin(x2)",
             labels, colors)


def casadi_graph(outfile_png: Path, outfile_html: Path) -> bool:
    """Use CasADi's ``export_graph`` to export the same expression, then render it.

    ``export_graph`` (a recent CasADi feature) writes GraphViz ``.dot`` and an
    interactive ``.html``/``.casadi_viz``.  We parse the DOT with pydot (pure Python,
    no GraphViz binary needed) and draw it with matplotlib.
    """
    try:
        import casadi as ca
        from networkx.drawing.nx_pydot import read_dot
    except ImportError:
        return False

    x1, x2 = ca.SX.sym("x1"), ca.SX.sym("x2")
    expr = ca.log(x1) + x1 * x2 - ca.sin(x2)
    dot_path = outfile_png.with_suffix(".dot")
    ca.export_graph(expr, str(dot_path))
    try:  # also emit CasADi's interactive HTML view
        ca.export_graph(expr, str(outfile_html))
    except Exception:
        pass

    raw = read_dot(str(dot_path))
    g = nx.DiGraph()
    labels, colors = {}, {}
    for name, data in raw.nodes(data=True):
        if ":" in name or data.get("label") is None:  # skip GraphViz port pseudo-nodes
            continue
        labels[name] = data["label"].strip('"')
        colors[name] = (data.get("fillcolor") or _OP_COLOR).strip('"') or _OP_COLOR
        g.add_node(name)
    for src, dst in raw.edges():
        base_src, base_dst = src.split(":")[0], dst.split(":")[0]  # strip :w / :e ports
        if base_src in labels and base_dst in labels and base_src != base_dst:
            g.add_edge(base_src, base_dst)
    draw_dag(g, outfile_png, "CasADi expression graph  (ca.export_graph)", labels, colors)
    dot_path.unlink(missing_ok=True)
    return True


def jaxpr_graph(outfile: Path) -> None:
    """Draw the JAX jaxpr as a primitive graph (a visual trace of the IR)."""
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)
    jaxpr = jax.make_jaxpr(lambda a, b: jnp.log(a) + a * b - jnp.sin(b))(2.0, 5.0)
    inner = jaxpr.jaxpr

    g = nx.DiGraph()
    labels, colors = {}, {}
    producer: dict[str, str] = {}
    for i, var in enumerate(inner.invars):
        name = f"in{i}"
        g.add_node(name)
        labels[name] = ["x1", "x2"][i] if i < 2 else str(var)
        colors[name] = _INPUT_COLOR
        producer[str(var)] = name
    for k, eqn in enumerate(inner.eqns):
        op = f"op{k}"
        g.add_node(op)
        labels[op] = eqn.primitive.name
        colors[op] = _OP_COLOR
        for invar in eqn.invars:
            src = producer.get(str(invar))
            if src is not None:
                g.add_edge(src, op)
        for outvar in eqn.outvars:
            producer[str(outvar)] = op
    draw_dag(g, outfile, "JAX jaxpr primitive graph  (jax.make_jaxpr)", labels, colors)


def part_a_textbook() -> dict:
    banner("PART A - textbook example:  f(x1, x2) = ln(x1) + x1·x2 − sin(x2)  at (2, 5)")
    x1, x2 = 2.0, 5.0
    exact = (1.0 / x1 + x2, x1 - math.cos(x2))

    print("\nForward evaluation trace:")
    print_table(["node", "operation", "value"],
                [(n, op, f"{v:.6f}") for n, op, v in evaluation_trace(x1, x2)], right={2})

    # Reverse mode: one backward sweep -> the whole gradient.
    vx1, vx2 = Var(x1, label="x1"), Var(x2, label="x2")
    f = (vlog(vx1) + vx1 * vx2) - vsin(vx2)
    reverse_mode(f)
    rev = (vx1.grad, vx2.grad)

    print(f"\nf(2, 5) = {f.value:.6f}")
    print("\nReverse-mode adjoint sweep (seed f-bar = 1, accumulate backward):")
    adj_rows = [(node.label, f"{node.value:.6f}", f"{node.grad:+.6f}")
                for node in reversed(topological_order(f))
                if node.parents or node.label in ("x1", "x2")]
    print_table(["node", "value", "adjoint (grad)"], adj_rows, right={1, 2})

    # Forward mode: one sweep per input.
    def f_dual(a: Dual, b: Dual) -> Dual:
        return (dlog(a) + a * b) - dsin(b)

    fwd = (f_dual(Dual(x1, 1.0), Dual(x2, 0.0)).deriv,
           f_dual(Dual(x1, 0.0), Dual(x2, 1.0)).deriv)

    # JAX.
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)
    fj = lambda a, b: jnp.log(a) + a * b - jnp.sin(b)
    jax_grad = jax.grad(fj, argnums=(0, 1))(x1, x2)
    _, jvp_x1 = jax.jvp(lambda a: fj(a, x2), (x1,), (1.0,))
    _, vjp_fn = jax.vjp(lambda a, b: fj(a, b), x1, x2)
    vjp_val = vjp_fn(1.0)

    print("\nGradient — every method agrees with the analytic answer:")
    print_table(
        ["component", "forward (dual)", "reverse (adjoint)", "jax.grad", "analytic"],
        [("df/dx1", f"{fwd[0]:.6f}", f"{rev[0]:.6f}", f"{float(jax_grad[0]):.6f}", f"{exact[0]:.6f}"),
         ("df/dx2", f"{fwd[1]:.6f}", f"{rev[1]:.6f}", f"{float(jax_grad[1]):.6f}", f"{exact[1]:.6f}")],
        right={1, 2, 3, 4},
    )
    print(f"\n  forward mode took 2 sweeps (one per input); reverse mode took 1 sweep.")
    print(f"  jax.jvp (seed x1) = {float(jvp_x1):.6f} = df/dx1;  "
          f"jax.vjp (seed 1)  = ({float(vjp_val[0]):.6f}, {float(vjp_val[1]):.6f}) = full gradient.")

    print("\njaxpr (traced primitives):")
    print("  " + str(jax.make_jaxpr(lambda a, b: fj(a, b))(x1, x2)).replace("\n", "\n  "))

    # Visualizations (textbook only).
    textbook_graph(OUTPUT_DIR / "graph_textbook.png")
    have_casadi = casadi_graph(OUTPUT_DIR / "graph_casadi.png", OUTPUT_DIR / "graph_casadi.html")
    jaxpr_graph(OUTPUT_DIR / "graph_jaxpr.png")
    figures = ["graph_textbook.png", "graph_jaxpr.png"] + (["graph_casadi.png", "graph_casadi.html"] if have_casadi else [])
    print(f"\nWrote visualizations: {', '.join(figures)}")

    import numpy as np

    assert np.allclose(rev, exact) and np.allclose(fwd, exact)
    assert np.allclose([float(jax_grad[0]), float(jax_grad[1])], exact)
    assert np.allclose(float(jvp_x1), exact[0]) and np.allclose([float(v) for v in vjp_val], exact)
    return {"exact": exact, "reverse": rev, "forward": fwd}


# =========================================================================== #
# PART B - a Sellar discipline through the same tape
# =========================================================================== #
def part_b_sellar() -> None:
    banner("PART B - Sellar discipline:  y1 = z1² + z2 + x − 0.2·y2")
    z1v, z2v, xv, y2v = 1.5, 0.3, 0.7, 2.0
    z1, z2, x, y2 = (Var(z1v, label="z1"), Var(z2v, label="z2"),
                     Var(xv, label="x"), Var(y2v, label="y2"))
    y1 = ((z1 * z1 + z2) + x) - 0.2 * y2
    reverse_mode(y1)

    tape = {"dy1/dz1": z1.grad, "dy1/dz2": z2.grad, "dy1/dx": x.grad, "dy1/dy2": y2.grad}
    analytic = {"dy1/dz1": 2.0 * z1v, "dy1/dz2": 1.0, "dy1/dx": 1.0, "dy1/dy2": -0.2}

    print(f"\ny1 = {y1.value:.4f}   (at z1={z1v}, z2={z2v}, x={xv}, y2={y2v})")
    print("\nPartials: Var tape vs. analytic  (dy1/dz1 = 2·z1, dy1/dz2 = 1, dy1/dx = 1, dy1/dy2 = −0.2)")
    print_table(
        ["partial", "Var (reverse)", "analytic", "match"],
        [(k, f"{tape[k]:+.6f}", f"{analytic[k]:+.6f}",
          "ok" if abs(tape[k] - analytic[k]) < 1e-12 else "DIFF") for k in analytic],
        right={1, 2},
    )
    import numpy as np

    assert np.allclose([tape[k] for k in analytic], [analytic[k] for k in analytic])


# =========================================================================== #
# Why reverse for many inputs, forward for many outputs
# =========================================================================== #
def part_c_cost() -> None:
    banner("Cost asymmetry: forward ∝ #inputs, reverse ∝ #outputs")

    # Case 1 - a scalar of many inputs (a gradient): g: R^4 -> R.
    vals = [1.0, 2.0, 3.0, 4.0]
    n = len(vals)

    def g_chain(xs):
        acc = xs[0] * xs[1]
        for i in range(1, n - 1):
            acc = acc + xs[i] * xs[i + 1]
        return acc

    # Forward: one sweep per input direction -> n sweeps for the full gradient.
    fwd_grad, fwd_sweeps = [], 0
    for j in range(n):
        duals = [Dual(vals[i], 1.0 if i == j else 0.0) for i in range(n)]
        fwd_grad.append(g_chain(duals).deriv)
        fwd_sweeps += 1
    # Reverse: a single backward sweep gives the whole gradient.
    vs = [Var(v) for v in vals]
    reverse_mode(g_chain(vs))
    rev_grad, rev_sweeps = [v.grad for v in vs], 1

    # Case 2 - a vector of one input (a Jacobian column): h: R -> R^3, h = [x, x², sin x].
    xv = 1.3
    # Forward: a single sweep carries the derivative to all 3 outputs.
    d = Dual(xv, 1.0)
    fwd_jac, fwd2_sweeps = [d.deriv, (d * d).deriv, dsin(d).deriv], 1
    # Reverse: one backward sweep per output -> 3 sweeps.
    rev_jac, rev2_sweeps = [], 0
    for k in range(3):
        v = Var(xv)
        outs = [v, v * v, vsin(v)]
        reverse_mode(outs[k])
        rev_jac.append(v.grad)
        rev2_sweeps += 1

    print("\nSweeps needed to get ALL derivatives (each sweep ≈ one function pass):")
    print_table(
        ["problem", "#inputs", "#outputs", "forward sweeps", "reverse sweeps"],
        [("g: R⁴ → R   (gradient)", n, 1, fwd_sweeps, rev_sweeps),
         ("h: R → R³   (Jacobian)", 1, 3, fwd2_sweeps, rev2_sweeps)],
        right={1, 2, 3, 4},
    )
    print("\n  Forward mode seeds ONE input direction per sweep, so it needs one sweep per input")
    print("  (cost ∝ #inputs). Reverse mode seeds ONE output adjoint per sweep, so it needs one")
    print("  sweep per output (cost ∝ #outputs). A design objective is ONE scalar of MANY design")
    print("  variables (e.g. ex_05 trim: 3N inputs → 1 objective), so reverse mode — one sweep for")
    print("  the whole gradient — is the clear winner; forward mode wins the opposite shape.")

    import numpy as np

    assert np.allclose(fwd_grad, rev_grad) and np.allclose(fwd_jac, rev_jac)


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    part_a_textbook()
    part_b_sellar()
    part_c_cost()
    print(f"\nWrote outputs to {OUTPUT_DIR.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
