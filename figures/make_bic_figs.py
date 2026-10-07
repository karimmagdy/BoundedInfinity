"""Schematic figures for the BIC manuscript.

  fig_bic_architecture.pdf  system architecture and inner components
  fig_bic_operations.pdf    eviction/summarisation and miss reconstruction

Rendered with matplotlib (vector PDF, no LaTeX). Every figure is checked
before it is saved: each label must lie entirely inside or entirely outside
every box, must not touch an arrow, a tree node or another label, and must
stay inside the figure. A failed check raises, so a layout change that makes
text run over a border cannot be saved silently.

    python make_bic_figs.py            # writes PDF + PNG next to this file
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch
from matplotlib.transforms import Bbox

OUT = Path(__file__).resolve().parent
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})

INK = "#222222"
BLUE, BLUEFILL = "#2c5d8f", "#dbe7f3"
GREEN, GREENFILL = "#3a7d44", "#dcecdd"
AMBER, AMBERFILL = "#b5860a", "#f6eccf"
GREY, GREYFILL = "#666666", "#ededed"
EVICT = "#b03a3a"
PAD_PX = 3  # minimum clearance, in pixels at 100 dpi, between text and any edge


# ------------------------------------------------------------------ drawing
def box(ax, x, y, w, h, ec, fc, lw=1.4, r=0.02):
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                       ec=ec, fc=fc, lw=lw, zorder=2)
    ax.add_patch(p)
    return p


def text(ax, x, y, s, fs=8.2, color=INK, bold=False, italic=False, ha="center", va="center"):
    return ax.text(x, y, s, ha=ha, va=va, fontsize=fs, color=color, zorder=5,
                   fontweight="bold" if bold else "normal", style="italic" if italic else "normal")


def arrow(ax, p0, p1, color=INK, lw=1.5, ls="-", rad=0.0):
    a = FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=11, color=color, lw=lw,
                        ls=ls, connectionstyle=f"arc3,rad={rad}", zorder=4,
                        shrinkA=0, shrinkB=0)
    ax.add_patch(a)
    return a


def node(ax, x, y, r, cached, holds=False, label=None, fs=8):
    c = Circle((x, y), r, fc=GREENFILL if cached else "#ffffff", ec=GREEN if cached else EVICT,
               lw=1.5, ls="-" if cached else (0, (2, 1.5)), zorder=3)
    ax.add_patch(c)
    if holds:
        ax.add_patch(Circle((x, y), r, fc="none", ec=AMBER, lw=2.6, zorder=3.5))
    if label:
        text(ax, x, y, label, fs=fs)
    return c


# ------------------------------------------------------------------ layout check
def check_layout(fig, name):
    """Raise if any text overlaps a box edge, an arrow, a node, another text or the figure edge."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    pad = PAD_PX * fig.dpi / 100
    problems = []
    fig_bb = fig.bbox.padded(-pad)
    texts, boxes, arrows, circles = [], [], [], []
    for ax in fig.axes:
        texts += [t for t in ax.texts if t.get_text().strip()]
        for p in ax.patches:
            if isinstance(p, FancyBboxPatch):
                boxes.append(p.get_window_extent(r))
            elif isinstance(p, FancyArrowPatch):
                arrows.append(p)
            elif isinstance(p, Circle):
                circles.append(p.get_window_extent(r))
    for leg in fig.legends:
        texts += leg.get_texts()

    def label(t):
        return repr(t.get_text().replace("\n", " ")[:40])

    bbs = [(t, t.get_window_extent(r).padded(pad)) for t in texts]
    for t, tb in bbs:
        if not (fig_bb.contains(tb.x0, tb.y0) and fig_bb.contains(tb.x1, tb.y1)):
            problems.append(f"{label(t)} leaves the figure")
        for b in boxes:
            inside = b.contains(tb.x0, tb.y0) and b.contains(tb.x1, tb.y1)
            if tb.overlaps(b) and not inside:
                problems.append(f"{label(t)} crosses a box edge")
        for c in circles:
            # a node label sits inside its own node; anything else must clear the node
            own = c.contains(*tb.get_points().mean(axis=0)) and tb.width < c.width * 1.2
            if tb.overlaps(c) and not own:
                problems.append(f"{label(t)} overlaps a tree node")
        for a in arrows:
            path = a.get_path().transformed(a.get_transform())
            pts = path.interpolated(60).vertices
            if any(tb.contains(x, y) for x, y in pts):
                problems.append(f"{label(t)} is crossed by an arrow")
    for i, (t1, b1) in enumerate(bbs):
        for t2, b2 in bbs[i + 1:]:
            if Bbox.intersection(b1, b2) is not None and b1.overlaps(b2):
                problems.append(f"{label(t1)} overlaps {label(t2)}")
    if problems:
        raise RuntimeError(f"{name}: layout problems:\n  " + "\n  ".join(sorted(set(problems))))


def save(fig, name):
    check_layout(fig, name)
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", pad_inches=0.05)
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight", pad_inches=0.05, dpi=200,
                facecolor="white")
    plt.close(fig)


# ------------------------------------------------------------------ FIG: ARCHITECTURE
def architecture():
    W, H = 7.6, 4.9
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W); ax.set_ylim(0, H); ax.set_aspect("equal"); ax.axis("off")

    # ---- left: recursive agent swarm
    text(ax, 0.95, 4.62, "Recursive agent swarm", fs=9.5, bold=True)
    box(ax, 0.1, 0.95, 1.7, 3.45, GREY, "#fbfbfb", lw=1.2, r=0.12)
    pos = {"r": (0.95, 3.95), "a": (0.45, 3.25), "b": (0.95, 3.25), "c": (1.45, 3.25),
           "a1": (0.3, 2.5), "a2": (0.6, 2.5), "c1": (1.3, 2.5), "c2": (1.6, 2.5),
           "a1x": (0.3, 1.8)}
    for u, v in [("r", "a"), ("r", "b"), ("r", "c"), ("a", "a1"), ("a", "a2"),
                 ("c", "c1"), ("c", "c2"), ("a1", "a1x")]:
        ax.plot(*zip(pos[u], pos[v]), color=GREY, lw=1.0, zorder=2.5)  # above the panel fill
    cached = {"r", "a", "b", "c", "a1"}
    for k, (x, y) in pos.items():
        node(ax, x, y, 0.1, k in cached)
    node(ax, 0.42, 1.25, 0.07, True)
    text(ax, 0.55, 1.25, "cached", fs=7.4, color=GREY, ha="left")
    node(ax, 1.12, 1.25, 0.07, False)
    text(ax, 1.25, 1.25, "evicted", fs=7.4, color=GREY, ha="left")

    arrow(ax, (1.8, 2.75), (2.3, 2.75), lw=1.8)
    text(ax, 2.05, 2.95, "spawn", fs=7.8, italic=True)

    # ---- centre: the runtime
    box(ax, 2.3, 0.15, 3.3, 4.6, BLUE, "#f7fafd", lw=1.8, r=0.12)
    text(ax, 3.95, 4.47, "Bounded Infinity Cache (runtime)", fs=9.5, bold=True, color=BLUE)

    box(ax, 2.5, 3.6, 2.9, 0.62, AMBER, AMBERFILL)
    text(ax, 3.95, 4.04, "Deterministic addressing", fs=8.4)
    text(ax, 3.95, 3.77, "Cantor pairing of ancestry path  $\\mapsto$  slot", fs=7.9)

    text(ax, 3.95, 3.3, "Bounded cache: $M$ slots of agent states", fs=8.2)
    M, sw, x0 = 8, 0.34, 2.6
    for i, filled in enumerate([1, 1, 1, 0, 1, 1, 0, 1]):
        box(ax, x0 + i * sw, 2.62, sw * 0.82, 0.42, BLUE, BLUEFILL if filled else "#ffffff",
            lw=1.1, r=0.05)

    box(ax, 2.5, 1.75, 2.9, 0.62, GREY, GREYFILL)
    text(ax, 3.95, 2.19, "Agent registry: $\\mathcal{O}(N)$", fs=8.4)
    text(ax, 3.95, 1.92, "one record per agent (parent, depth, task, ...)", fs=7.7)

    box(ax, 2.5, 0.35, 2.9, 1.15, GREEN, GREENFILL, lw=1.5)
    text(ax, 3.95, 1.3, "Eviction manager (when full)", fs=8.6, bold=True, color=GREEN)
    text(ax, 3.95, 1.03, "1. select the deepest agents", fs=7.9)
    text(ax, 3.95, 0.8, "2. summarise each with $\\varphi$", fs=7.9)
    text(ax, 3.95, 0.57, "3. fold into nearest cached ancestor", fs=7.9)

    arrow(ax, (3.95, 3.6), (3.95, 3.42), color=AMBER, lw=1.3)
    arrow(ax, (2.45, 2.83), (2.45, 1.5), color=GREEN, lw=1.2, rad=0.25)

    # ---- right: queries
    arrow(ax, (5.6, 2.83), (6.0, 2.83), lw=1.8)
    text(ax, 5.8, 3.03, "query", fs=7.8, italic=True)
    box(ax, 6.0, 3.05, 1.5, 0.95, GREEN, GREENFILL)
    text(ax, 6.75, 3.7, "Cache hit", fs=8.4)
    text(ax, 6.75, 3.38, "$\\mathcal{O}(1)$, exact state", fs=7.9)
    box(ax, 6.0, 1.6, 1.5, 1.2, AMBER, AMBERFILL)
    text(ax, 6.75, 2.5, "Cache miss", fs=8.4)
    text(ax, 6.75, 2.2, "$\\mathcal{O}(\\log N)$ walk up", fs=7.9)
    text(ax, 6.75, 1.92, "the registry", fs=7.9)
    arrow(ax, (6.0, 2.83), (6.25, 3.05), color=GREEN, lw=1.2)
    arrow(ax, (6.0, 2.83), (6.25, 2.8), color=AMBER, lw=1.2)
    arrow(ax, (5.4, 2.0), (6.0, 2.0), color=GREY, lw=1.1, ls=(0, (3, 2)))

    save(fig, "fig_bic_architecture")


# ------------------------------------------------------------------ FIG: OPERATIONS
POS = {"r": (1.5, 2.55), "a": (0.75, 1.75), "b": (1.5, 1.75), "c": (2.25, 1.75),
       "a1": (0.45, 0.95), "a2": (1.05, 0.95), "c1": (1.95, 0.95), "c2": (2.55, 0.95)}
EDGES = [("r", "a"), ("r", "b"), ("r", "c"), ("a", "a1"), ("a", "a2"), ("c", "c1"), ("c", "c2")]
R = 0.17


def tree_panel(ax, title, cached, holds, caption, caption_color):
    ax.set_xlim(0, 3.0); ax.set_ylim(-0.05, 3.15); ax.set_aspect("equal"); ax.axis("off")
    text(ax, 1.5, 3.0, title, fs=9.5, bold=True)
    for u, v in EDGES:
        ax.plot(*zip(POS[u], POS[v]), color="#999999", lw=1.0, zorder=1)
    for k, (x, y) in POS.items():
        node(ax, x, y, R, k in cached, holds=k in holds, label=k)
    for i, line in enumerate(caption):
        text(ax, 1.5, 0.42 - i * 0.22, line, fs=7.9, color=caption_color)


def fold(ax, child, parent, color, rad):
    (x0, y0), (x1, y1) = POS[child], POS[parent]
    d = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
    ux, uy = (x1 - x0) / d, (y1 - y0) / d
    arrow(ax, (x0 + ux * R, y0 + uy * R), (x1 - ux * R, y1 - uy * R), color=color, lw=1.8, rad=rad)


def operations():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.4, 3.6))
    fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.12, wspace=0.08)

    # (a) a1 and a2 are evicted and folded into their nearest cached ancestor, a
    tree_panel(ax1, "(a) Eviction and summarisation", cached={"r", "a", "b", "c"},
               holds={"a"}, caption_color=AMBER,
               caption=["evicted children are folded into their",
                        "nearest cached ancestor ($\\varphi$)"])
    fold(ax1, "a1", "a", AMBER, rad=-0.35)
    fold(ax1, "a2", "a", AMBER, rad=0.35)

    # (b) a query for the evicted a1 walks up to a, which holds a1's summary
    tree_panel(ax2, "(b) Miss reconstruction", cached={"r", "a", "b", "c"},
               holds={"a"}, caption_color=BLUE,
               caption=["query for evicted a1: walk up to the nearest",
                        "cached ancestor (a) and return its summaries"])
    fold(ax2, "a1", "a", BLUE, rad=-0.35)
    x, y = POS["a1"]
    ax2.add_patch(Circle((x, y), R + 0.07, fc="none", ec=INK, lw=1.3, ls=(0, (1, 1)), zorder=6))
    text(ax2, x - 0.35, y + 0.33, "query", fs=7.8)

    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=GREENFILL, markeredgecolor=GREEN,
               markersize=10, label="cached"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="white", markeredgecolor=EVICT,
               markersize=10, label="evicted"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="none", markeredgecolor=AMBER,
               markersize=11, markeredgewidth=2.4, label="holds summaries"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=8,
               bbox_to_anchor=(0.5, 0.0))
    save(fig, "fig_bic_operations")


if __name__ == "__main__":
    architecture()
    operations()
    print("wrote fig_bic_architecture and fig_bic_operations (PDF + PNG), layout checks passed")
