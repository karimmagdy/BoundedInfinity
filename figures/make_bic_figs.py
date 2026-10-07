"""Schematic figures for the BIC manuscript:
  fig_bic_architecture.pdf  — system architecture + inner components
  fig_bic_operations.pdf    — eviction/summarisation + miss reconstruction
Rendered with matplotlib (vector PDF). No LaTeX needed.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
from matplotlib.lines import Line2D

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})

INK = "#222222"
BLUE = "#2c5d8f"; BLUEFILL = "#dbe7f3"
GREEN = "#3a7d44"; GREENFILL = "#dcecdd"
AMBER = "#b5860a"; AMBERFILL = "#f6eccf"
GREY = "#666666"; GREYFILL = "#ededed"
EVICT = "#b03a3a"; EVICTFILL = "#f3dede"


def box(ax, x, y, w, h, text, ec=BLUE, fc=BLUEFILL, fs=9, bold=False, round=0.02, lw=1.4, ha="center"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.004,rounding_size={round}",
                                ec=ec, fc=fc, lw=lw, zorder=2))
    ax.text(x + (w/2 if ha == "center" else 0.012), y + h/2, text, ha=ha, va="center",
            fontsize=fs, color=INK, zorder=3, fontweight="bold" if bold else "normal")


def arrow(ax, p0, p1, color=INK, lw=1.6, style="-|>", ls="-", rad=0.0):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=12, color=color,
                                 lw=lw, ls=ls, connectionstyle=f"arc3,rad={rad}", zorder=1))


# ============================================================ FIG 1: ARCHITECTURE
def architecture():
    fig, ax = plt.subplots(figsize=(7.2, 4.7)); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    # ---- Left: recursive agent swarm (mini tree) ----
    ax.text(0.115, 0.96, "Recursive agent swarm", ha="center", fontsize=9.5, fontweight="bold", color=INK)
    box(ax, 0.01, 0.27, 0.21, 0.63, "", ec=GREY, fc="#fbfbfb", round=0.03, lw=1.2)
    nodes = {"r": (0.115, 0.83), "a": (0.055, 0.66), "b": (0.115, 0.66), "c": (0.175, 0.66),
             "a1": (0.04, 0.49), "a2": (0.09, 0.49), "c1": (0.16, 0.49), "c2": (0.20, 0.49),
             "a1x": (0.055, 0.34)}
    edges = [("r","a"),("r","b"),("r","c"),("a","a1"),("a","a2"),("c","c1"),("c","c2"),("a1","a1x")]
    for u, v in edges:
        ax.plot([nodes[u][0], nodes[v][0]], [nodes[u][1], nodes[v][1]], color=GREY, lw=1.0, zorder=1)
    cached = {"r","a","b","c","a1"}
    for k,(x,y) in nodes.items():
        fc = GREENFILL if k in cached else "#ffffff"
        ec = GREEN if k in cached else EVICT
        ax.add_patch(Circle((x,y), 0.016, fc=fc, ec=ec, lw=1.3, zorder=3,
                            ls="-" if k in cached else (0,(2,1.5))))
    ax.text(0.115, 0.30, "● cached   ○ evicted", ha="center", fontsize=7.2, color=GREY)

    # spawn arrow into runtime
    arrow(ax, (0.225, 0.60), (0.30, 0.60), color=INK, lw=1.8)
    ax.text(0.262, 0.635, "spawn", ha="center", fontsize=7.6, color=INK, style="italic")

    # ---- Center: BIC runtime with inner components ----
    box(ax, 0.30, 0.05, 0.45, 0.90, "", ec=BLUE, fc="#f7fafd", round=0.02, lw=1.8)
    ax.text(0.525, 0.905, "Bounded Infinity Cache (runtime)", ha="center", fontsize=9.5,
            fontweight="bold", color=BLUE)

    box(ax, 0.325, 0.77, 0.40, 0.10,
        "Deterministic addressing\n" + r"Cantor pairing  $\kappa(\mathrm{parent},\,\mathrm{child})\;\mapsto$ slot",
        ec=AMBER, fc=AMBERFILL, fs=8.2)

    # bounded cache: row of M slots
    ax.text(0.525, 0.705, "Bounded cache — $M$ fixed slots (full agent states)", ha="center",
            fontsize=8.2, color=INK)
    M = 8; sw = 0.044; x0 = 0.345
    filled = [1,1,1,0,1,1,0,1]
    for i in range(M):
        fc = BLUEFILL if filled[i] else "#ffffff"
        box(ax, x0 + i*sw, 0.60, sw*0.86, 0.075, "", ec=BLUE, fc=fc, round=0.012, lw=1.1)
    ax.text(0.525, 0.638, "", ha="center")

    # registry
    box(ax, 0.325, 0.45, 0.40, 0.105,
        "Agent registry — $\\mathcal{O}(N)$\n" + r"$\langle$id, parent, depth, slot, status$\rangle$ per agent",
        ec=GREY, fc=GREYFILL, fs=8.2)

    # eviction manager
    box(ax, 0.325, 0.22, 0.40, 0.185, "", ec=GREEN, fc=GREENFILL, round=0.02, lw=1.5)
    ax.text(0.525, 0.378, "Eviction manager (on full)", ha="center", fontsize=8.6,
            fontweight="bold", color=GREEN)
    ax.text(0.525, 0.32, "1. depth-priority select (evict deepest)", ha="center", fontsize=7.9, color=INK)
    ax.text(0.525, 0.285, r"2. summarise  $\varphi$  (hierarchical)", ha="center", fontsize=7.9, color=INK)
    ax.text(0.525, 0.25, "3. fold into nearest cached ancestor", ha="center", fontsize=7.9, color=INK)

    ax.text(0.525, 0.10, r"optional: Hilbert reorder for locality", ha="center", fontsize=7.4,
            color=GREY, style="italic")

    # internal arrows
    arrow(ax, (0.525, 0.77), (0.525, 0.685), color=AMBER, lw=1.3)
    arrow(ax, (0.525, 0.595), (0.525, 0.557), color=BLUE, lw=1.3)
    arrow(ax, (0.70, 0.50), (0.70, 0.407), color=GREEN, lw=1.2, rad=0.0)
    arrow(ax, (0.345, 0.407), (0.345, 0.50), color=GREEN, lw=1.2)

    # ---- Right: query paths ----
    arrow(ax, (0.75, 0.60), (0.83, 0.60), color=INK, lw=1.8)
    ax.text(0.79, 0.635, "query", ha="center", fontsize=7.6, color=INK, style="italic")
    box(ax, 0.80, 0.64, 0.195, 0.16,
        "Cache hit\n$\\mathcal{O}(1)$ exact state", ec=GREEN, fc=GREENFILL, fs=8.0)
    box(ax, 0.80, 0.40, 0.195, 0.18,
        "Cache miss\n$\\mathcal{O}(\\log N)$ ancestor-walk\nreconstruction", ec=AMBER, fc=AMBERFILL, fs=8.0)
    arrow(ax, (0.83, 0.585), (0.86, 0.64), color=GREEN, lw=1.2)
    arrow(ax, (0.83, 0.585), (0.86, 0.58), color=AMBER, lw=1.2)
    # registry feeds reconstruction
    arrow(ax, (0.725, 0.50), (0.80, 0.42), color=GREY, lw=1.1, ls=(0,(3,2)), rad=-0.25)
    ax.text(0.80, 0.345, "registry → reconstruction", ha="center", fontsize=6.6, color=GREY, style="italic")

    fig.savefig("fig_bic_architecture.pdf", bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


# ============================================================ FIG 2: OPERATIONS
def tree_panel(ax, title, cached, summarised, evicting=None, query=None, walk=None):
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.text(0.5, 0.98, title, ha="center", fontsize=9.5, fontweight="bold", color=INK)
    pos = {"r": (0.5, 0.82), "a": (0.27, 0.58), "b": (0.5, 0.58), "c": (0.73, 0.58),
           "a1": (0.17, 0.30), "a2": (0.37, 0.30), "c1": (0.63, 0.30), "c2": (0.83, 0.30)}
    edges = [("r","a"),("r","b"),("r","c"),("a","a1"),("a","a2"),("c","c1"),("c","c2")]
    for u,v in edges:
        ax.plot([pos[u][0],pos[v][0]],[pos[u][1],pos[v][1]], color="#999999", lw=1.0, zorder=1)
    for k,(x,y) in pos.items():
        if k in cached:
            fc, ec, ls = (GREENFILL, GREEN, "-")
        else:
            fc, ec, ls = ("#ffffff", EVICT, (0,(2,1.5)))
        ax.add_patch(Circle((x,y), 0.052, fc=fc, ec=ec, lw=1.6, zorder=3, ls=ls))
        if k in summarised:
            ax.add_patch(Circle((x,y), 0.052, fc="none", ec=AMBER, lw=2.4, zorder=4))
        ax.text(x, y, k, ha="center", va="center", fontsize=8, zorder=5, color=INK)
    # eviction fold arrows (child -> parent, amber, curved)
    if evicting:
        for child, parent in evicting:
            arrow(ax, pos[child], pos[parent], color=AMBER, lw=1.8, rad=0.25)
        ax.text(0.5, 0.10, r"evicted child  $\to$  nearest cached ancestor  ($\varphi$ summarise)",
                ha="center", fontsize=8.0, color=AMBER)
    # reconstruction walk arrows
    if walk:
        for u, v in walk:
            arrow(ax, pos[u], pos[v], color=BLUE, lw=1.8, rad=0.25)
        ax.text(0.5, 0.10, "walk registry up to nearest cached\nancestor; aggregate summaries",
                ha="center", fontsize=8.0, color=BLUE)
    if query:
        x,y = pos[query]
        ax.add_patch(Circle((x,y), 0.075, fc="none", ec=INK, lw=1.4, ls=(0,(1,1)), zorder=6))
        ax.annotate("query", (x, y-0.075), (x, y-0.17), ha="center", fontsize=7.6,
                    arrowprops=dict(arrowstyle="-|>", color=INK), color=INK)


def operations():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.3))
    # eviction: a1 evicted, folds into a; a2 evicted, folds into a (its parent, cached)
    tree_panel(ax1, "(a) Eviction & summarisation",
               cached={"r","a","b","c"}, summarised={"a"},
               evicting=[("a1","a"),("a2","a")])
    # reconstruction: query a1 (evicted), walk a1->a (cached, has summary)
    tree_panel(ax2, "(b) Miss reconstruction",
               cached={"r","b","c"}, summarised={"r"}, query="a1",
               walk=[("a1","a"),("a","r")])
    # legend
    handles = [
        Line2D([0],[0], marker='o', color='w', markerfacecolor=GREENFILL, markeredgecolor=GREEN, markersize=10, label='cached'),
        Line2D([0],[0], marker='o', color='w', markerfacecolor='white', markeredgecolor=EVICT, markersize=10, label='evicted', ls='--'),
        Line2D([0],[0], marker='o', color='w', markerfacecolor='none', markeredgecolor=AMBER, markersize=11, markeredgewidth=2.2, label='holds summaries'),
    ]
    fig.legend(handles=handles, loc='lower center', ncol=3, frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.02))
    fig.savefig("fig_bic_operations.pdf", bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


architecture()
operations()
print("wrote fig_bic_architecture.pdf, fig_bic_operations.pdf")
