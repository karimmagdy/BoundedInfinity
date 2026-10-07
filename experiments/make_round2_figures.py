"""Round-2 figures, generated from the committed result files.

  fig_quality_vs_cache.pdf  GPT-5.4 50-agent sweep (results/round2/sweep.*.jsonl)
  fig_memory_vs_n.pdf       working-state memory vs swarm size
                            (results/synthetic_fixed/memory_scaling.json)

Run from the repository root:  python experiments/make_round2_figures.py
Writes to figures/round2/.
"""
import glob
import json
import statistics as st
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS = Path(__file__).resolve().parent / "results"
HERE = Path(__file__).resolve().parent.parent / "figures" / "round2"
HERE.mkdir(parents=True, exist_ok=True)

INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3dc"
SERIES = {  # fixed categorical order; marker shape is the secondary encoding
    "bic-aw": ("BIC", "#2a78d6", "o"),
    "lru-summary-aw-pin": ("Combined heuristic", "#eb6834", "s"),
    "lru": ("LRU", "#1baf7a", "^"),
}
REF = ("Unbounded", "#8a8984")

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "legend.frameon": False,
})


LABEL_NUDGE = {"BIC": 5, "Combined heuristic": -5}  # points; the two end values sit close


def label_end(ax, x, y, text, color):
    ax.annotate(text, (x, y), xytext=(6, LABEL_NUDGE.get(text, 0)), textcoords="offset points",
                va="center", fontsize=7.5, color=INK)


def quality_vs_cache():
    recs = [json.loads(l) for f in glob.glob(str(RESULTS / "round2" / "sweep.*.jsonl"))
            for l in open(f)]
    caches = [8, 16, 32]
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    for arm, (name, color, marker) in SERIES.items():
        means, sems = [], []
        for c in caches:
            xs = [r["quality"]["semantic_reconstruction_quality"] for r in recs
                  if r["backend"] == arm and r["cache_size"] == c]
            means.append(st.mean(xs))
            sems.append(st.stdev(xs) / len(xs) ** 0.5)
        ax.errorbar(caches, means, yerr=sems, color=color, marker=marker, markersize=5,
                    linewidth=1.5, capsize=2, markeredgecolor="white", markeredgewidth=0.8)
        label_end(ax, caches[-1], means[-1], name, color)
    ax.axhline(1.0, color=REF[1], linestyle="--", linewidth=1)
    ax.annotate("Unbounded (ceiling)", (8, 1.0), xytext=(0, -9), textcoords="offset points",
                fontsize=7.5, color=MUTED)
    ax.set_xscale("log", base=2)
    ax.set_xticks(caches, [str(c) for c in caches])
    ax.set_xlim(7, 60)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Cache capacity M (slots)")
    ax.set_ylabel("Reconstruction quality (TF-IDF cosine)")
    ax.legend(handles=[plt.Line2D([], [], color=c, marker=m, label=n, linewidth=1.5)
                       for n, c, m in SERIES.values()], loc="lower right", fontsize=7)
    fig.tight_layout()
    fig.savefig(HERE / "fig_quality_vs_cache.pdf")
    plt.close(fig)


def memory_vs_n():
    data = json.loads((RESULTS / "synthetic_fixed" / "memory_scaling.json").read_text())
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    for arm, (name, color, marker) in SERIES.items():
        rows = sorted((r["agents"], r["run_peak_bytes"] / 1024) for r in data if r["backend"] == arm)
        ax.plot(*zip(*rows), color=color, marker=marker, markersize=5, linewidth=1.5,
                markeredgecolor="white", markeredgewidth=0.8, label=name)
    rows = sorted((r["agents"], r["run_peak_bytes"] / 1024) for r in data if r["backend"] == "unbounded")
    ax.plot(*zip(*rows), color=REF[1], linestyle="--", marker="D", markersize=4, linewidth=1.2,
            label=REF[0])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Agents in the swarm (N)")
    ax.set_ylabel("Peak working memory (KB)")
    ax.legend(loc="upper left", fontsize=7)
    fig.tight_layout()
    fig.savefig(HERE / "fig_memory_vs_n.pdf")
    plt.close(fig)


if __name__ == "__main__":
    quality_vs_cache()
    memory_vs_n()
