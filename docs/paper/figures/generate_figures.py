"""Generate publication-quality figures for the BIC paper."""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 9,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})

DATA_DIR = '../../../experiments/results'
FIG_DIR = '.'


def load_json(name):
    with open(f'{DATA_DIR}/{name}') as f:
        return json.load(f)


# ── Figure 1: Quality vs Cache Size (line plot) ──
def fig_quality_vs_cache():
    data = load_json('experiment_2_quality.json')
    cache_sizes = [8, 16, 32, 64, 128, 256]
    n_backends = 7
    backend_order = ['bic', 'lru', 'lru-summary', 'tiered', 'fifo', 'random', 'unbounded']

    results = {b: [] for b in backend_order}
    for g, M in enumerate(cache_sizes):
        for j, b in enumerate(backend_order):
            entry = data[g * n_backends + j]
            assert entry['backend'] == b, f"Expected {b}, got {entry['backend']}"
            results[b].append(entry['quality']['reconstruction_quality'])

    fig, ax = plt.subplots(figsize=(5.5, 3.5))

    styles = {
        'bic':         {'color': '#d62728', 'marker': 'o', 'linewidth': 2.2, 'zorder': 10, 'label': 'BIC (ours)'},
        'lru':         {'color': '#1f77b4', 'marker': 's', 'linewidth': 1.3, 'zorder': 5, 'label': 'LRU'},
        'lru-summary': {'color': '#2ca02c', 'marker': '^', 'linewidth': 1.3, 'zorder': 5, 'label': 'LRU+Summary', 'linestyle': '--'},
        'tiered':      {'color': '#9467bd', 'marker': 'D', 'linewidth': 1.3, 'zorder': 5, 'label': 'Tiered (MemGPT)'},
        'fifo':        {'color': '#8c564b', 'marker': 'v', 'linewidth': 1.0, 'zorder': 3, 'label': 'FIFO', 'linestyle': ':'},
        'random':      {'color': '#7f7f7f', 'marker': 'x', 'linewidth': 1.0, 'zorder': 3, 'label': 'Random', 'linestyle': ':'},
        'unbounded':   {'color': '#bcbd22', 'marker': '*', 'linewidth': 1.0, 'zorder': 2, 'label': 'Unbounded', 'linestyle': '-.'},
    }

    for b in backend_order:
        s = styles[b]
        ax.plot(cache_sizes, results[b], markersize=5, **s)

    ax.set_xlabel('Cache Capacity $M$')
    ax.set_ylabel('Reconstruction Quality (Jaccard)')
    ax.set_xscale('log', base=2)
    ax.set_xticks(cache_sizes)
    ax.set_xticklabels([str(m) for m in cache_sizes])
    ax.set_ylim(-0.02, 1.05)
    ax.legend(loc='lower right', ncol=2, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_title('Quality vs. Cache Size ($N=121$ agents)')

    fig.savefig(f'{FIG_DIR}/quality_vs_cache.pdf')
    fig.savefig(f'{FIG_DIR}/quality_vs_cache.png')
    print('Saved quality_vs_cache.pdf/png')
    plt.close(fig)


# ── Figure 2: Stress test bar chart ──
def fig_stress_test():
    data = load_json('experiment_4_stress.json')

    backends = []
    recon = []
    query_success = []
    for entry in data:
        backends.append(entry['backend'])
        recon.append(entry['quality']['reconstruction_quality'])
        query_success.append(entry['quality']['query_success_rate'])

    labels = {
        'bic': 'BIC\n(ours)', 'lru': 'LRU', 'lru-summary': 'LRU+\nSummary',
        'tiered': 'Tiered\n(MemGPT)', 'fifo': 'FIFO', 'random': 'Random'
    }
    display = [labels.get(b, b) for b in backends]

    x = np.arange(len(backends))
    width = 0.35

    fig, ax = plt.subplots(figsize=(5.5, 3.2))

    colors_recon = ['#d62728' if b == 'bic' else '#4a7fb5' for b in backends]
    colors_qs = ['#ff7f7f' if b == 'bic' else '#a0c4e8' for b in backends]

    bars1 = ax.bar(x - width/2, recon, width, label='Recon. Quality',
                   color=colors_recon, edgecolor='white', linewidth=0.5)
    bars2 = ax.bar(x + width/2, query_success, width, label='Query Success',
                   color=colors_qs, edgecolor='white', linewidth=0.5)

    # Add value labels on BIC bars
    ax.text(0 - width/2, recon[0] + 0.02, f'{recon[0]:.3f}',
            ha='center', va='bottom', fontsize=8, fontweight='bold')
    ax.text(0 + width/2, query_success[0] + 0.02, f'{query_success[0]:.0%}',
            ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax.set_ylabel('Score')
    ax.set_xticks(x)
    ax.set_xticklabels(display, fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(True, axis='y', alpha=0.3)
    ax.set_title('Stress Test: 3,280 Agents, $M=64$ (1.9% retention)')

    fig.savefig(f'{FIG_DIR}/stress_test.pdf')
    fig.savefig(f'{FIG_DIR}/stress_test.png')
    print('Saved stress_test.pdf/png')
    plt.close(fig)


if __name__ == '__main__':
    fig_quality_vs_cache()
    fig_stress_test()
    print('All figures generated.')
