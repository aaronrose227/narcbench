"""Plot a comparison bar chart from two CSV files.

Edit the config section below to point at the two CSV files, then run:
    python comparison_plot.py
"""

import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from colours import TEAL_COLORS, RED_COLORS, GREEN_COLORS  # noqa: F401  (options for COLORS below)

# ── Config ────────────────────────────────────────────────────────────────

CSV_DIR = config.RESULTS_ROOT / "auroc_scores"

COLORS = TEAL_COLORS  # swap to TEAL_COLORS or RED_COLORS as desired

MAIN_STEM = "method_comparison_LR_C200_maxIters8"

MAIN_CSV      = CSV_DIR / f"{MAIN_STEM}.csv"
REFERENCE_CSV = None #CSV_DIR / "method_comparison_LR_C200.csv"

OUT_PATH = config.RESULTS_ROOT / "figures" / f"{MAIN_STEM}.png"

# ─────────────────────────────────────────────────────────────────────────

METHOD_KEYS = [
    "sorted_concat",
    "activation_alignment",
    "peak_suspicion",
    "deception_split",
    "asymmetry_probe",
]
METHOD_NAMES = [
    "Sorted\nConcat",
    "Activation\nAlignment",
    "Peak\nSuspicion",
    "Deception\nSplit",
    "Asymmetry\nProbe",
]
DISPLAY_NAMES = {
    "Held-out": "Held-Out",
    "cicd": "CI/CD\nPipeline",
    "code_review": "Code\nReview",
    "news": "News\nFraming",
    "pump_dump": "Pump &\nDump",
    "rag": "RAG\nPoisoning",
    "research": "Research\nPipeline",
    "blackjack": "Blackjack\nStego",
}


def _draw_bracket(ax, x_left, x_right, label, y_data=-0.17):
    """Draw a labelled bracket below the x-axis."""
    kw = dict(color="black", linewidth=0.8, clip_on=False,
              transform=ax.get_xaxis_transform())
    tick_h = 0.02
    ax.plot([x_left, x_right], [y_data, y_data], **kw)
    ax.plot([x_left, x_left], [y_data, y_data + tick_h], **kw)
    ax.plot([x_right, x_right], [y_data, y_data + tick_h], **kw)
    ax.text((x_left + x_right) / 2, y_data - 0.03, label,
            ha="center", va="top", fontsize=8, fontweight="bold",
            transform=ax.get_xaxis_transform(), clip_on=False)


def plot_figure(group_names, means, stds, reference_means=None):
    """Create the grouped bar chart.

    If reference_means is provided (dict of group -> method -> float), draws a
    thin horizontal black line across each bar at the reference value.
    """
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 8,
        "axes.linewidth": 0.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": "#e0e0e0",
        "grid.linewidth": 0.3,
        "legend.frameon": False,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
    })

    n_groups = len(group_names)
    n_methods = len(METHOD_KEYS)
    bar_width = 0.12
    group_width = n_methods * bar_width + 0.10
    group_centers = np.arange(n_groups) * group_width

    fig, ax = plt.subplots(figsize=(7.2, 3.5))

    for i, (method, color, label) in enumerate(zip(METHOD_KEYS, COLORS, METHOD_NAMES)):
        offsets = group_centers + (i - n_methods / 2 + 0.5) * bar_width
        vals = [means[g][method] for g in group_names]
        errs = [stds[g][method] for g in group_names]
        ax.bar(offsets, vals, bar_width, color=color, label=label,
               edgecolor="white", linewidth=0.2,
               yerr=errs, capsize=1.5,
               error_kw={"linewidth": 0.6, "color": "#333333", "capthick": 0.5})

        if reference_means is not None:
            for j, g in enumerate(group_names):
                if g in reference_means and method in reference_means[g]:
                    ref_val = reference_means[g][method]
                    x = offsets[j]
                    ax.plot([x - bar_width / 2, x + bar_width / 2],
                            [ref_val, ref_val],
                            color="black", linewidth=1.0, zorder=3,
                            solid_capstyle="butt")

    ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.6, alpha=0.5, zorder=0)
    ax.set_ylim(0.0, 1.08)
    ax.set_ylabel("AUROC", fontsize=8)
    ax.set_yticks(np.arange(0.0, 1.05, 0.2))
    ax.set_axisbelow(True)
    ax.set_xticks(group_centers)
    ax.set_xticklabels(
        [DISPLAY_NAMES.get(g) or g for g in group_names], fontsize=6.5, ha="center")

    pad = group_width * 0.45
    _draw_bracket(ax, group_centers[0] - pad, group_centers[0] + pad, "NARCBench-Core")
    _draw_bracket(ax, group_centers[1] - pad, group_centers[-2] + pad, "NARCBench-Transfer")
    _draw_bracket(ax, group_centers[-1] - pad, group_centers[-1] + pad, "NARCBench-Stego")

    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.18),
              ncol=n_methods, fontsize=7, frameon=False,
              handlelength=1.0, handletextpad=0.3, columnspacing=0.8)
    plt.subplots_adjust(bottom=0.28, top=0.85)
    return fig


def load_csv(path):
    """Load a results CSV and return (group_names, means, stds).

    group_names  - list of group names in the order they first appear
    means        - dict[group][method] -> float
    stds         - dict[group][method] -> float  (std-error across raw values)
    """
    means = {}
    raw_vals = defaultdict(lambda: defaultdict(list))  # group -> method -> [auroc]
    group_order = []

    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            rt = row["record_type"]
            if rt == "raw":
                g, m = row["group"], row["method"]
                if g not in raw_vals and g not in group_order:
                    group_order.append(g)
                try:
                    raw_vals[g][m].append(float(row["auroc"]))
                except ValueError:
                    pass
            elif rt == "mean":
                g, m = row["group"], row["method"]
                if g not in means:
                    means[g] = {}
                    if g not in group_order:
                        group_order.append(g)
                means[g][m] = float(row["auroc"])

    stds = {}
    for g in group_order:
        stds[g] = {}
        for m in METHOD_KEYS:
            vals = np.array(raw_vals[g][m], dtype=float)
            vals = vals[~np.isnan(vals)]
            if len(vals) > 1:
                stds[g][m] = np.std(vals) / np.sqrt(len(vals))
            else:
                stds[g][m] = 0.0
            means.setdefault(g, {}).setdefault(m, np.nanmean(vals) if len(vals) > 0 else 0.5)

    return group_order, means, stds


def load_reference_means(path):
    """Load only the mean rows from a CSV as reference_means dict[group][method]."""
    reference_means = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if row["record_type"] == "mean":
                g, m = row["group"], row["method"]
                if g not in reference_means:
                    reference_means[g] = {}
                reference_means[g][m] = float(row["auroc"])
    return reference_means


def main():
    print(f"Loading main CSV:      {MAIN_CSV}")
    group_names, means, stds = load_csv(MAIN_CSV)

    reference_means = None
    if REFERENCE_CSV is not None:
        print(f"Loading reference CSV: {REFERENCE_CSV}")
        reference_means = load_reference_means(REFERENCE_CSV)

    print(f"Groups: {group_names}")

    fig = plot_figure(group_names, means, stds, reference_means=reference_means)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to {OUT_PATH}")


if __name__ == "__main__":
    main()
