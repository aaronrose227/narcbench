"""Generate ROC curves and Recall@1%FPR analysis for NARCBench.

  - Raw projected scores (not probabilities)
  - 1000-threshold sweep for smooth ROC curves
  - Recall@1%FPR via quantile threshold on control scores
  - Two-panel ROC: full range + zoomed to 3% FPR

Usage:
    PYTHONPATH=. python3 probes/roc_analysis.py                              # default model
    PYTHONPATH=. python3 probes/roc_analysis.py --model <hf_id> --layers 32-37
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import config
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, roc_auc_score

# Reuse data loading and probe infrastructure from reproduce.py
from reproduce import (
    METHOD_KEYS, DISPLAY_NAMES,
    load_data, build_groups, train_deception_probe,
    compute_scores, _run_labels,
    _train_asymmetry_classifier, _train_concat_classifier,
)

# Set in main() from --model and --layers; see config.MODEL_LAYERS for defaults.
DATA_DIR = None
CORE_DIR = None
TRANSFER_DIR = None
STEGO_DIR = None
LAYERS = None
OUT_DIR = config.RESULTS_ROOT / "figures"

# Pastel blue-toned palette
# Pastel reds, oranges, pinks palette
COLORS = {
    "sorted_concat": "#C0392B",
    "activation_alignment": "#E74C3C",
    "peak_suspicion": "#E67E22",
    "deception_split": "#F39C8B",
    "asymmetry_probe": "#D4577A",
}
METHOD_LABELS = {
    "sorted_concat": "Sorted Concatenation",
    "activation_alignment": "Activation Alignment",
    "peak_suspicion": "Peak Suspicion",
    "deception_split": "Deception Split",
    "asymmetry_probe": "Asymmetry Probe",
}


# ── ROC computation (Apollo-style: 1000-threshold sweep) ─────────────────

def get_fprs_tprs(honest_scores, deceptive_scores, n_thresholds=1000):
    """Compute FPR/TPR curve by sweeping thresholds.

    Same approach as Apollo: sweep linearly spaced thresholds and compute
    the fraction of positive/negative samples exceeding each threshold.
    """
    all_scores = np.concatenate([honest_scores, deceptive_scores])
    min_s, max_s = all_scores.min(), all_scores.max()
    thresholds = np.linspace(min_s - 0.1, max_s + 0.1, n_thresholds)

    tprs = [(deceptive_scores > t).mean() for t in thresholds]
    fprs = [(honest_scores > t).mean() for t in thresholds]

    return np.array(fprs), np.array(tprs), thresholds


def recall_at_fpr(honest_scores, deceptive_scores, target_fpr=0.01):
    """Compute recall (TPR) at a target FPR.

    Same approach as Apollo: threshold = (1-fpr) quantile of honest scores,
    recall = fraction of deceptive scores exceeding threshold.
    """
    threshold = np.quantile(honest_scores, 1 - target_fpr)
    recall = (deceptive_scores > threshold).mean()
    return recall


# ── Score collection ─────────────────────────────────────────────────────

def collect_scores_for_tier(groups, rids, scaler, w, core_scores=None, core_y=None):
    """Collect per-method scores split into collusion and control."""
    scores = compute_scores(groups, rids, scaler, w)
    y = _run_labels(groups, rids)

    collusion_mask = y == 1
    control_mask = y == 0

    result = {}
    for key in ["peak_suspicion", "deception_split", "activation_alignment"]:
        result[key] = {
            "collusion": scores[key][collusion_mask],
            "control": scores[key][control_mask],
        }

    # Asymmetry Probe (needs classifier)
    if core_scores is not None and core_y is not None:
        asym_fs, asym_clf = _train_asymmetry_classifier(core_scores, core_y)
        asym_proba = asym_clf.predict_proba(
            asym_fs.transform(scores["asymmetry_probe"]))[:, 1]
        result["asymmetry_probe"] = {
            "collusion": asym_proba[collusion_mask],
            "control": asym_proba[control_mask],
        }

        # Sorted Concatenation (needs PCA + classifier)
        concat_pca, concat_fs, concat_clf = _train_concat_classifier(core_scores, core_y)
        concat_proba = concat_clf.predict_proba(
            concat_fs.transform(
                concat_pca.transform(scores["sorted_concat"])))[:, 1]
        result["sorted_concat"] = {
            "collusion": concat_proba[collusion_mask],
            "control": concat_proba[control_mask],
        }

    return result


# ── Plotting ─────────────────────────────────────────────────────────────

def _setup_style():
    """Set up professional plot style (seaborn deep inspired)."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "font.size": 10,
        "axes.linewidth": 0.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 150,
        "savefig.dpi": 400,
        "pdf.fonttype": 42,
    })


def plot_roc_single(tier_scores, tier_name, out_path):
    """Single-panel ROC plot, Apollo style: square, percentage axes, clean."""
    _setup_style()

    fig, ax = plt.subplots(figsize=(5, 5))

    for method in METHOD_KEYS:
        if method not in tier_scores:
            continue
        collusion = tier_scores[method]["collusion"]
        control = tier_scores[method]["control"]

        fprs, tprs, _ = get_fprs_tprs(control, collusion)
        auroc = auc(fprs, tprs)

        color = COLORS[method]
        ax.plot(fprs * 100, tprs * 100, color=color, linewidth=1.5,
                label=f"{METHOD_LABELS[method]} ({auroc:.2f})")

    ax.plot([0, 100], [0, 100], "k--", linewidth=0.8, alpha=0.3)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 102)
    ax.set_xlabel("False Positive Rate (%)")
    ax.set_ylabel("True Positive Rate (%)")
    ax.set_title(tier_name)
    ax.set_aspect("equal")
    ax.legend(fontsize=8, loc="lower right", frameon=False)

    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


def plot_roc_all_tiers(all_tier_scores, out_path):
    """Three-panel ROC plot, one per tier. Square, percentage axes."""
    _setup_style()

    fig = plt.figure(figsize=(7, 2.3))
    # Two plot panels on the left, legend space on the right
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.45], wspace=0.35)
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
    axes[1].sharey(axes[0])
    tier_names = ["Transfer", "Stego"]

    for ax, tier_name in zip(axes, tier_names):
        tier_key = tier_name.lower()
        if tier_key not in all_tier_scores:
            continue
        tier_scores = all_tier_scores[tier_key]

        for method in METHOD_KEYS:
            if method not in tier_scores:
                continue
            collusion = tier_scores[method]["collusion"]
            control = tier_scores[method]["control"]

            fprs, tprs, _ = get_fprs_tprs(control, collusion)
            auroc = auc(fprs, tprs)
            color = COLORS[method]
            ax.plot(fprs * 100, tprs * 100, color=color, linewidth=1.5,
                    label=f"{METHOD_LABELS[method]} ({auroc:.2f})")

        ax.plot([0, 100], [0, 100], "k--", linewidth=0.8, alpha=0.3)
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 102)
        ax.set_xlabel("False Positive Rate (%)")
        ax.set_title(f"NARCBench-{tier_name}", fontsize=9)
        ax.set_aspect("equal")

    axes[0].set_ylabel("True Positive Rate (%)")
    plt.setp(axes[1].get_yticklabels(), visible=False)
    # Legend on the right side
    handles, labels = axes[0].get_legend_handles_labels()
    leg_ax = fig.add_subplot(gs[0, 2])
    leg_ax.axis("off")
    leg_ax.legend(handles, labels, loc="center left", fontsize=7,
                  frameon=False, borderpad=0)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


# ── Main ─────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default=config.DEFAULT_MODEL,
                   help=f"HuggingFace model ID (default: {config.DEFAULT_MODEL}).")
    p.add_argument("--layers", default=None,
                   help="Probe-layer range, e.g. '26-30' or '26,28,30'. "
                        "Defaults to config.MODEL_LAYERS for known models.")
    return p.parse_args()


def main():
    global DATA_DIR, CORE_DIR, TRANSFER_DIR, STEGO_DIR, LAYERS
    args = parse_args()

    short = config.model_short_name(args.model)
    DATA_DIR = config.activations_dir(short)
    CORE_DIR = DATA_DIR / "core"
    TRANSFER_DIR = DATA_DIR / "transfer"
    STEGO_DIR = DATA_DIR / "stego"
    LAYERS = (config.parse_layer_range(args.layers) if args.layers
              else config.default_layers(args.model))

    print(f"Model: {args.model}  (short: {short})")
    print(f"Layers: {LAYERS}")

    print("\nLoading data...")
    core_meta, core_npz = load_data(CORE_DIR)
    transfer_meta, transfer_npz = load_data(TRANSFER_DIR)
    stego_meta, stego_npz = load_data(STEGO_DIR)

    layer = LAYERS[len(LAYERS) // 2]
    print(f"Computing scores at layer {layer}...")

    # Train global probe on all Core
    scaler, w = train_deception_probe(core_meta, core_npz, layer)

    # Build groups
    core_groups = build_groups(core_meta, core_npz, layer, phase="discussion", avg_rounds=True)
    transfer_groups = build_groups(transfer_meta, transfer_npz, layer, phase="final", avg_rounds=False)
    stego_groups = build_groups(stego_meta, stego_npz, layer, phase="final", avg_rounds=False)

    core_rids = sorted(core_groups.keys())
    transfer_rids = sorted(transfer_groups.keys())
    stego_rids = sorted(stego_groups.keys())

    core_scores_raw = compute_scores(core_groups, core_rids, scaler, w)
    core_y = _run_labels(core_groups, core_rids)

    # Collect scores per tier
    print("Collecting scores...")
    all_tier_scores = {}

    all_tier_scores["core"] = collect_scores_for_tier(
        core_groups, core_rids, scaler, w, core_scores_raw, core_y)
    all_tier_scores["transfer"] = collect_scores_for_tier(
        transfer_groups, transfer_rids, scaler, w, core_scores_raw, core_y)
    all_tier_scores["stego"] = collect_scores_for_tier(
        stego_groups, stego_rids, scaler, w, core_scores_raw, core_y)

    # Print Recall@1%FPR table
    print("\n" + "=" * 70)
    print(f"{'':>25}{'AUROC':>10}{'R@1%FPR':>10}{'R@5%FPR':>10}")
    print("-" * 70)
    for tier_name, tier_key in [("Core", "core"), ("Transfer", "transfer"), ("Stego", "stego")]:
        print(f"\n  {tier_name}")
        for method in METHOD_KEYS:
            if method not in all_tier_scores[tier_key]:
                continue
            col = all_tier_scores[tier_key][method]["collusion"]
            ctrl = all_tier_scores[tier_key][method]["control"]
            fprs, tprs, _ = get_fprs_tprs(ctrl, col)
            auroc_val = auc(fprs, tprs)
            r1 = recall_at_fpr(ctrl, col, 0.01)
            r5 = recall_at_fpr(ctrl, col, 0.05)
            print(f"    {METHOD_LABELS[method]:>23}{auroc_val:>10.3f}{r1:>10.3f}{r5:>10.3f}")
    print("=" * 70)

    # Generate plots
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\nGenerating ROC curves...")
    for tier_name, tier_key in [("Core", "core"), ("Transfer", "transfer"), ("Stego", "stego")]:
        plot_roc_single(
            all_tier_scores[tier_key], f"NARCBench-{tier_name}",
            OUT_DIR / f"roc_{tier_key}.pdf")
        plot_roc_single(
            all_tier_scores[tier_key], f"NARCBench-{tier_name}",
            OUT_DIR / f"roc_{tier_key}.png")

    plot_roc_all_tiers(all_tier_scores, OUT_DIR / "roc_all_tiers.pdf")
    plot_roc_all_tiers(all_tier_scores, OUT_DIR / "roc_all_tiers.png")

    print("\nDone.")


if __name__ == "__main__":
    main()
