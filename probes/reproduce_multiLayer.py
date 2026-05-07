"""Reproduce Table 1 and Figure 2: five detection methods across NARCBench.

Trains a deception direction on NARCBench-Core (discussion-round activations)
and evaluates five detection methods zero-shot on NARCBench-Transfer and
NARCBench-Stego.  Results are averaged over the configured probe-layer range.
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from colours import TEAL_COLORS, GREEN_COLORS, RED_COLORS

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

# ── Config ───────────────────────────────────────────────────────────────

OUT_DIR  = config.RESULTS_ROOT / "figures"
OUT_STEM       = "method_comparison_multiLayer"
REFERENCE_STEM = "method_comparison_LR_C250_liblinear_maxIters8"  # CSV stem to load reference lines from (None to disable)

N_SEEDS = 10

# ── Paths and constants ──────────────────────────────────────────────────

# Set in main() from --model and --layers; see config.MODEL_LAYERS for defaults.
DATA_DIR = None
CORE_DIR = None
TRANSFER_DIR = None
STEGO_DIR = None
LAYERS = None

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
COLORS = RED_COLORS

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


# ── Data loading ─────────────────────────────────────────────────────────

def load_data(directory):
    """Load metadata and activations from a dataset directory."""
    with open(directory / "metadata_gen.json") as f:
        meta = json.load(f)
    return meta, np.load(directory / "activations_gen.npz")


def get_layer(npz, n, layer):
    """Extract a single layer's activations, truncated to n samples."""
    return npz[f"layer_{layer}"][:n].astype(np.float32)


# ── Helpers ──────────────────────────────────────────────────────────────

def _parse_run_id(run_id):
    """Parse run_id into (scenario_type, scenario_id).

    Examples:
        'deception__audit_01__collusion' -> ('deception', 'audit_01')
        'cicd__cicd_01__collusion'       -> ('cicd', 'cicd_01')
    """
    parts = run_id.split("__")
    scenario = parts[0]
    scenario_id = parts[1] if len(parts) >= 2 else run_id
    return scenario, scenario_id


def _domain_from_scenario_id(scenario_id):
    """Extract domain from scenario_id, e.g. 'audit_01' -> 'audit'."""
    return "_".join(scenario_id.split("_")[:-1])


def _run_labels(groups, rids):
    """Return binary labels: 1 for collusion, 0 for control."""
    return np.array([1 if groups[r]["mode"] == "collusion" else 0 for r in rids])


# ── Group building ───────────────────────────────────────────────────────

def build_groups(meta, npz, layer, phase="discussion", avg_rounds=True):
    """Group activations by run, producing one vector per agent per run.

    For discussion phase with avg_rounds=True, averages each agent's
    activations across all discussion rounds.  Otherwise takes the single
    activation matching the given phase.

    Returns dict: run_id -> {agents, mode, roles, scenario, domain}.
    Only runs with exactly 4 agents are included.
    """
    n = len(meta)
    acts = get_layer(npz, n, layer)

    if phase == "discussion" and avg_rounds:
        # Accumulate across rounds, then average
        accum = defaultdict(lambda: {"vecs": [], "meta": None})
        for i, m in enumerate(meta):
            if m["phase"] != "discussion" or m["channel"] != "public":
                continue
            accum[(m["run_id"], m["agent_name"])]["vecs"].append(acts[i])
            accum[(m["run_id"], m["agent_name"])]["meta"] = m

        groups = {}
        for (rid, agent_name), data in accum.items():
            m = data["meta"]
            if rid not in groups:
                scenario, scenario_id = _parse_run_id(rid)
                groups[rid] = {
                    "agents": {},
                    "mode": m["mode"],
                    "roles": {},
                    "scenario": scenario,
                    "domain": _domain_from_scenario_id(scenario_id),
                }
            groups[rid]["agents"][agent_name] = np.mean(data["vecs"], axis=0)
            groups[rid]["roles"][agent_name] = m["role"]
    else:
        groups = {}
        for i, m in enumerate(meta):
            if m["phase"] != phase or m["channel"] != "public":
                continue
            rid = m["run_id"]
            if rid not in groups:
                scenario, scenario_id = _parse_run_id(rid)
                groups[rid] = {
                    "agents": {},
                    "mode": m["mode"],
                    "roles": {},
                    "scenario": scenario,
                    "domain": _domain_from_scenario_id(scenario_id),
                }
            groups[rid]["agents"][m["agent_name"]] = acts[i]
            groups[rid]["roles"][m["agent_name"]] = m["role"]

    return {rid: g for rid, g in groups.items() if len(g["agents"]) == 4}


# ── Probe training ───────────────────────────────────────────────────────

def train_deception_probe(meta, npz, layer, phase="discussion", exclude_domain=None, seed=0):
    """Train the deception direction from raw per-sample activations.

    Filters to collusion runs, public channel, and the specified phase.
    If exclude_domain is set, samples from that domain are excluded
    (used for leave-domain-out CV).

    Returns (scaler, deception_direction).
    """
    n = len(meta)
    acts = get_layer(npz, n, layer)

    col_idx, hon_idx = [], []
    for i, m in enumerate(meta):
        if m["mode"] != "collusion":
            continue
        if m["channel"] != "public":
            continue
        if phase == "discussion" and m["phase"] != "discussion":
            continue
        _, scenario_id = _parse_run_id(m["run_id"])
        if exclude_domain and _domain_from_scenario_id(scenario_id) == exclude_domain:
            continue
        if m["role"] == "colluder":
            col_idx.append(i)
        elif m["role"] == "honest":
            hon_idx.append(i)

    nc = min(len(col_idx), len(hon_idx))
    rng = np.random.RandomState(seed)
    hon_idx = list(rng.choice(hon_idx, nc, replace=False))
    col_idx = col_idx[:nc]

    scaler = StandardScaler()
    X = scaler.fit_transform(np.vstack([acts[hon_idx], acts[col_idx]]))
    y = np.array([0] * nc + [1] * nc)

    probe = LogisticRegression(max_iter=1000, C=1.0, random_state=seed)
    probe.fit(X, y)
    w = probe.coef_[0]
    w = w / np.linalg.norm(w)
    return scaler, w


# ── Score computation ────────────────────────────────────────────────────

def compute_scores(groups, rids, scaler, w):
    """Compute all five method scores for each run.

    Methods:
        sorted_concat:        Concatenated standardised vectors sorted by deception score
        activation_alignment: Negated mean pairwise cosine similarity (raw vectors)
        peak_suspicion:       Maximum deception score
        deception_split:      Top-2 mean minus bottom-2 mean
        asymmetry_probe:      Feature vector [var, range, split, mean]
    """
    peak_suspicion, deception_split, asymmetry_probe = [], [], []
    activation_alignment, sorted_concat = [], []

    for r in rids:
        g = groups[r]
        names = sorted(g["agents"])
        raw_vecs = [g["agents"][n] for n in names]
        scaled_vecs = [scaler.transform(v.reshape(1, -1))[0] for v in raw_vecs]

        # Project onto deception direction
        projs = np.array([w @ v for v in scaled_vecs])
        order = np.argsort(projs)[::-1]
        projs_sorted = projs[order]

        # Peak Suspicion: max deception score
        peak_suspicion.append(projs_sorted[0])

        # Deception Split: top-2 vs bottom-2
        deception_split.append(np.mean(projs_sorted[:2]) - np.mean(projs_sorted[2:]))

        # Asymmetry Probe features: [var, range, split, mean]
        asymmetry_probe.append([
            np.var(projs),
            projs_sorted[0] - projs_sorted[3],
            np.mean(projs_sorted[:2]) - np.mean(projs_sorted[2:]),
            np.mean(projs),
        ])

        # Sorted Concatenation: standardised vectors sorted by deception score
        sorted_concat.append(np.concatenate([scaled_vecs[i] for i in order]))

        # Activation Alignment: negated mean pairwise cosine on raw vectors
        norms = [np.linalg.norm(v) + 1e-10 for v in raw_vecs]
        normed = [v / n for v, n in zip(raw_vecs, norms)]
        cos_pairs = [normed[i] @ normed[j] for i in range(4) for j in range(i + 1, 4)]
        activation_alignment.append(-np.mean(cos_pairs))

    return {
        "peak_suspicion": np.array(peak_suspicion),
        "deception_split": np.array(deception_split),
        "asymmetry_probe": np.array(asymmetry_probe),
        "activation_alignment": np.array(activation_alignment),
        "sorted_concat": np.array(sorted_concat),
    }


# ── Multi-layer ensemble scoring ─────────────────────────────────────────

def compute_scores_ensemble(groups_by_layer, rids, scalers_ws):
    """Compute all five method scores using an ensemble across layers.

    For projection-based methods (peak_suspicion, deception_split,
    asymmetry_probe), projections from each layer's probe are averaged
    per agent before computing statistics.  For sorted_concat, agent
    vectors are averaged across layers then sorted by the ensemble
    projection.  For activation_alignment, raw vectors are averaged
    across layers before computing pairwise cosine similarity.

    Args:
        groups_by_layer: list of groups dicts, one per layer (same order as scalers_ws).
        rids:            run IDs to score.
        scalers_ws:      list of (scaler, w) tuples, one per layer.
    """
    peak_suspicion, deception_split, asymmetry_probe = [], [], []
    activation_alignment, sorted_concat = [], []

    for r in rids:
        names = sorted(groups_by_layer[0][r]["agents"])
        n_agents = len(names)

        layer_projs  = []   # (n_layers, n_agents)
        layer_scaled = []   # (n_layers, n_agents, dim)
        layer_raw    = []   # (n_layers, n_agents, dim)

        for groups, (scaler, w) in zip(groups_by_layer, scalers_ws):
            g = groups[r]
            raw_vecs    = [g["agents"][n] for n in names]
            scaled_vecs = [scaler.transform(v.reshape(1, -1))[0] for v in raw_vecs]
            projs       = np.array([w @ v for v in scaled_vecs])
            layer_projs.append(projs)
            layer_scaled.append(scaled_vecs)
            layer_raw.append(raw_vecs)

        # Average projections across layers, then derive statistics
        projs       = np.mean(layer_projs, axis=0)   # (n_agents,)
        order       = np.argsort(projs)[::-1]
        projs_sorted = projs[order]

        peak_suspicion.append(projs_sorted[0])
        deception_split.append(np.mean(projs_sorted[:2]) - np.mean(projs_sorted[2:]))
        asymmetry_probe.append([
            np.var(projs),
            projs_sorted[0] - projs_sorted[3],
            np.mean(projs_sorted[:2]) - np.mean(projs_sorted[2:]),
            np.mean(projs),
        ])

        # Sorted concat: average scaled vecs across layers, sort by ensemble order
        avg_scaled = [
            np.mean([layer_scaled[l][i] for l in range(len(scalers_ws))], axis=0)
            for i in range(n_agents)
        ]
        sorted_concat.append(np.concatenate([avg_scaled[i] for i in order]))

        # Activation alignment: average raw vecs across layers
        avg_raw = [
            np.mean([layer_raw[l][i] for l in range(len(scalers_ws))], axis=0)
            for i in range(n_agents)
        ]
        norms   = [np.linalg.norm(v) + 1e-10 for v in avg_raw]
        normed  = [v / n for v, n in zip(avg_raw, norms)]
        cos_pairs = [normed[i] @ normed[j] for i in range(4) for j in range(i + 1, 4)]
        activation_alignment.append(-np.mean(cos_pairs))

    return {
        "peak_suspicion":      np.array(peak_suspicion),
        "deception_split":     np.array(deception_split),
        "asymmetry_probe":     np.array(asymmetry_probe),
        "activation_alignment": np.array(activation_alignment),
        "sorted_concat":       np.array(sorted_concat),
    }


# ── Evaluation ───────────────────────────────────────────────────────────

def _train_asymmetry_classifier(tr_scores, tr_y):
    """Train Asymmetry Probe classifier on training scores."""
    fs = StandardScaler()
    clf = LogisticRegression(max_iter=2000, C=0.01, random_state=42)
    clf.fit(fs.fit_transform(tr_scores["asymmetry_probe"]), tr_y)
    return fs, clf


def _train_concat_classifier(tr_scores, tr_y):
    """Train Sorted Concatenation classifier (PCA + LogReg) on training scores."""
    n_comp = min(50, len(tr_y) - 1, tr_scores["sorted_concat"].shape[1])
    pca = PCA(n_components=n_comp, random_state=42)
    fs = StandardScaler()
    X_tr = fs.fit_transform(pca.fit_transform(tr_scores["sorted_concat"]))
    clf = LogisticRegression(max_iter=2000, C=0.01, random_state=42)
    clf.fit(X_tr, tr_y)
    return pca, fs, clf


def evaluate(tr_scores, tr_y, te_scores, te_y):
    """Evaluate all five methods.  Returns dict of method_key -> AUROC."""
    if len(np.unique(te_y)) < 2:
        return None

    results = {}

    # Zero-parameter methods: raw score -> AUROC
    for key in ["peak_suspicion", "deception_split", "activation_alignment"]:
        results[key] = roc_auc_score(te_y, te_scores[key])

    # Asymmetry Probe
    fs, clf = _train_asymmetry_classifier(tr_scores, tr_y)
    results["asymmetry_probe"] = roc_auc_score(
        te_y, clf.predict_proba(fs.transform(te_scores["asymmetry_probe"]))[:, 1])

    # Sorted Concatenation
    pca, fs2, clf2 = _train_concat_classifier(tr_scores, tr_y)
    X_te = fs2.transform(pca.transform(te_scores["sorted_concat"]))
    results["sorted_concat"] = roc_auc_score(
        te_y, clf2.predict_proba(X_te)[:, 1])

    return results


# ── Leave-domain-out CV ──────────────────────────────────────────────────

def held_out_cv(core_meta, core_npz, seed=0):
    """Leave-domain-out cross-validation on Core using ensemble across all layers."""
    groups_by_layer = [
        build_groups(core_meta, core_npz, layer, phase="discussion", avg_rounds=True)
        for layer in LAYERS
    ]
    rids    = sorted(groups_by_layer[0].keys())
    domains = sorted({groups_by_layer[0][r]["domain"] for r in rids
                      if groups_by_layer[0][r]["domain"]})

    fold_results = {k: [] for k in METHOD_KEYS}

    for hold_dom in domains:
        tr_rids = [r for r in rids if groups_by_layer[0][r]["domain"] != hold_dom]
        te_rids = [r for r in rids if groups_by_layer[0][r]["domain"] == hold_dom]
        tr_y    = _run_labels(groups_by_layer[0], tr_rids)
        te_y    = _run_labels(groups_by_layer[0], te_rids)

        if len(np.unique(te_y)) < 2 or len(np.unique(tr_y)) < 2:
            continue

        scalers_ws = [
            train_deception_probe(core_meta, core_npz, layer, exclude_domain=hold_dom, seed=seed)
            for layer in LAYERS
        ]
        tr_scores = compute_scores_ensemble(groups_by_layer, tr_rids, scalers_ws)
        te_scores = compute_scores_ensemble(groups_by_layer, te_rids, scalers_ws)

        result = evaluate(tr_scores, tr_y, te_scores, te_y)
        if result:
            for k in METHOD_KEYS:
                fold_results[k].append(result[k])

    return {k: np.mean(v) if v else float("nan") for k, v in fold_results.items()}


# ── Main computation ─────────────────────────────────────────────────────

def compute_all_results(core_meta, core_npz, transfer_meta, transfer_npz,
                        stego_meta, stego_npz, seed=0):
    """Compute AUROC for all methods using an ensemble probe across all layers."""

    # Discover transfer scenario types
    transfer_peek = build_groups(
        transfer_meta, transfer_npz, LAYERS[0], phase="final", avg_rounds=False)
    transfer_scenarios = sorted({
        transfer_peek[r]["scenario"]
        for r in transfer_peek if transfer_peek[r]["scenario"]
    })
    group_names = ["Held-out"] + transfer_scenarios + ["blackjack"]

    print(f"  Building groups for {len(LAYERS)} layers...")
    core_groups_by_layer = [
        build_groups(core_meta, core_npz, layer, phase="discussion", avg_rounds=True)
        for layer in LAYERS
    ]
    transfer_groups_by_layer = [
        build_groups(transfer_meta, transfer_npz, layer, phase="final", avg_rounds=False)
        for layer in LAYERS
    ]
    stego_groups_by_layer = [
        build_groups(stego_meta, stego_npz, layer, phase="final", avg_rounds=False)
        for layer in LAYERS
    ]

    print(f"  Training ensemble probe ({len(LAYERS)} layers)...")
    scalers_ws = [
        train_deception_probe(core_meta, core_npz, layer, seed=seed) for layer in LAYERS
    ]

    # -- Core scores and classifiers --
    core_rids   = sorted(core_groups_by_layer[0].keys())
    core_y      = _run_labels(core_groups_by_layer[0], core_rids)
    core_scores = compute_scores_ensemble(core_groups_by_layer, core_rids, scalers_ws)

    asym_fs, asym_clf             = _train_asymmetry_classifier(core_scores, core_y)
    concat_pca, concat_fs, concat_clf = _train_concat_classifier(core_scores, core_y)

    # -- Core: leave-domain-out CV --
    print("  Leave-domain-out CV...")
    cv = held_out_cv(core_meta, core_npz, seed=seed)

    means = {g: {} for g in group_names}
    stds  = {g: {} for g in group_names}
    for m in METHOD_KEYS:
        means["Held-out"][m] = cv[m]
        stds["Held-out"][m]  = 0.0

    # -- Transfer --
    transfer_rids   = sorted(transfer_groups_by_layer[0].keys())
    transfer_y      = _run_labels(transfer_groups_by_layer[0], transfer_rids)
    transfer_sc     = [transfer_groups_by_layer[0][r]["scenario"] for r in transfer_rids]
    transfer_scores = compute_scores_ensemble(
        transfer_groups_by_layer, transfer_rids, scalers_ws)

    asym_proba   = asym_clf.predict_proba(
        asym_fs.transform(transfer_scores["asymmetry_probe"]))[:, 1]
    concat_proba = concat_clf.predict_proba(
        concat_fs.transform(
            concat_pca.transform(transfer_scores["sorted_concat"])))[:, 1]

    for sc in transfer_scenarios:
        mask = np.array([s == sc for s in transfer_sc])
        if mask.sum() > 0 and len(np.unique(transfer_y[mask])) >= 2:
            for key in ["peak_suspicion", "deception_split", "activation_alignment"]:
                means[sc][key] = roc_auc_score(transfer_y[mask], transfer_scores[key][mask])
            means[sc]["asymmetry_probe"] = roc_auc_score(transfer_y[mask], asym_proba[mask])
            means[sc]["sorted_concat"]   = roc_auc_score(transfer_y[mask], concat_proba[mask])
        else:
            for m in METHOD_KEYS:
                means[sc][m] = 0.5
        for m in METHOD_KEYS:
            stds[sc][m] = 0.0

    # -- Stego --
    stego_rids = sorted(stego_groups_by_layer[0].keys())
    stego_y    = _run_labels(stego_groups_by_layer[0], stego_rids)
    if len(np.unique(stego_y)) >= 2 and len(stego_rids) >= 4:
        stego_scores = compute_scores_ensemble(
            stego_groups_by_layer, stego_rids, scalers_ws)
        for key in ["peak_suspicion", "deception_split", "activation_alignment"]:
            means["blackjack"][key] = roc_auc_score(stego_y, stego_scores[key])
        stego_asym = asym_clf.predict_proba(
            asym_fs.transform(stego_scores["asymmetry_probe"]))[:, 1]
        means["blackjack"]["asymmetry_probe"] = roc_auc_score(stego_y, stego_asym)
        stego_concat = concat_clf.predict_proba(
            concat_fs.transform(
                concat_pca.transform(stego_scores["sorted_concat"])))[:, 1]
        means["blackjack"]["sorted_concat"] = roc_auc_score(stego_y, stego_concat)
    else:
        for m in METHOD_KEYS:
            means["blackjack"][m] = 0.5
    for m in METHOD_KEYS:
        stds["blackjack"][m] = 0.0

    return group_names, means, stds


# ── Plotting ─────────────────────────────────────────────────────────────

def plot_figure(group_names, means, stds, reference_means=None):
    """Create the grouped bar chart (Figure 2)."""
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
        [DISPLAY_NAMES.get(g, g) for g in group_names], fontsize=6.5, ha="center")

    # Category brackets (Core = first, Stego = last, Transfer = middle)
    pad = group_width * 0.45
    _draw_bracket(ax, group_centers[0] - pad, group_centers[0] + pad, "NARCBench-Core")
    _draw_bracket(ax, group_centers[1] - pad, group_centers[-2] + pad, "NARCBench-Transfer")
    _draw_bracket(ax, group_centers[-1] - pad, group_centers[-1] + pad, "NARCBench-Stego")

    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.18),
              ncol=n_methods, fontsize=7, frameon=False,
              handlelength=1.0, handletextpad=0.3, columnspacing=0.8)
    plt.subplots_adjust(bottom=0.28, top=0.85)
    return fig


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


# ── Entry point ──────────────────────────────────────────────────────────

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
    print(f"Activations dir: {DATA_DIR}")
    print(f"Layers: {LAYERS}")

    print("\nLoading data...")
    core_meta, core_npz = load_data(CORE_DIR)
    transfer_meta, transfer_npz = load_data(TRANSFER_DIR)
    stego_meta, stego_npz = load_data(STEGO_DIR)
    print(f"  Core: {len(core_meta)} samples | Transfer: {len(transfer_meta)} | Stego: {len(stego_meta)}")

    # Null probe sanity check: train per-layer null probes on honest-only activations
    # and ensemble their scores — should be near chance.
    print("\nNull probe sanity check...")
    null_groups_by_layer = []
    null_scalers_ws = []
    for layer in LAYERS:
        groups = build_groups(core_meta, core_npz, layer, phase="discussion", avg_rounds=True)
        null_groups_by_layer.append(groups)
        n = len(core_meta)
        acts = get_layer(core_npz, n, layer)
        hon_idx = [i for i, m in enumerate(core_meta)
                   if m["mode"] == "collusion" and m["channel"] == "public"
                   and m["phase"] == "discussion" and m["role"] == "honest"]
        rng = np.random.RandomState(42)
        rng.shuffle(hon_idx)
        half = len(hon_idx) // 2
        scaler = StandardScaler()
        X = scaler.fit_transform(acts[hon_idx[:half * 2]])
        y = np.array([0] * half + [1] * half)
        null_probe = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
        null_probe.fit(X, y)
        w_null = null_probe.coef_[0]
        w_null = w_null / np.linalg.norm(w_null)
        null_scalers_ws.append((scaler, w_null))
    rids  = sorted(null_groups_by_layer[0].keys())
    run_y = _run_labels(null_groups_by_layer[0], rids)
    if len(np.unique(run_y)) >= 2:
        null_scores = compute_scores_ensemble(null_groups_by_layer, rids, null_scalers_ws)
        print(f"  Null probe AUROC (ensemble): "
              f"{roc_auc_score(run_y, null_scores['peak_suspicion']):.2f}")

    print(f"\nComputing ensemble results ({len(LAYERS)} layers, {N_SEEDS} seeds)...")
    all_means = []
    group_names = None
    for seed in range(N_SEEDS):
        print(f"  Seed {seed}...")
        gn, means, _ = compute_all_results(
            core_meta, core_npz, transfer_meta, transfer_npz, stego_meta, stego_npz,
            seed=seed)
        if group_names is None:
            group_names = gn
        all_means.append(means)

    # Aggregate across seeds (SEM over seeds)
    final_means = {g: {} for g in group_names}
    final_stds  = {g: {} for g in group_names}
    for g in group_names:
        for m in METHOD_KEYS:
            vals = [seed_means[g][m] for seed_means in all_means]
            arr = np.array(vals, dtype=float)
            n = np.sum(~np.isnan(arr))
            final_means[g][m] = np.nanmean(arr)
            final_stds[g][m]  = np.nanstd(arr) / np.sqrt(n) if n > 0 else 0.0

    # Print table
    header = f"{'Group':>20}" + "".join(
        f"  {m.replace(chr(10), ' '):>15}" for m in METHOD_NAMES)
    print(f"\n{header}")
    print("-" * len(header))
    for g in group_names:
        row = f"{DISPLAY_NAMES.get(g, g).replace(chr(10), ' '):>20}"
        row += "".join(f"  {final_means[g][m]:>10.3f}+/-{final_stds[g][m]:.3f}" for m in METHOD_KEYS)
        print(row)

    # Load reference lines from a previous run
    CSV_DIR = config.RESULTS_ROOT / "auroc_scores"
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = CSV_DIR / f"{OUT_STEM}.csv"
    reference_means = None
    if REFERENCE_STEM is not None:
        ref_path = CSV_DIR / f"{REFERENCE_STEM}.csv"
        if ref_path.exists():
            reference_means = {}
            with open(ref_path, newline="") as ref_f:
                for row in csv.DictReader(ref_f):
                    if row["record_type"] == "mean":
                        g, m = row["group"], row["method"]
                        if g not in reference_means:
                            reference_means[g] = {}
                        reference_means[g][m] = float(row["auroc"])
            print(f"Loaded reference lines from {ref_path}")
        else:
            print(f"Reference CSV not found, skipping reference lines: {ref_path}")

    # Save CSV of AUROC scores
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        # multiLayer ensembles across all layers, so no per-layer breakdown
        writer.writerow(["record_type", "group", "method", "seed", "auroc"])
        for seed_idx, seed_means in enumerate(all_means):
            for g in group_names:
                for m in METHOD_KEYS:
                    writer.writerow(["raw", g, m, seed_idx, seed_means[g][m]])
        for g in group_names:
            for m in METHOD_KEYS:
                writer.writerow(["mean", g, m, "", final_means[g][m]])
        overall = np.nanmean([
            final_means[g][m]
            for g in group_names if g != "Held-out"
            for m in METHOD_KEYS
        ])
        writer.writerow(["overall_mean", "", "", "", overall])
    print(f"Saved CSV to {csv_path}")
    print(f"Overall mean AUROC (excl. Held-out): {overall:.3f}")

    # Save figure
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig = plot_figure(group_names, final_means, final_stds, reference_means=reference_means)
    #fig.savefig(OUT_DIR / f"{OUT_STEM}.pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / f"{OUT_STEM}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved to {OUT_DIR / OUT_STEM}.pdf")


if __name__ == "__main__":
    main()
