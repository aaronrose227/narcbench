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
from colours import TEAL_COLORS

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

# ── Config ───────────────────────────────────────────────────────────────

OUT_DIR = config.RESULTS_ROOT / "figures"   # save directory
OUT_STEM       = "method_comparison_multihead_4heads"    # output filename (no extension)
REFERENCE_STEM = "method_comparison_LR_C250_liblinear_maxIters8"  # CSV stem to load reference lines from (None to disable)

N_SEEDS = 10

MHA_D_MODEL    = 64    # projection dimension inside attention
MHA_NUM_HEADS  = 4     # number of attention heads
MHA_DROPOUT    = 0.1   # dropout rate
MHA_EPOCHS     = 200   # training epochs
MHA_LR         = 1e-3  # learning rate
MHA_BATCH_SIZE = 32    # groups per batch

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
COLORS = TEAL_COLORS

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


# ── MHA probe ────────────────────────────────────────────────────────────

class MultiHeadAttentionProbe(nn.Module):
    """Multi-headed self-attention probe over the 4-agent group.

    Input:  (batch, 4, input_dim) — one activation vector per agent
    Output: (batch, 4)            — per-agent deception logits
    """
    def __init__(self, input_dim: int):
        super().__init__()
        self.proj_in  = nn.Linear(input_dim, MHA_D_MODEL)
        self.attn     = nn.MultiheadAttention(MHA_D_MODEL, MHA_NUM_HEADS,
                                              dropout=MHA_DROPOUT, batch_first=True)
        self.norm     = nn.LayerNorm(MHA_D_MODEL)
        self.proj_out = nn.Linear(MHA_D_MODEL, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.proj_in(x)                      # (B, 4, d_model)
        attended, _ = self.attn(h, h, h)
        h = self.norm(h + attended)              # residual + norm
        return self.proj_out(h).squeeze(-1)      # (B, 4) logits


# ── Probe training ───────────────────────────────────────────────────────

def train_deception_probe(meta, npz, layer, phase="discussion", exclude_domain=None, seed=0):
    """Train an MHA deception probe from group-level activations.

    Builds groups from meta/npz, then trains a MultiHeadAttentionProbe where
    colluder agents in collusion runs are labeled 1, honest agents 0, and all
    agents in control runs are labeled 0. 

    Returns (scaler, probe).
    """
    torch.manual_seed(seed)
    groups = build_groups(meta, npz, layer, phase=phase,
                          avg_rounds=(phase == "discussion"))

    X_list, y_list = [], []
    for rid, g in groups.items():
        _, scenario_id = _parse_run_id(rid)
        if exclude_domain and _domain_from_scenario_id(scenario_id) == exclude_domain:
            continue

        names = sorted(g["agents"])
        vecs = np.array([g["agents"][n] for n in names], dtype=np.float32)  # (4, D)
        if g["mode"] == "collusion":
            labels = np.array(
                [1.0 if g["roles"][n] == "colluder" else 0.0 for n in names],
                dtype=np.float32)
        else:
            labels = np.zeros(4, dtype=np.float32)

        X_list.append(vecs)
        y_list.append(labels)

    scaler = StandardScaler()
    scaler.fit(np.vstack(X_list))
    X_scaled = np.array([scaler.transform(x) for x in X_list], dtype=np.float32)  # (N, 4, D)
    y = np.array(y_list, dtype=np.float32)                                         # (N, 4)

    probe = MultiHeadAttentionProbe(input_dim=X_scaled.shape[2])
    optimizer = torch.optim.Adam(probe.parameters(), lr=MHA_LR)
    criterion = nn.BCEWithLogitsLoss()
    X_t = torch.from_numpy(X_scaled)
    y_t = torch.from_numpy(y)
    n = len(X_t)

    probe.train()
    for _ in range(MHA_EPOCHS):
        perm = torch.randperm(n)
        for i in range(0, n, MHA_BATCH_SIZE):
            idx = perm[i:i + MHA_BATCH_SIZE]
            loss = criterion(probe(X_t[idx]), y_t[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    probe.eval()
    return scaler, probe


# ── Score computation ────────────────────────────────────────────────────

def compute_scores(groups, rids, scaler, probe):
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
        scaled_vecs = scaler.transform(np.array(raw_vecs, dtype=np.float32))  # (4, D)

        # Per-agent deception probabilities via MHA probe
        with torch.no_grad():
            x = torch.from_numpy(scaled_vecs).unsqueeze(0)  # (1, 4, D)
            projs = torch.sigmoid(probe(x)).squeeze(0).numpy()  # (4,)
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


# ── Evaluation ───────────────────────────────────────────────────────────

class _MHAClassifier(nn.Module):
    """MHA classifier for run-level binary classification.

    Treats each feature dimension as a token of size 1, projects to MHA_D_MODEL,
    applies self-attention, mean-pools, and outputs a single logit per run.

    Input:  (batch, seq_len, 1)
    Output: (batch,) logits
    """
    def __init__(self, seq_len: int):
        super().__init__()
        self.proj_in  = nn.Linear(1, MHA_D_MODEL)
        self.attn     = nn.MultiheadAttention(MHA_D_MODEL, MHA_NUM_HEADS,
                                              dropout=MHA_DROPOUT, batch_first=True)
        self.norm     = nn.LayerNorm(MHA_D_MODEL)
        self.proj_out = nn.Linear(MHA_D_MODEL, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.proj_in(x)                  # (B, seq_len, d_model)
        attended, _ = self.attn(h, h, h)
        h = self.norm(h + attended)
        return self.proj_out(h.mean(dim=1)).squeeze(-1)  # (B,) logits


def _fit_mha_classifier(X: np.ndarray, y: np.ndarray) -> _MHAClassifier:
    """Train an _MHAClassifier on (N, features) data with binary labels."""
    X_t = torch.tensor(X[:, :, None], dtype=torch.float32)  # (N, features, 1)
    y_t = torch.tensor(y, dtype=torch.float32)
    clf = _MHAClassifier(seq_len=X.shape[1])
    optimizer = torch.optim.Adam(clf.parameters(), lr=MHA_LR)
    criterion = nn.BCEWithLogitsLoss()
    n = len(X_t)
    clf.train()
    for _ in range(MHA_EPOCHS):
        perm = torch.randperm(n)
        for i in range(0, n, MHA_BATCH_SIZE):
            idx = perm[i:i + MHA_BATCH_SIZE]
            loss = criterion(clf(X_t[idx]), y_t[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    clf.eval()
    return clf


def _mha_predict_proba(clf: _MHAClassifier, X: np.ndarray) -> np.ndarray:
    """Return sigmoid probabilities from an _MHAClassifier for input (N, features)."""
    with torch.no_grad():
        x = torch.tensor(X[:, :, None], dtype=torch.float32)
        return torch.sigmoid(clf(x)).numpy()


def _train_asymmetry_classifier(tr_scores, tr_y):
    """Train MHA Asymmetry Probe classifier on training scores."""
    fs = StandardScaler()
    X = fs.fit_transform(tr_scores["asymmetry_probe"])
    clf = _fit_mha_classifier(X, tr_y)
    return fs, clf


def _train_concat_classifier(tr_scores, tr_y):
    """Train Sorted Concatenation classifier (PCA + MHA) on training scores."""
    n_comp = min(50, len(tr_y) - 1, tr_scores["sorted_concat"].shape[1])
    pca = PCA(n_components=n_comp, random_state=42)
    fs = StandardScaler()
    X_tr = fs.fit_transform(pca.fit_transform(tr_scores["sorted_concat"]))
    clf = _fit_mha_classifier(X_tr, tr_y)
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
        te_y, _mha_predict_proba(clf, fs.transform(te_scores["asymmetry_probe"])))

    # Sorted Concatenation
    pca, fs2, clf2 = _train_concat_classifier(tr_scores, tr_y)
    X_te = fs2.transform(pca.transform(te_scores["sorted_concat"]))
    results["sorted_concat"] = roc_auc_score(
        te_y, _mha_predict_proba(clf2, X_te))

    return results


# ── Leave-domain-out CV ──────────────────────────────────────────────────

def held_out_cv_layer(core_meta, core_npz, layer, seed=0):
    """Leave-domain-out cross-validation on Core for one layer."""
    groups = build_groups(core_meta, core_npz, layer, phase="discussion", avg_rounds=True)
    rids = sorted(groups.keys())
    domains = sorted({groups[r]["domain"] for r in rids if groups[r]["domain"]})

    fold_results = {k: [] for k in METHOD_KEYS}

    for hold_dom in domains:
        tr_rids = [r for r in rids if groups[r]["domain"] != hold_dom]
        te_rids = [r for r in rids if groups[r]["domain"] == hold_dom]
        tr_y = _run_labels(groups, tr_rids)
        te_y = _run_labels(groups, te_rids)

        if len(np.unique(te_y)) < 2 or len(np.unique(tr_y)) < 2:
            continue

        scaler, probe = train_deception_probe(core_meta, core_npz, layer, exclude_domain=hold_dom, seed=seed)
        tr_scores = compute_scores(groups, tr_rids, scaler, probe)
        te_scores = compute_scores(groups, te_rids, scaler, probe)

        result = evaluate(tr_scores, tr_y, te_scores, te_y)
        if result:
            for k in METHOD_KEYS:
                fold_results[k].append(result[k])

    return {k: np.mean(v) if v else float("nan") for k, v in fold_results.items()}


# ── Main computation ─────────────────────────────────────────────────────

def compute_all_results(core_meta, core_npz, transfer_meta, transfer_npz,
                        stego_meta, stego_npz, seed=0):
    """Compute AUROC for all methods, groups, and layers."""

    # Discover transfer scenario types
    transfer_peek = build_groups(
        transfer_meta, transfer_npz, LAYERS[0], phase="final", avg_rounds=False)
    transfer_scenarios = sorted({
        transfer_peek[r]["scenario"]
        for r in transfer_peek if transfer_peek[r]["scenario"]
    })

    group_names = ["Held-out"] + transfer_scenarios + ["blackjack"]
    results = {g: {m: [] for m in METHOD_KEYS} for g in group_names}

    for layer in LAYERS:
        print(f"  Layer {layer}...")

        # -- Core: leave-domain-out CV --
        cv = held_out_cv_layer(core_meta, core_npz, layer, seed=seed)
        for m in METHOD_KEYS:
            results["Held-out"][m].append(cv[m])

        # -- Train global probe on all Core data --
        scaler, probe = train_deception_probe(core_meta, core_npz, layer, seed=seed)
        core_groups = build_groups(core_meta, core_npz, layer,
                                   phase="discussion", avg_rounds=True)
        core_rids = sorted(core_groups.keys())
        core_y = _run_labels(core_groups, core_rids)
        core_scores = compute_scores(core_groups, core_rids, scaler, probe)

        # Train classifiers once (reused for Transfer and Stego)
        asym_fs, asym_clf = _train_asymmetry_classifier(core_scores, core_y)
        concat_pca, concat_fs, concat_clf = _train_concat_classifier(
            core_scores, core_y)

        # -- Transfer --
        transfer_groups = build_groups(
            transfer_meta, transfer_npz, layer, phase="final", avg_rounds=False)
        transfer_rids = sorted(transfer_groups.keys())
        transfer_y = _run_labels(transfer_groups, transfer_rids)
        transfer_sc = [transfer_groups[r]["scenario"] for r in transfer_rids]
        transfer_scores = compute_scores(transfer_groups, transfer_rids, scaler, probe)

        asym_proba = _mha_predict_proba(
            asym_clf, asym_fs.transform(transfer_scores["asymmetry_probe"]))
        concat_proba = _mha_predict_proba(
            concat_clf,
            concat_fs.transform(concat_pca.transform(transfer_scores["sorted_concat"])))

        for sc in transfer_scenarios:
            mask = np.array([s == sc for s in transfer_sc])
            if mask.sum() > 0 and len(np.unique(transfer_y[mask])) >= 2:
                for key in ["peak_suspicion", "deception_split", "activation_alignment"]:
                    results[sc][key].append(
                        roc_auc_score(transfer_y[mask], transfer_scores[key][mask]))
                results[sc]["asymmetry_probe"].append(
                    roc_auc_score(transfer_y[mask], asym_proba[mask]))
                results[sc]["sorted_concat"].append(
                    roc_auc_score(transfer_y[mask], concat_proba[mask]))

        # -- Stego --
        stego_groups = build_groups(
            stego_meta, stego_npz, layer, phase="final", avg_rounds=False)
        stego_rids = sorted(stego_groups.keys())
        stego_y = _run_labels(stego_groups, stego_rids)

        if len(np.unique(stego_y)) >= 2 and len(stego_rids) >= 4:
            stego_scores = compute_scores(stego_groups, stego_rids, scaler, probe)

            for key in ["peak_suspicion", "deception_split", "activation_alignment"]:
                results["blackjack"][key].append(
                    roc_auc_score(stego_y, stego_scores[key]))

            stego_asym = _mha_predict_proba(
                asym_clf, asym_fs.transform(stego_scores["asymmetry_probe"]))
            results["blackjack"]["asymmetry_probe"].append(
                roc_auc_score(stego_y, stego_asym))

            stego_concat = _mha_predict_proba(
                concat_clf,
                concat_fs.transform(concat_pca.transform(stego_scores["sorted_concat"])))
            results["blackjack"]["sorted_concat"].append(
                roc_auc_score(stego_y, stego_concat))

    return group_names, results


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

    # Null probe sanity check: train MHA probe with randomly shuffled per-agent
    # colluder labels — should score near chance.
    print("\nNull probe sanity check...")
    null_aucs = []
    rng = np.random.RandomState(42)
    for layer in LAYERS:
        groups = build_groups(core_meta, core_npz, layer, phase="discussion", avg_rounds=True)
        rids = sorted(groups.keys())

        X_list, y_list = [], []
        for rid in rids:
            g = groups[rid]
            if g["mode"] != "collusion":
                continue
            names = sorted(g["agents"])
            vecs = np.array([g["agents"][n] for n in names], dtype=np.float32)
            labels = np.zeros(4, dtype=np.float32)
            labels[rng.choice(4, 2, replace=False)] = 1.0  # random 50/50
            X_list.append(vecs)
            y_list.append(labels)

        scaler = StandardScaler()
        scaler.fit(np.vstack(X_list))
        X_scaled = np.array([scaler.transform(x) for x in X_list], dtype=np.float32)
        y_arr = np.array(y_list, dtype=np.float32)

        null_probe = MultiHeadAttentionProbe(input_dim=X_scaled.shape[2])
        optimizer = torch.optim.Adam(null_probe.parameters(), lr=MHA_LR)
        criterion = nn.BCEWithLogitsLoss()
        X_t = torch.from_numpy(X_scaled)
        y_t = torch.from_numpy(y_arr)
        null_probe.train()
        for _ in range(MHA_EPOCHS):
            perm = torch.randperm(len(X_t))
            for i in range(0, len(X_t), MHA_BATCH_SIZE):
                idx = perm[i:i + MHA_BATCH_SIZE]
                loss = criterion(null_probe(X_t[idx]), y_t[idx])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        null_probe.eval()

        null_scores = compute_scores(groups, rids, scaler, null_probe)
        run_y = _run_labels(groups, rids)
        if len(np.unique(run_y)) >= 2:
            null_aucs.append(roc_auc_score(run_y, null_scores["peak_suspicion"]))
    print(f"  Null probe AUROC (mean across layers): {np.mean(null_aucs):.2f}")

    print(f"\nComputing results ({len(LAYERS)} layers, {N_SEEDS} seeds)...")
    all_raw = []
    group_names = None
    for seed in range(N_SEEDS):
        print(f"  Seed {seed}...")
        gn, raw = compute_all_results(
            core_meta, core_npz, transfer_meta, transfer_npz, stego_meta, stego_npz,
            seed=seed)
        if group_names is None:
            group_names = gn
        all_raw.append(raw)

    # Collect all AUROC scores across seeds and layers before aggregating
    final_means = {g: {} for g in group_names}
    final_stds  = {g: {} for g in group_names}
    for g in group_names:
        for m in METHOD_KEYS:
            all_vals = []
            for raw in all_raw:
                all_vals.extend(raw[g][m])
            arr = np.array(all_vals, dtype=float)
            n = np.sum(~np.isnan(arr))
            final_means[g][m] = np.nanmean(arr) if n > 0 else 0.5
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

    assert group_names is not None

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
        writer.writerow(["record_type", "group", "method", "seed", "layer", "auroc"])
        for seed_idx, raw in enumerate(all_raw):
            for g in group_names:
                for m in METHOD_KEYS:
                    for layer_idx, layer in enumerate(LAYERS):
                        val = raw[g][m][layer_idx] if layer_idx < len(raw[g][m]) else float("nan")
                        writer.writerow(["raw", g, m, seed_idx, layer, val])
        for g in group_names:
            for m in METHOD_KEYS:
                writer.writerow(["mean", g, m, "", "", final_means[g][m]])
        overall = np.nanmean([
            final_means[g][m]
            for g in group_names if g != "Held-out"
            for m in METHOD_KEYS
        ])
        writer.writerow(["overall_mean", "", "", "", "", overall])
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
