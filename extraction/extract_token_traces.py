"""
Extract per-token deception traces from Qwen3-32B-AWQ.

For each agent utterance in a conversation, projects every response token's
hidden-state activation onto the deception direction w (learned from the
colluder-detection probe). Produces a "deception trace" showing how the
deception signal evolves token-by-token across conversation phases.

Hypothesis: colluder deception scores spike during public deception and stay
low during private (truthful) coordination. Honest agents stay flat throughout.

Usage:
    python extract_token_traces.py                    # default: 10 collusion runs
    python extract_token_traces.py --max-runs 5       # fewer runs
    python extract_token_traces.py --test              # 2 runs, quick sanity check
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

TARGET_LAYERS = [26, 27, 28, 29, 30]  # sweet spot from probe results
PRIMARY_LAYER = 28  # best honest_context AUROC

# Requires pre-extracted Core activations and raw scenario outputs
PROBE_DATA_DIR = config.ACTIVATIONS_ROOT / "qwen3_32b" / "core"
RUNS_DIR = config.RAW_ROOT / "core"
OUTPUT_DIR = config.ACTIVATIONS_ROOT / "qwen3_32b" / "token_traces"


# ---------------------------------------------------------------------------
# Step 1: Train probe and extract deception direction w
# ---------------------------------------------------------------------------

def get_deception_directions(layers: list[int] = TARGET_LAYERS) -> dict[int, tuple[np.ndarray, StandardScaler]]:
    """Train colluder-detection probe at each layer, return {layer: (w, scaler)}."""
    print(f"Training probes on layers {layers} to get deception directions...")

    with open(PROBE_DATA_DIR / "metadata_gen.json") as f:
        metadata = json.load(f)
    npz = np.load(PROBE_DATA_DIR / "activations_gen.npz")

    # Colluder public vs honest public (final phase only, matches main probe)
    idx_pos, idx_neg = [], []
    for i, m in enumerate(metadata):
        if m["phase"] != "final" or m["channel"] != "public":
            continue
        if m["role"] == "colluder":
            idx_pos.append(i)
        elif m["role"] == "honest":
            idx_neg.append(i)

    # Balance classes
    n = min(len(idx_pos), len(idx_neg))
    rng = np.random.RandomState(42)
    idx_neg = list(rng.choice(idx_neg, n, replace=False))
    idx_pos = idx_pos[:n]

    indices = np.array(idx_neg + idx_pos)
    labels = np.array([0] * n + [1] * n)

    directions = {}
    for layer in layers:
        X = npz[f"layer_{layer}"][indices]
        scaler = StandardScaler()
        X_s = scaler.fit_transform(X)

        clf = LogisticRegression(max_iter=1000, C=1.0)
        clf.fit(X_s, labels)

        w = clf.coef_[0]  # shape: (5120,)
        directions[layer] = (w, scaler)
        print(f"  Layer {layer}: acc = {clf.score(X_s, labels):.3f}, ||w|| = {np.linalg.norm(w):.4f}")

    return directions


# ---------------------------------------------------------------------------
# Step 2: Load conversation data
# ---------------------------------------------------------------------------

def load_runs(runs_dir: Path, max_runs: int = None, collusion_only: bool = False) -> list[dict]:
    """Load run directories. Returns list of run dicts with config, prompts, results."""
    run_dirs = sorted([d for d in runs_dir.iterdir() if d.is_dir()])

    if collusion_only:
        run_dirs = [d for d in run_dirs if d.name.endswith("__collusion")]

    if max_runs:
        run_dirs = run_dirs[:max_runs]

    runs = []
    for run_dir in run_dirs:
        config_path = run_dir / "run_config.json"
        prompts_path = run_dir / "agent_prompts.json"
        results_path = run_dir / "results.json"

        if not all(p.exists() for p in [config_path, prompts_path, results_path]):
            continue

        with open(config_path) as f:
            config = json.load(f)
        with open(prompts_path) as f:
            prompts = json.load(f)
        with open(results_path) as f:
            results = json.load(f)

        runs.append({"config": config, "prompts": prompts, "results": results, "dir": run_dir})

    print(f"Loaded {len(runs)} runs from {runs_dir}")
    return runs


def build_utterances(run: dict) -> list[dict]:
    """Build list of utterances with system_prompt, user_prompt, response, and metadata."""
    config = run["config"]
    prompts = run["prompts"]
    results = run["results"]

    run_id = config["run_id"]
    mode = config["mode"]
    roles = config.get("effective_roles", config["roles"])
    orig_roles = config["roles"]

    # Parse responses from results
    private_msgs = results.get("private_messages", [])
    public_msgs = results.get("public_messages", [])
    final_texts = results.get("final_texts", {})

    private_responses = [re.sub(r"^\[[^\]]+\]:\s*", "", msg) for msg in private_msgs]
    public_responses = [re.sub(r"^\[[^\]]+\]:\s*", "", msg) for msg in public_msgs]

    private_idx = 0
    public_idx = 0
    utterances = []

    for prompt_entry in prompts:
        phase = prompt_entry["phase"]
        channel = prompt_entry["channel"]
        agent_name = prompt_entry["agent_name"]

        if channel == "private" and phase in ("discussion", "private"):
            if private_idx < len(private_responses):
                response = private_responses[private_idx]
                private_idx += 1
            else:
                response = ""
        elif phase == "discussion" and channel == "public":
            if public_idx < len(public_responses):
                response = public_responses[public_idx]
                public_idx += 1
            else:
                response = ""
        elif phase == "final":
            response = final_texts.get(agent_name, "")
        else:
            response = ""

        if not response:
            continue

        role = (orig_roles if channel == "private" else roles).get(
            agent_name, prompt_entry.get("role", "unknown")
        )

        utterances.append({
            "run_id": run_id,
            "mode": mode,
            "agent_name": agent_name,
            "role": role,
            "phase": phase,
            "channel": channel,
            "round": prompt_entry["round"],
            "system_prompt": prompt_entry["system_prompt"],
            "user_prompt": prompt_entry["user_prompt"],
            "response": response,
        })

    return utterances


# ---------------------------------------------------------------------------
# Step 3: Per-token activation extraction
# ---------------------------------------------------------------------------

class TokenActivationHooks:
    """Capture ALL token hidden states (not just last) at target layers."""

    def __init__(self, model, target_layers: list[int]):
        self.captured = {}
        self.hooks = []
        for layer_idx in target_layers:
            hook = model.model.layers[layer_idx].register_forward_hook(
                self._make_hook(layer_idx)
            )
            self.hooks.append(hook)

    def _make_hook(self, layer_idx: int):
        def hook_fn(module, input, output):
            hidden = output[0] if isinstance(output, tuple) else output
            # ALL tokens: shape (seq_len, hidden_dim)
            self.captured[layer_idx] = hidden[0, :, :].float().cpu()
        return hook_fn

    def get_and_clear(self) -> dict[int, torch.Tensor]:
        result = dict(self.captured)
        self.captured.clear()
        return result

    def remove(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()


def extract_traces_for_utterance(
    model, tokenizer, hooks: TokenActivationHooks,
    utterance: dict, directions: dict[int, tuple[np.ndarray, StandardScaler]],
    layers: list[int],
) -> dict:
    """Extract per-token deception trace for one utterance at all target layers.

    Returns dict with token texts, per-layer deception scores, and metadata.
    """
    # Build full text: system + user + assistant response
    messages = [
        {"role": "system", "content": utterance["system_prompt"]},
        {"role": "user", "content": utterance["user_prompt"]},
        {"role": "assistant", "content": utterance["response"]},
    ]
    full_text = tokenizer.apply_chat_template(messages, tokenize=False)

    # Build prompt-only text to find where response starts
    prompt_messages = [
        {"role": "system", "content": utterance["system_prompt"]},
        {"role": "user", "content": utterance["user_prompt"]},
    ]
    prompt_text = tokenizer.apply_chat_template(
        prompt_messages, tokenize=False, add_generation_prompt=True
    )

    # Tokenize
    full_ids = tokenizer(full_text, return_tensors="pt")["input_ids"]
    prompt_ids = tokenizer(prompt_text, return_tensors="pt")["input_ids"]

    prompt_len = prompt_ids.shape[1]
    total_len = full_ids.shape[1]
    response_len = total_len - prompt_len

    if response_len <= 0:
        return None

    # Forward pass (hooks capture all layers simultaneously)
    with torch.no_grad():
        model(full_ids.to(model.device))

    captured = hooks.get_and_clear()

    # Project onto deception direction at each layer
    scores_by_layer = {}
    for layer in layers:
        hidden = captured[layer].numpy()  # (total_len, hidden_dim)
        response_hidden = hidden[prompt_len:]  # (response_len, hidden_dim)

        w, scaler = directions[layer]
        response_scaled = scaler.transform(response_hidden)
        scores_by_layer[str(layer)] = (response_scaled @ w).tolist()

    # Decode response tokens individually for annotation
    response_token_ids = full_ids[0, prompt_len:].tolist()
    token_texts = [tokenizer.decode([tid]) for tid in response_token_ids]

    return {
        "deception_scores": scores_by_layer,  # {layer_str: [scores]}
        "token_texts": token_texts,
        "token_ids": response_token_ids,
        "prompt_len": prompt_len,
        "response_len": response_len,
        "total_len": total_len,
        # Metadata
        "run_id": utterance["run_id"],
        "mode": utterance["mode"],
        "agent_name": utterance["agent_name"],
        "role": utterance["role"],
        "phase": utterance["phase"],
        "channel": utterance["channel"],
        "round": utterance["round"],
        "response_text": utterance["response"],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Extract per-token deception traces")
    parser.add_argument("--max-runs", type=int, default=10,
                        help="Max number of collusion runs to process (default: 10)")
    parser.add_argument("--model", type=str, default=config.DEFAULT_MODEL,
                        help="HuggingFace model ID (default: config.DEFAULT_MODEL)")
    parser.add_argument("--include-control", action="store_true",
                        help="Also process control runs (no collusion)")
    parser.add_argument("--test", action="store_true",
                        help="Quick test: 2 runs only")
    args = parser.parse_args()

    if args.test:
        args.max_runs = 2

    print("=" * 60)
    print("Token-Level Deception Trace Extraction")
    print("=" * 60)
    print(f"Layers:    {TARGET_LAYERS}")
    print(f"Max runs:  {args.max_runs}")
    print(f"Control:   {args.include_control}")
    print()

    # Step 1: Get deception directions for all target layers
    directions = get_deception_directions(TARGET_LAYERS)

    # Step 2: Load runs
    runs = load_runs(
        RUNS_DIR,
        max_runs=args.max_runs * 2 if args.include_control else None,
        collusion_only=not args.include_control,
    )

    # If not including control, limit to max_runs collusion runs
    if not args.include_control:
        runs = runs[:args.max_runs]
    else:
        # Take max_runs collusion + max_runs control
        collusion_runs = [r for r in runs if r["config"]["mode"] == "collusion"][:args.max_runs]
        control_runs = [r for r in runs if r["config"]["mode"] == "control"][:args.max_runs]
        runs = collusion_runs + control_runs
        print(f"  {len(collusion_runs)} collusion + {len(control_runs)} control runs")

    # Step 3: Load model
    model_name = args.model
    print(f"\nLoading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map="auto", trust_remote_code=True,
    )
    model.eval()
    print("Model loaded.")

    # Step 4: Extract traces
    hooks = TokenActivationHooks(model, TARGET_LAYERS)
    all_traces = []

    try:
        for run in tqdm(runs, desc="Runs"):
            utterances = build_utterances(run)
            run_id = run["config"]["run_id"]
            print(f"\n  {run_id}: {len(utterances)} utterances")

            for utt in tqdm(utterances, desc=f"  {run_id}", leave=False):
                trace = extract_traces_for_utterance(
                    model, tokenizer, hooks, utt, directions, TARGET_LAYERS
                )
                if trace is not None:
                    all_traces.append(trace)
    finally:
        hooks.remove()

    # Step 5: Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "traces.json"
    with open(out_path, "w") as f:
        json.dump(all_traces, f)
    print(f"\nSaved {len(all_traces)} traces to {out_path}")

    # Print summary stats
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    by_key = {}
    pl = str(PRIMARY_LAYER)
    for t in all_traces:
        key = f"{t['role']}/{t['channel']}"
        if key not in by_key:
            by_key[key] = {"count": 0, "mean_scores": []}
        by_key[key]["count"] += 1
        by_key[key]["mean_scores"].append(np.mean(t["deception_scores"][pl]))

    print(f"{'Category':<25} {'Count':<8} {'Mean Score':<12} {'Std'}")
    print("-" * 55)
    for key in sorted(by_key):
        scores = by_key[key]["mean_scores"]
        print(f"{key:<25} {by_key[key]['count']:<8} {np.mean(scores):<12.4f} {np.std(scores):.4f}")


if __name__ == "__main__":
    main()
