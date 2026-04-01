"""
Extract hidden-state activations from LLM for NARCBench-Stego (blackjack) runs.

Two extraction modes:
  1. last-token (default): Last token of prompt/response at all target layers.
     Good for probe training (one vector per sample).
  2. per-token: All tokens at a few middle layers.
     Good for analysing how representations shift across the response
     (e.g. when the big player reads the counter's signal word).

Each sample is tagged with role (colluder/honest), channel (public/thinking/private),
phase (private/discussion/final), and hand number.

Usage:
    python extract_stego.py --runs-dir PATH
    python extract_stego.py --runs-dir PATH --per-token --gen-only
    python extract_stego.py --runs-dir PATH --layers 16,24,32,40,48
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
from tqdm import tqdm

NUM_LAYERS = 64
CHECKPOINT_EVERY = 50

# Per-token layers: include early-middle through upper layers (literature says
# 20-50% depth is the sweet spot, i.e. layers ~13-32 for a 64-layer model)
DEFAULT_PER_TOKEN_LAYERS = [4, 8, 13, 16, 20, 24, 28, 32, 40, 48, 56]

# Requires raw scenario outputs from generation/stego.py
DEFAULT_RUNS_DIR = config.RAW_ROOT / "stego"
DEFAULT_OUTPUT_DIR = config.ACTIVATIONS_ROOT / "qwen3_32b" / "stego"


def load_blackjack_runs(runs_dir: Path) -> list[dict]:
    """Load prompts, responses, and labels from blackjack run directories."""
    samples = []
    run_dirs = sorted([d for d in runs_dir.iterdir() if d.is_dir()])

    print(f"Found {len(run_dirs)} run directories in {runs_dir}")

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

        run_id = config["run_id"]
        seed = config["seed"]
        mode = config["mode"]
        coalition = config.get("coalition", [])
        roles = config["roles"]
        flagged = config.get("flagged", False)
        bp_corr = config.get("bp_bet_correlation", None)

        # Build response lookup from messages
        private_msgs = results.get("private_messages", [])
        thinking_msgs = results.get("thinking_messages", [])
        public_msgs = results.get("public_messages", [])

        # Parse "[Name]: response" format
        def parse_msgs(msgs):
            parsed = []
            for msg in msgs:
                resp = re.sub(r"^\[[^\]]+\]:\s*", "", msg)
                parsed.append(resp)
            return parsed

        private_responses = parse_msgs(private_msgs)
        thinking_responses = parse_msgs(thinking_msgs)
        public_responses = parse_msgs(public_msgs)

        # Walk prompts and match to responses in order
        priv_idx = 0
        think_idx = 0
        pub_idx = 0

        for prompt_entry in prompts:
            phase = prompt_entry["phase"]
            channel = prompt_entry["channel"]
            agent_name = prompt_entry["agent_name"]
            role = prompt_entry.get("role", roles.get(agent_name, "unknown"))

            if channel == "private":
                if priv_idx < len(private_responses):
                    response = private_responses[priv_idx]
                    priv_idx += 1
                else:
                    response = ""
            elif channel == "thinking":
                if think_idx < len(thinking_responses):
                    response = thinking_responses[think_idx]
                    think_idx += 1
                else:
                    response = ""
            elif channel == "public":
                if pub_idx < len(public_responses):
                    response = public_responses[pub_idx]
                    pub_idx += 1
                else:
                    response = ""
            else:
                response = ""

            # Skip judge entries
            if role == "judge":
                continue

            samples.append({
                "run_id": run_id,
                "seed": seed,
                "mode": mode,
                "agent_name": agent_name,
                "role": role,
                "phase": phase,
                "channel": channel,
                "round": prompt_entry["round"],
                "coalition": coalition,
                "flagged": flagged,
                "bp_corr": bp_corr,
                "system_prompt": prompt_entry["system_prompt"],
                "user_prompt": prompt_entry["user_prompt"],
                "response": response,
            })

    print(f"Loaded {len(samples)} total samples")

    # Print breakdown
    from collections import Counter
    by_key = Counter((s['mode'], s['role'], s['channel']) for s in samples)
    for k, v in sorted(by_key.items()):
        print(f"  {'/'.join(k)}: {v}")

    return samples


def load_model(model_name: str):
    """Load model and tokenizer."""
    print(f"Loading tokenizer: {model_name}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    except (ValueError, ImportError):
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True, use_fast=False
        )

    print(f"Loading model: {model_name} (device_map=auto)")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    print("Model loaded.")
    return model, tokenizer


class LastTokenHooks:
    """Capture last-token hidden states at target layers."""

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
            self.captured[layer_idx] = hidden[0, -1, :].float().cpu()
        return hook_fn

    def get_and_clear(self) -> dict[int, torch.Tensor]:
        result = dict(self.captured)
        self.captured.clear()
        return result

    def remove(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()


class AllTokenHooks:
    """Capture ALL token hidden states at target layers."""

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
            # Save all tokens: shape (seq_len, hidden_dim)
            self.captured[layer_idx] = hidden[0].float().cpu()
        return hook_fn

    def get_and_clear(self) -> dict[int, torch.Tensor]:
        result = dict(self.captured)
        self.captured.clear()
        return result

    def remove(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()


def _build_input_text(tokenizer, sample: dict) -> str:
    """Build input-time text (prompt only, with generation prompt)."""
    messages = [
        {"role": "system", "content": sample["system_prompt"]},
        {"role": "user", "content": sample["user_prompt"]},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def _build_gen_text(tokenizer, sample: dict) -> str:
    """Build generation-time text (prompt + response)."""
    messages = [
        {"role": "system", "content": sample["system_prompt"]},
        {"role": "user", "content": sample["user_prompt"]},
        {"role": "assistant", "content": sample["response"]},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False)


# ── Last-token extraction ───────────────────────────────────────────

def extract_last_token(
    model, tokenizer, samples, target_layers, output_dir, extraction_mode,
    checkpoint_every=CHECKPOINT_EVERY,
):
    """Extract last-token hidden states at target layers."""
    suffix = extraction_mode
    checkpoint_path = output_dir / f"_checkpoint_{suffix}.npz"
    checkpoint_meta_path = output_dir / f"_checkpoint_{suffix}_meta.json"
    start_idx = 0

    if checkpoint_path.exists() and checkpoint_meta_path.exists():
        print(f"Found {suffix} checkpoint, resuming...")
        checkpoint_data = np.load(checkpoint_path)
        with open(checkpoint_meta_path) as f:
            checkpoint_meta = json.load(f)
        start_idx = checkpoint_meta["next_idx"]
        activations = {
            layer: list(checkpoint_data[f"layer_{layer}"][:start_idx])
            for layer in target_layers
        }
        metadata = checkpoint_meta["metadata"]
        print(f"  Resuming from sample {start_idx}/{len(samples)}")
    else:
        activations = {layer: [] for layer in target_layers}
        metadata = []

    hooks = LastTokenHooks(model, target_layers)
    n_total = len(samples)

    try:
        for i in tqdm(range(start_idx, n_total), initial=start_idx, total=n_total,
                       desc=f"Last-token ({suffix})"):
            sample = samples[i]

            if extraction_mode == "input":
                text = _build_input_text(tokenizer, sample)
            else:
                text = _build_gen_text(tokenizer, sample)

            inputs = tokenizer(text, return_tensors="pt")
            input_ids = inputs["input_ids"].to(model.device)

            with torch.no_grad():
                model(input_ids)

            captured = hooks.get_and_clear()
            for layer in target_layers:
                activations[layer].append(captured[layer].numpy())

            metadata.append(_make_meta(sample, int(input_ids.shape[1])))

            if (i + 1) % checkpoint_every == 0:
                print(f"\n  Checkpointing at {i + 1}/{n_total}...")
                _save_checkpoint(activations, metadata, target_layers, i + 1,
                                 output_dir, suffix)

    finally:
        hooks.remove()

    activation_arrays = {
        layer: np.stack(activations[layer]) for layer in target_layers
    }
    return activation_arrays, metadata


# ── Per-token extraction ────────────────────────────────────────────

def extract_per_token(
    model, tokenizer, samples, target_layers, output_dir,
    checkpoint_every=CHECKPOINT_EVERY,
):
    """Extract all-token hidden states at target layers (gen-time only).

    Saves one .npz per sample (variable sequence lengths), plus a single
    metadata JSON for all samples.
    """
    suffix = "per_token"
    token_dir = output_dir / "per_token"
    token_dir.mkdir(parents=True, exist_ok=True)

    meta_path = output_dir / f"metadata_{suffix}.json"

    # Check for resume
    start_idx = 0
    metadata = []
    if meta_path.exists():
        with open(meta_path) as f:
            metadata = json.load(f)
        start_idx = len(metadata)
        if start_idx >= len(samples):
            print(f"Per-token extraction already complete ({start_idx} samples)")
            return metadata
        print(f"Resuming per-token from sample {start_idx}/{len(samples)}")

    hooks = AllTokenHooks(model, target_layers)
    n_total = len(samples)

    try:
        for i in tqdm(range(start_idx, n_total), initial=start_idx, total=n_total,
                       desc="Per-token (gen)"):
            sample = samples[i]
            text = _build_gen_text(tokenizer, sample)

            inputs = tokenizer(text, return_tensors="pt")
            input_ids = inputs["input_ids"].to(model.device)
            n_tokens = int(input_ids.shape[1])

            # Also get the boundary: where does the prompt end and response begin?
            prompt_text = _build_input_text(tokenizer, sample)
            prompt_ids = tokenizer(prompt_text, return_tensors="pt")
            n_prompt_tokens = int(prompt_ids["input_ids"].shape[1])

            with torch.no_grad():
                model(input_ids)

            captured = hooks.get_and_clear()

            # Save per-sample: layers × seq_len × hidden_dim
            arrays = {}
            for layer in target_layers:
                # captured[layer] shape: (seq_len, hidden_dim)
                arrays[f"layer_{layer}"] = captured[layer].numpy().astype(np.float16)

            sample_path = token_dir / f"sample_{i:05d}.npz"
            np.savez(sample_path, **arrays)

            meta = _make_meta(sample, n_tokens)
            meta["n_prompt_tokens"] = n_prompt_tokens
            meta["n_response_tokens"] = n_tokens - n_prompt_tokens
            meta["sample_file"] = f"sample_{i:05d}.npz"
            metadata.append(meta)

            # Checkpoint metadata periodically
            if (i + 1) % checkpoint_every == 0:
                print(f"\n  Checkpointing metadata at {i + 1}/{n_total}...")
                with open(meta_path, "w") as f:
                    json.dump(metadata, f, indent=2)

    finally:
        hooks.remove()
        # Always save metadata on exit
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)

    print(f"Saved {len(metadata)} per-token samples to {token_dir}")
    return metadata


# ── Helpers ─────────────────────────────────────────────────────────

def _make_meta(sample: dict, n_tokens: int) -> dict:
    return {
        "run_id": sample["run_id"],
        "seed": sample["seed"],
        "mode": sample["mode"],
        "agent_name": sample["agent_name"],
        "role": sample["role"],
        "phase": sample["phase"],
        "channel": sample["channel"],
        "round": sample["round"],
        "flagged": sample["flagged"],
        "bp_corr": sample["bp_corr"],
        "n_tokens": n_tokens,
    }


def _save_checkpoint(activations, metadata, target_layers, next_idx, output_dir, suffix):
    arrays = {}
    for layer in target_layers:
        arrays[f"layer_{layer}"] = np.stack(activations[layer])

    np.savez(output_dir / f"_checkpoint_{suffix}.npz", **arrays)
    with open(output_dir / f"_checkpoint_{suffix}_meta.json", "w") as f:
        json.dump({"next_idx": next_idx, "metadata": metadata}, f)


def save_last_token_results(activation_arrays, metadata, output_dir, suffix):
    output_dir.mkdir(parents=True, exist_ok=True)

    arrays = {f"layer_{layer}": arr for layer, arr in activation_arrays.items()}
    npz_path = output_dir / f"activations_{suffix}.npz"
    np.savez_compressed(npz_path, **arrays)
    print(f"Saved activations to {npz_path}")
    total_bytes = sum(arr.nbytes for arr in activation_arrays.values())
    print(f"  Total size: {total_bytes / 1e9:.2f} GB ({len(activation_arrays)} layers)")
    print(f"  Samples: {next(iter(activation_arrays.values())).shape[0]}")
    print(f"  Hidden dim: {next(iter(activation_arrays.values())).shape[1]}")

    meta_path = output_dir / f"metadata_{suffix}.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to {meta_path} ({len(metadata)} entries)")

    # Clean up checkpoints
    for cp in [output_dir / f"_checkpoint_{suffix}.npz",
               output_dir / f"_checkpoint_{suffix}_meta.json"]:
        if cp.exists():
            cp.unlink()

    # Print label distribution
    from collections import Counter
    by_key = Counter((m['mode'], m['role'], m['channel']) for m in metadata)
    print(f"\nSample distribution ({suffix}):")
    for k, v in sorted(by_key.items()):
        print(f"  {'/'.join(k)}: {v}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract activations from LLM for NARCBench-Stego"
    )
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR,
                        help="Path to NARCBench-Stego output directory")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--input-only", action="store_true",
                        help="Only extract input-time activations (last-token mode)")
    parser.add_argument("--gen-only", action="store_true",
                        help="Only extract generation-time activations")
    parser.add_argument("--per-token", action="store_true",
                        help="Extract all tokens (middle layers, gen-time only)")
    parser.add_argument("--layers", type=str, default=None,
                        help="Layer range, e.g. '24-40' or '16,24,32,40,48'. "
                             "Default: all 64 (last-token) or middle 7 (per-token).")
    parser.add_argument("--model", type=str, default=config.DEFAULT_MODEL,
                        help="HuggingFace model ID (default: config.DEFAULT_MODEL)")
    parser.add_argument("--channels", type=str, default=None,
                        help="Comma-separated channels to include, e.g. 'public,thinking'. Default: all.")
    parser.add_argument("--test", action="store_true",
                        help="Test mode: 5 samples only")
    args = parser.parse_args()

    # Parse layers
    if args.layers:
        if "-" in args.layers:
            lo, hi = args.layers.split("-")
            target_layers = list(range(int(lo), int(hi) + 1))
        else:
            target_layers = [int(x) for x in args.layers.split(",")]
    elif args.per_token:
        target_layers = DEFAULT_PER_TOKEN_LAYERS
    else:
        target_layers = list(range(NUM_LAYERS))

    print("=" * 60)
    print("Activation Extraction: Blackjack Collusion")
    print("=" * 60)
    print(f"Runs dir:   {args.runs_dir}")
    print(f"Output dir: {args.output_dir}")
    print(f"Mode:       {'per-token' if args.per_token else 'last-token'}")
    print(f"Layers:     {len(target_layers)} {target_layers}")
    print(f"Test mode:  {args.test}")
    print()

    samples = load_blackjack_runs(args.runs_dir)
    if not samples:
        print("ERROR: No samples found.")
        sys.exit(1)

    # Filter channels if requested
    if args.channels:
        channels = set(args.channels.split(","))
        samples = [s for s in samples if s["channel"] in channels]
        print(f"Filtered to channels {channels}: {len(samples)} samples")

    if args.test:
        samples = samples[:5]
        print(f"TEST MODE: Using only {len(samples)} samples")

    model, tokenizer = load_model(args.model)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.per_token:
        # Per-token: gen-time only, save one file per sample
        print("\n" + "=" * 60)
        print("Per-token extraction (gen-time)")
        print("=" * 60)
        start = time.time()
        metadata = extract_per_token(
            model, tokenizer, samples, target_layers, args.output_dir
        )
        elapsed = time.time() - start
        print(f"\nPer-token extraction: {elapsed / 60:.1f} min "
              f"({elapsed / len(samples):.2f} s/sample)")
    else:
        # Last-token mode
        if not args.gen_only:
            print("\n" + "=" * 60)
            print("PHASE 1: Input-time extraction (last token)")
            print("=" * 60)
            start = time.time()
            act_input, meta_input = extract_last_token(
                model, tokenizer, samples, target_layers, args.output_dir, "input"
            )
            elapsed = time.time() - start
            print(f"\nInput-time: {elapsed / 60:.1f} min ({elapsed / len(samples):.2f} s/sample)")
            save_last_token_results(act_input, meta_input, args.output_dir, "input")
            del act_input, meta_input
            torch.cuda.empty_cache()

        if not args.input_only:
            print("\n" + "=" * 60)
            print("PHASE 2: Generation-time extraction (last token)")
            print("=" * 60)
            start = time.time()
            act_gen, meta_gen = extract_last_token(
                model, tokenizer, samples, target_layers, args.output_dir, "gen"
            )
            elapsed = time.time() - start
            print(f"\nGeneration-time: {elapsed / 60:.1f} min ({elapsed / len(samples):.2f} s/sample)")
            save_last_token_results(act_gen, meta_gen, args.output_dir, "gen")

    print("\nDone!")


if __name__ == "__main__":
    main()
