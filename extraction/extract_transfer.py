"""
Extract hidden-state activations for NARCBench-Transfer.

Extracts layers 16-55 (mid-to-late, where deception signal is strongest) at both:
  - input-time:      last token of the prompt before generation
  - generation-time: last token of the prompt+response

Each sample tagged with: role, channel, phase, round, scenario_id, scenario_domain,
label_variant — all fields needed for the probes.

Usage:
    python extract_transfer.py
    python extract_transfer.py --runs-dir PATH
    python extract_transfer.py --gen-only
    python extract_transfer.py --test       # 10 samples
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
import torch.nn as nn

# AutoAWQ 0.2.9 imports PytorchGELUTanh from transformers.activations, which was
# removed in transformers >=4.56. Shim it back in before any AWQ code runs.
import transformers.activations as _ta
if not hasattr(_ta, "PytorchGELUTanh"):
    class _PytorchGELUTanh(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.nn.functional.gelu(x, approximate="tanh")
    _ta.PytorchGELUTanh = _PytorchGELUTanh

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from tqdm import tqdm

CHECKPOINT_EVERY = 100
DEFAULT_LAYER_RANGE = "16-55"

# Requires raw scenario outputs from generation/transfer.py
DEFAULT_RUNS_DIR = config.RAW_ROOT / "transfer"
DEFAULT_OUTPUT_DIR = None  # derived from --model in main() if --output-dir not given


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _strip_name_prefix(msg: str) -> str:
    """Remove '[Agent Name]: ' prefix from a stored message."""
    return re.sub(r"^\[[^\]]+\]:\s*", "", msg)


def load_runs(runs_dir: Path, modes: set[str] | None = None) -> list[dict]:
    """Load all samples from NARCBench-Transfer run directories.

    If `modes` is provided, only include runs whose config['mode'] is in that set.
    """
    run_dirs = sorted([d for d in runs_dir.iterdir() if d.is_dir()])
    if modes:
        run_dirs = [
            d for d in run_dirs
            if json.loads((d / "run_config.json").read_text()).get("mode") in modes
        ]
        print(f"Found {len(run_dirs)} run directories in {runs_dir} (modes={sorted(modes)})")
    else:
        print(f"Found {len(run_dirs)} run directories in {runs_dir}")

    samples = []
    missing = []

    for run_dir in run_dirs:
        config_path  = run_dir / "run_config.json"
        prompts_path = run_dir / "agent_prompts.json"
        results_path = run_dir / "results.json"

        if not all(p.exists() for p in [config_path, prompts_path, results_path]):
            missing.append(run_dir.name)
            continue

        config  = json.loads(config_path.read_text())
        prompts = json.loads(prompts_path.read_text())
        results = json.loads(results_path.read_text())

        # --- Response lists (ordered, matching prompt order) ---
        private_responses = [_strip_name_prefix(m) for m in results.get("private_messages", [])]
        public_responses  = [_strip_name_prefix(m) for m in results.get("public_messages",  [])]
        final_texts       = results.get("final_texts", {})

        private_idx = 0
        public_idx  = 0

        for entry in prompts:
            phase   = entry["phase"]
            channel = entry["channel"]
            agent   = entry["agent_name"]

            if channel == "private":
                # Private-round messages (phase may be "private" or "discussion"
                # depending on generator; channel is the stable discriminator).
                response = private_responses[private_idx] if private_idx < len(private_responses) else ""
                private_idx += 1
            elif phase == "discussion" and channel == "public":
                response = public_responses[public_idx] if public_idx < len(public_responses) else ""
                public_idx += 1
            elif phase == "final":
                response = final_texts.get(agent, "")
            else:
                response = ""

            samples.append({
                # Identification
                "run_id":           config["run_id"],
                "seed":             config.get("seed", 0),
                "scenario_id":      config["scenario_id"],
                "scenario_domain":  config.get("scenario_domain", config.get("scenario_type", "")),
                "scenario_title":   config.get("scenario_title", ""),
                "mode":             config["mode"],
                # Agent / role
                "agent_name":       agent,
                "role":             entry["role"],
                "phase":            phase,
                "channel":          channel,
                "round":            entry["round"],
                # Outcome labels
                "collusion_success": config.get("collusion_success", False),
                "target_majority":   config.get("target_majority", False),
                "target_option":     config.get("target_option", ""),
                "fair_option":       config.get("fair_option", ""),
                "label_variant":     config.get("label_variant", 0),
                "option_labels":     config.get("option_labels", ["A", "B"]),
                "agent_vote":        config.get("votes", {}).get(agent, "?"),
                "flagged":           config.get("flagged", False),
                # Prompts & response
                "system_prompt": entry["system_prompt"],
                "user_prompt":   entry["user_prompt"],
                "response":      response,
            })

    if missing:
        print(f"  WARNING: {len(missing)} dirs missing files, skipped: {missing[:5]}{'...' if len(missing)>5 else ''}")

    print(f"Loaded {len(samples)} total samples")
    _print_distribution(samples)
    return samples


def _print_distribution(samples: list[dict]):
    counts: dict[str, int] = {}
    for s in samples:
        key = f"{s['mode']:10s}  {s['role']:8s}  {s['channel']:8s}  phase={s['phase']}"
        counts[key] = counts.get(key, 0) + 1
    for k in sorted(counts):
        print(f"  {k}: {counts[k]}")


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def load_model(model_name: str, quantise: str | None = None):
    print(f"Loading tokenizer: {model_name}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    except (ValueError, ImportError):
        # Some GPTQ models have tokenizer issues with fast tokenizer; fall back to slow
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True, use_fast=False
        )

    if quantise == "8bit":
        print(f"Loading model: {model_name} (8-bit quantisation)")
        model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map="auto", trust_remote_code=True,
            quantization_config=BitsAndBytesConfig(load_in_8bit=True),
        )
        model.eval()
        print("Model loaded.")
        return model, tokenizer

    if quantise == "4bit":
        print(f"Loading model: {model_name} (4-bit quantisation)")
        model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map="auto", trust_remote_code=True,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16),
        )
        model.eval()
        print("Model loaded.")
        return model, tokenizer

    if "awq" in model_name.lower():
        # Transformers 5.x dropped direct AWQ support in favour of gptqmodel, so use
        # AutoAWQ's own loader. It returns a wrapper with .model (the HF module).
        from awq import AutoAWQForCausalLM
        print(f"Loading model: {model_name} (AutoAWQ, device_map=auto)")
        wrapper = AutoAWQForCausalLM.from_quantized(
            model_name,
            device_map="auto",
            fuse_layers=False,
            trust_remote_code=True,
        )
        model = wrapper.model
    elif "gpt-oss" in model_name.lower():
        # GPT-OSS loads in MXFP4 native when `kernels` is installed (~12 GB total).
        # Eager is the only attention impl supported in transformers 5.5 for this
        # architecture — callers must ensure samples stay under ~4 k tokens.
        print(f"Loading model: {model_name} (MXFP4 native, eager attention)")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            max_memory={0: "22GiB", 1: "22GiB", "cpu": "64GiB"},
            trust_remote_code=True,
        )
    else:
        # Generic path: transformers handles FP16/BF16 and any other
        # quantisation scheme declared in the model's config.json.
        print(f"Loading model: {model_name} (AutoModelForCausalLM, device_map=auto)")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            trust_remote_code=True,
        )
    model.eval()
    print("Model loaded.")
    return model, tokenizer


# ---------------------------------------------------------------------------
# Activation hooks
# ---------------------------------------------------------------------------

def _get_layers(model):
    """Return the transformer layer list, handling common architectures."""
    if hasattr(model.model, "layers"):
        return model.model.layers
    # Multimodal / language-model-nested (e.g. Gemma 4)
    if hasattr(model.model, "language_model") and hasattr(model.model.language_model, "layers"):
        return model.model.language_model.layers
    raise AttributeError(f"Cannot find transformer layers in {type(model).__name__}")


class ActivationHooks:
    """Capture last-token hidden state at each target layer via forward hooks."""

    def __init__(self, model, target_layers: list[int]):
        self.captured: dict[int, torch.Tensor] = {}
        self.hooks = []
        layers = _get_layers(model)
        for idx in target_layers:
            hook = layers[idx].register_forward_hook(self._make_hook(idx))
            self.hooks.append(hook)

    def _make_hook(self, idx: int):
        def hook_fn(module, input, output):
            hidden = output[0] if isinstance(output, tuple) else output
            # hidden shape: (batch=1, seq_len, hidden_dim) — take last token, float32
            self.captured[idx] = hidden[0, -1, :].float().cpu()
        return hook_fn

    def get_and_clear(self) -> dict[int, torch.Tensor]:
        result = dict(self.captured)
        self.captured.clear()
        return result

    def remove(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()


# ---------------------------------------------------------------------------
# Text builders
# ---------------------------------------------------------------------------

def _chat_kwargs(tokenizer):
    """Return chat-template kwargs that suppress thinking traces for Qwen3.

    Other tokenisers reject this kwarg, so apply only when the template
    actually understands it.
    """
    name = (getattr(tokenizer, "name_or_path", "") or "").lower()
    if "qwen3" in name:
        return {"chat_template_kwargs": {"enable_thinking": False}}
    return {}


def _build_input_text(tokenizer, sample: dict) -> str:
    """Prompt only, with generation prompt appended (input-time)."""
    messages = [
        {"role": "system",  "content": sample["system_prompt"]},
        {"role": "user",    "content": sample["user_prompt"]},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        **_chat_kwargs(tokenizer),
    )


def _build_gen_text(tokenizer, sample: dict) -> str:
    """Prompt + response (generation-time)."""
    messages = [
        {"role": "system",    "content": sample["system_prompt"]},
        {"role": "user",      "content": sample["user_prompt"]},
        {"role": "assistant", "content": sample["response"]},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False,
        **_chat_kwargs(tokenizer),
    )


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def extract_activations(
    model,
    tokenizer,
    samples: list[dict],
    target_layers: list[int],
    output_dir: Path,
    mode: str,                  # "input" or "gen"
    checkpoint_every: int = CHECKPOINT_EVERY,
) -> tuple[dict[int, np.ndarray], list[dict]]:

    ckpt_npz  = output_dir / f"_checkpoint_{mode}.npz"
    ckpt_meta = output_dir / f"_checkpoint_{mode}_meta.json"
    start_idx = 0

    if ckpt_npz.exists() and ckpt_meta.exists():
        print(f"  Resuming {mode} from checkpoint...")
        ckpt_data = np.load(ckpt_npz)
        ckpt_info = json.loads(ckpt_meta.read_text())
        start_idx = ckpt_info["next_idx"]
        activations = {
            layer: list(ckpt_data[f"layer_{layer}"][:start_idx])
            for layer in target_layers
        }
        metadata = ckpt_info["metadata"]
        print(f"  Resuming from sample {start_idx}/{len(samples)}")
    else:
        activations = {layer: [] for layer in target_layers}
        metadata: list[dict] = []

    hooks = ActivationHooks(model, target_layers)

    try:
        for i in tqdm(range(start_idx, len(samples)), initial=start_idx,
                      total=len(samples), desc=f"Extracting ({mode})"):
            sample = samples[i]

            text = _build_input_text(tokenizer, sample) if mode == "input" \
                   else _build_gen_text(tokenizer, sample)

            if not sample["response"] and mode == "gen":
                # No response stored — fill with zeros so indices stay aligned
                n_tokens = 0
                for layer in target_layers:
                    dummy = np.zeros(model.config.hidden_size, dtype=np.float32)
                    activations[layer].append(dummy)
            else:
                inputs = tokenizer(text, return_tensors="pt")
                input_ids = inputs["input_ids"].to(model.device)
                n_tokens = int(input_ids.shape[1])
                with torch.no_grad():
                    model(input_ids)
                captured = hooks.get_and_clear()
                for layer in target_layers:
                    activations[layer].append(captured[layer].numpy())

            metadata.append({
                "run_id":           sample["run_id"],
                "seed":             sample["seed"],
                "scenario_id":      sample["scenario_id"],
                "scenario_domain":  sample["scenario_domain"],
                "mode":             sample["mode"],
                "agent_name":       sample["agent_name"],
                "role":             sample["role"],
                "phase":            sample["phase"],
                "channel":          sample["channel"],
                "round":            sample["round"],
                "collusion_success": sample["collusion_success"],
                "target_majority":  sample["target_majority"],
                "target_option":    sample["target_option"],
                "fair_option":      sample["fair_option"],
                "label_variant":    sample["label_variant"],
                "agent_vote":       sample["agent_vote"],
                "flagged":          sample["flagged"],
                "n_tokens":         n_tokens,
            })

            if (i + 1) % checkpoint_every == 0:
                print(f"\n  Checkpoint at {i+1}/{len(samples)}...")
                _save_checkpoint(activations, metadata, target_layers, i + 1,
                                 output_dir, mode)
    finally:
        hooks.remove()

    return {layer: np.stack(activations[layer]) for layer in target_layers}, metadata


def _save_checkpoint(activations, metadata, target_layers, next_idx, output_dir, suffix):
    arrays = {f"layer_{l}": np.stack(activations[l]) for l in target_layers}
    np.savez(output_dir / f"_checkpoint_{suffix}.npz", **arrays)
    (output_dir / f"_checkpoint_{suffix}_meta.json").write_text(
        json.dumps({"next_idx": next_idx, "metadata": metadata})
    )


def save_results(activation_arrays, metadata, output_dir: Path, suffix: str):
    output_dir.mkdir(parents=True, exist_ok=True)

    npz_path = output_dir / f"activations_{suffix}.npz"
    np.savez_compressed(npz_path, **{f"layer_{l}": arr for l, arr in activation_arrays.items()})
    total_gb = sum(a.nbytes for a in activation_arrays.values()) / 1e9
    n_samples = next(iter(activation_arrays.values())).shape[0]
    hidden_dim = next(iter(activation_arrays.values())).shape[1]
    print(f"Saved {npz_path.name}  |  {n_samples} samples × {hidden_dim} dim × {len(activation_arrays)} layers  |  {total_gb:.2f} GB")

    meta_path = output_dir / f"metadata_{suffix}.json"
    meta_path.write_text(json.dumps(metadata, indent=2))
    print(f"Saved {meta_path.name}  ({len(metadata)} entries)")

    # Clean up checkpoint files
    for f in [output_dir / f"_checkpoint_{suffix}.npz",
              output_dir / f"_checkpoint_{suffix}_meta.json"]:
        if f.exists():
            f.unlink()

    # Distribution summary
    counts: dict[str, int] = {}
    for m in metadata:
        key = f"{m['mode']:10s}  {m['role']:8s}  {m['channel']:8s}  phase={m['phase']}"
        counts[key] = counts.get(key, 0) + 1
    print(f"\nDistribution ({suffix}):")
    for k in sorted(counts):
        print(f"  {k}: {counts[k]}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Extract mid-layer activations for NARCBench-Transfer"
    )
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR,
                        help="Path to NARCBench-Transfer output directory")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Output directory (default: data/activations/<model_short>/transfer)")
    parser.add_argument("--layers", type=str, default=None,
                        help="Layer range to extract, e.g. '16-55' or '26,28,30'. "
                             "Defaults to config.MODEL_LAYERS for known models, "
                             "else the middle 5 layers of the architecture.")
    parser.add_argument("--model", type=str, default=config.DEFAULT_MODEL,
                        help="HuggingFace model ID (default: config.DEFAULT_MODEL)")
    parser.add_argument("--quantise", type=str, default=None, choices=["4bit", "8bit"],
                        help="Quantisation mode (default: none)")
    parser.add_argument("--input-only", action="store_true")
    parser.add_argument("--gen-only",   action="store_true")
    parser.add_argument("--modes",      type=str, default=None,
                        help="Comma-separated list of run modes to include "
                             "(e.g. 'collusion,control' to skip implicit). "
                             "Default: include all.")
    parser.add_argument("--test",       action="store_true", help="10 samples only")
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = config.activations_dir(config.model_short_name(args.model)) / "transfer"

    # Defer layer-range default to after model load so we can read num_hidden_layers.
    target_layers = config.parse_layer_range(args.layers) if args.layers else None

    print("=" * 60)
    print("Activation Extraction: NARCBench-Transfer")
    print("=" * 60)
    print(f"Runs dir:   {args.runs_dir}")
    print(f"Output dir: {args.output_dir}")
    print(f"Layers:     {target_layers[0]}–{target_layers[-1]} ({len(target_layers)} layers)")
    print(f"Test mode:  {args.test}")
    print()

    if not args.runs_dir.exists():
        print(f"ERROR: runs dir not found: {args.runs_dir}")
        sys.exit(1)

    modes_filter = set(m.strip() for m in args.modes.split(",")) if args.modes else None
    samples = load_runs(args.runs_dir, modes=modes_filter)
    if not samples:
        print("ERROR: No samples found.")
        sys.exit(1)

    if args.test:
        samples = samples[:10]
        print(f"\nTEST MODE: {len(samples)} samples only")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model, tokenizer = load_model(args.model, quantise=args.quantise)

    # If no --layers was given, use config defaults (or middle 5 of the model).
    if target_layers is None:
        target_layers = config.default_layers(args.model, model.config.num_hidden_layers)
    target_layers = [l for l in target_layers if l < model.config.num_hidden_layers]
    print(f"Extracting layers: {target_layers}")

    # Input-time
    if not args.gen_only:
        print("\n" + "=" * 60)
        print("PHASE 1: Input-time extraction")
        print("=" * 60)
        t0 = time.time()
        act, meta = extract_activations(model, tokenizer, samples, target_layers,
                                        args.output_dir, "input")
        elapsed = time.time() - t0
        print(f"\nInput-time: {elapsed/60:.1f} min  ({elapsed/len(samples):.2f} s/sample)")
        save_results(act, meta, args.output_dir, "input")
        del act, meta
        torch.cuda.empty_cache()

    # Generation-time
    if not args.input_only:
        print("\n" + "=" * 60)
        print("PHASE 2: Generation-time extraction")
        print("=" * 60)
        t0 = time.time()
        act, meta = extract_activations(model, tokenizer, samples, target_layers,
                                        args.output_dir, "gen")
        elapsed = time.time() - t0
        print(f"\nGeneration-time: {elapsed/60:.1f} min  ({elapsed/len(samples):.2f} s/sample)")
        save_results(act, meta, args.output_dir, "gen")

    print("\nDone.")


if __name__ == "__main__":
    main()
