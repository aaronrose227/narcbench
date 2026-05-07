"""Shared configuration for NARCBench.

Paths, model defaults, and a small set of helpers used by the
generation, extraction, and probe-training scripts.
"""

import os
from pathlib import Path

# ── Project root ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent

# ── Data paths ────────────────────────────────────────────────────────────────
DATA_ROOT        = ROOT / "data"
RAW_ROOT         = DATA_ROOT / "raw"
ACTIVATIONS_ROOT = DATA_ROOT / "activations"
RESULTS_ROOT     = ROOT / "results"
FIGURES_ROOT     = RESULTS_ROOT / "figures"

# ── Model defaults ────────────────────────────────────────────────────────────
DEFAULT_MODEL    = "Qwen/Qwen3-32B-AWQ"
# Port of the local vLLM server. Override via the VLLM_PORT env var; the
# default here must stay in sync with scripts/run_model.sh.
VLLM_PORT        = int(os.environ.get("VLLM_PORT", 8020))
VLLM_BASE_URL    = f"http://127.0.0.1:{VLLM_PORT}/v1"
VLLM_TEMPERATURE = 0.7
VLLM_MAX_TOKENS  = 512

# ── Recommended probe-layer ranges (mid-band of each model) ───────────────────
# Used as defaults when --layers is not passed to a probe script.
# For models not listed here, the script falls back to the middle 5 layers
# of the architecture (computed from model.config.num_hidden_layers).
MODEL_LAYERS = {
    "qwen3_32b":       "26-30",   # 64 layers total
    "llama_3_1_70b":   "32-37",   # 80 layers total
    "deepseek_r1_32b": "26-30",   # 64 layers total
    "gpt_oss_20b":     "10-14",   # 24 layers total
}


# ── Helper functions ──────────────────────────────────────────────────────────

def activations_dir(model_short: str) -> Path:
    """Return the activations directory for a given model short name."""
    return ACTIVATIONS_ROOT / model_short


def raw_dir(dataset: str) -> Path:
    """Return the raw game-outputs directory for a dataset, e.g. 'core'."""
    return RAW_ROOT / dataset


def model_short_name(model_id: str) -> str:
    """Derive a short directory name from a HuggingFace model ID.

    Examples:
        'Qwen/Qwen3-32B-AWQ'                       -> 'qwen3_32b'
        'meta-llama/Llama-3.1-70B-Instruct-AWQ-INT4' -> 'llama_3_1_70b'
        'deepseek-ai/DeepSeek-R1-Distill-Qwen-32B' -> 'deepseek_r1_32b'
        'openai/gpt-oss-20b'                       -> 'gpt_oss_20b'
    """
    name = model_id.split("/")[-1].lower()
    # Strip quantisation / variant suffixes
    for suffix in ["-awq", "-gptq", "-int4", "-int8", "-4b-128g",
                   "-autoround-awq-sym", "-quantized", ".w4a16"]:
        name = name.replace(suffix, "")
    # Strip distillation / base-model suffixes (e.g. "DeepSeek-R1-Distill-Qwen-32B"
    # -> "deepseek_r1_32b" — the distillation base is implementation detail).
    for suffix in ["-distill-qwen", "-distill-llama", "-distill"]:
        name = name.replace(suffix, "")
    # Strip post-training suffixes
    for suffix in ["-instruct", "-it"]:
        name = name.replace(suffix, "")
    # Convert hyphens / dots to underscores, collapse
    name = name.replace("-", "_").replace(".", "_")
    while "__" in name:
        name = name.replace("__", "_")
    return name.strip("_")


def parse_layer_range(spec: str) -> list[int]:
    """Parse a layer specification like '26-30' or '26,27,28' into a list.

    Examples:
        '26-30'        -> [26, 27, 28, 29, 30]
        '26,28,30'     -> [26, 28, 30]
        '26'           -> [26]
    """
    spec = spec.strip()
    if "-" in spec:
        lo, hi = spec.split("-", 1)
        return list(range(int(lo), int(hi) + 1))
    return [int(x) for x in spec.split(",") if x.strip()]


def default_layers(model_id: str, num_hidden_layers: int | None = None) -> list[int]:
    """Choose a sensible default probe-layer range for a model.

    1. If the model is in MODEL_LAYERS, use that range.
    2. Otherwise, return the middle 5 layers of the architecture
       (requires num_hidden_layers to be passed in).
    """
    short = model_short_name(model_id)
    if short in MODEL_LAYERS:
        return parse_layer_range(MODEL_LAYERS[short])
    if num_hidden_layers is None:
        raise ValueError(
            f"Model '{model_id}' not in MODEL_LAYERS and no num_hidden_layers given. "
            f"Pass --layers explicitly or add an entry to config.MODEL_LAYERS."
        )
    mid = num_hidden_layers // 2
    return list(range(mid - 2, mid + 3))
