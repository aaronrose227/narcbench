#!/bin/bash
# Run the full pipeline (generation + extraction) for a given model.
#
# Usage:
#   bash scripts/run_model.sh <hf_model_id> [--quantise 8bit|4bit]
#
# Examples:
#   bash scripts/run_model.sh "Qwen/Qwen3-32B-AWQ"
#   bash scripts/run_model.sh "google/gemma-3-27b-it" --quantise 8bit
#
# The script will:
#   1. Start vLLM with the model
#   2. Generate Core, Transfer, and Stego scenario data
#   3. Stop vLLM and free GPU memory
#   4. Extract activations for all three tiers
#
# Outputs go to data/raw/<model_short>/{core,transfer,stego}/
# and data/activations/<model_short>/{core,transfer,stego}/
#
# ── GPU / vLLM tuning (override via environment variables) ────────────────────
# All vLLM-related knobs can be overridden by exporting the variable before
# running the script, e.g.:
#   TENSOR_PARALLEL=4 GPU_MEM_UTIL=0.85 bash scripts/run_model.sh <hf_id>
#
# Defaults (used in the paper's experiments):
#   VLLM_PORT          = 8020          # change if the port is in use
#   TENSOR_PARALLEL    = 2             # number of GPUs to shard across
#   GPU_MEM_UTIL       = 0.90          # fraction of GPU memory vLLM may use
#   MAX_MODEL_LEN      = 8192          # context window for vLLM
#
# Hardware used in the paper:
#   - Qwen3-32B-AWQ, GPT-OSS-20B:           workstation, 2x NVIDIA RTX (24 GB)
#   - Llama-3.1-70B-AWQ-INT4, DeepSeek-R1:  server, 2-4x NVIDIA A40 (48 GB)
# A single 24 GB GPU is enough for ~13B models with TENSOR_PARALLEL=1; a 70B
# model in 4-bit needs at least 2x 48 GB or equivalent.

set -e

# ── Arguments ─────────────────────────────────────────────────────────────────
MODEL_ID="${1:?Usage: bash scripts/run_model.sh <hf_model_id> [--quantise 8bit|4bit]}"
QUANTISE_ARG=""
if [[ "$2" == "--quantise" && -n "$3" ]]; then
    QUANTISE_ARG="--quantise $3"
    echo "Quantisation: $3"
fi

# ── vLLM / GPU defaults (override via environment) ────────────────────────────
VLLM_PORT="${VLLM_PORT:-8020}"
TENSOR_PARALLEL="${TENSOR_PARALLEL:-2}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${PYTHON:-python3}"
LOGDIR="$PROJECT/logs"
mkdir -p "$LOGDIR"
cd "$PROJECT"

# ── Derive short model name ───────────────────────────────────────────────────
MODEL_SHORT=$($PYTHON -c "import config; print(config.model_short_name('$MODEL_ID'))")
echo "Model ID:        $MODEL_ID"
echo "Short name:      $MODEL_SHORT"
echo "vLLM port:       $VLLM_PORT"
echo "Tensor parallel: $TENSOR_PARALLEL"
echo "GPU mem util:    $GPU_MEM_UTIL"
echo "Max model len:   $MAX_MODEL_LEN"
echo ""

# Output directories
RAW_CORE="$PROJECT/data/raw/$MODEL_SHORT/core"
RAW_TRANSFER="$PROJECT/data/raw/$MODEL_SHORT/transfer"
RAW_STEGO="$PROJECT/data/raw/$MODEL_SHORT/stego"
ACT_DIR="$PROJECT/data/activations/$MODEL_SHORT"

echo "=========================================="
echo "FULL PIPELINE: $MODEL_SHORT"
echo "Started: $(date)"
echo "=========================================="

# ── Step 1: Start vLLM ────────────────────────────────────────────────────────
echo ""
echo "[Step 1] Starting vLLM server with $MODEL_ID ..."
$PYTHON -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_ID" \
    --port "$VLLM_PORT" \
    --tensor-parallel-size "$TENSOR_PARALLEL" \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    > "$LOGDIR/vllm_${MODEL_SHORT}.log" 2>&1 &
VLLM_PID=$!
echo "  vLLM PID: $VLLM_PID"

echo "  Waiting for vLLM to accept connections..."
for i in $(seq 1 600); do
    if curl -s http://localhost:$VLLM_PORT/v1/models > /dev/null 2>&1; then
        echo "  vLLM ready after ${i}s"
        break
    fi
    sleep 1
done

if ! curl -s http://localhost:$VLLM_PORT/v1/models > /dev/null 2>&1; then
    echo "ERROR: vLLM failed to start. Check $LOGDIR/vllm_${MODEL_SHORT}.log"
    kill $VLLM_PID 2>/dev/null
    exit 1
fi

# ── Step 2: Generate Core ─────────────────────────────────────────────────────
echo ""
echo "[Step 2] Generating Core scenarios..."
$PYTHON -u generation/core.py \
    --model "$MODEL_ID" \
    --output-dir "$RAW_CORE" \
    > "$LOGDIR/gen_${MODEL_SHORT}_core.log" 2>&1
echo "  Done: $(date)"

# ── Step 3: Generate Transfer ─────────────────────────────────────────────────
echo ""
echo "[Step 3] Generating Transfer scenarios..."
$PYTHON -u generation/transfer.py \
    --model "$MODEL_ID" \
    --output-dir "$RAW_TRANSFER" \
    > "$LOGDIR/gen_${MODEL_SHORT}_transfer.log" 2>&1
echo "  Done: $(date)"

# ── Step 4: Generate Stego ────────────────────────────────────────────────────
echo ""
echo "[Step 4] Generating Stego scenarios..."
$PYTHON -u generation/stego.py \
    --model "$MODEL_ID" \
    --output-dir "$RAW_STEGO" \
    > "$LOGDIR/gen_${MODEL_SHORT}_stego.log" 2>&1
echo "  Done: $(date)"

# ── Step 5: Stop vLLM ─────────────────────────────────────────────────────────
echo ""
echo "[Step 5] Stopping vLLM..."
kill $VLLM_PID 2>/dev/null || true
sleep 10
pkill -f "vllm" 2>/dev/null || true
sleep 10
echo "  vLLM stopped, GPU memory freed"

# ── Step 6: Extract activations (Core) ────────────────────────────────────────
# --gen-only halves disk/time; probes only read activations_gen.npz.
echo ""
echo "[Step 6] Extracting activations: Core..."
mkdir -p "$ACT_DIR/core"
$PYTHON -u extraction/extract_core.py \
    --model "$MODEL_ID" $QUANTISE_ARG --gen-only \
    --runs-dir "$RAW_CORE" \
    --output-dir "$ACT_DIR/core" \
    > "$LOGDIR/extract_${MODEL_SHORT}_core.log" 2>&1
echo "  Done: $(date)"

# ── Step 7: Extract activations (Transfer) ────────────────────────────────────
echo ""
echo "[Step 7] Extracting activations: Transfer..."
mkdir -p "$ACT_DIR/transfer"
$PYTHON -u extraction/extract_transfer.py \
    --model "$MODEL_ID" $QUANTISE_ARG --gen-only \
    --modes collusion,control \
    --runs-dir "$RAW_TRANSFER" \
    --output-dir "$ACT_DIR/transfer" \
    > "$LOGDIR/extract_${MODEL_SHORT}_transfer.log" 2>&1
echo "  Done: $(date)"

# ── Step 8: Extract activations (Stego) ───────────────────────────────────────
echo ""
echo "[Step 8] Extracting activations: Stego..."
mkdir -p "$ACT_DIR/stego"
$PYTHON -u extraction/extract_stego.py \
    --model "$MODEL_ID" $QUANTISE_ARG --gen-only \
    --runs-dir "$RAW_STEGO" \
    --output-dir "$ACT_DIR/stego" \
    > "$LOGDIR/extract_${MODEL_SHORT}_stego.log" 2>&1
echo "  Done: $(date)"

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo "=========================================="
echo "PIPELINE COMPLETE: $MODEL_SHORT"
echo "Finished: $(date)"
echo "=========================================="
echo ""
echo "Raw data:     $PROJECT/data/raw/$MODEL_SHORT/"
echo "Activations:  $ACT_DIR/"
echo "Logs:         $LOGDIR/*${MODEL_SHORT}*"
