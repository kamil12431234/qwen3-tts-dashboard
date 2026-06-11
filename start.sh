#!/usr/bin/env bash
set -euo pipefail

# ── Qwen3-TTS Web Panel — launch script ──────────────
cd "$(dirname "$0")"

# IMPORTANT: set your model path here or via env var QWEN_MODEL_PATH.
# Examples:
#   MODEL_PATH="Qwen/Qwen3-TTS-1.7B"             # public HF ID (if accessible)
#   MODEL_PATH="/path/to/your/local/model/folder" # local directory
MODEL_PATH="${QWEN_MODEL_PATH:-/home/chapa/.cache/huggingface/hub/models--Qwen--Qwen3-TTS-12Hz-1.7B-CustomVoice/snapshots/0c0e3051f131929182e2c023b9537f8b1c68adfe}"

if [ -z "$MODEL_PATH" ]; then
    echo "ERROR: QWEN_MODEL_PATH is not set." >&2
    exit 1
fi

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

HOST="0.0.0.0"
PORT=8000
DEVICE="${DEVICE:-cuda}"        # or cpu if GPU is full (e.g., by LM Studio)
DTYPE="bf16"
ATTN="sdpa"               # sdpa or flash_attention_2 (if installed)
MAX_SEQ_LEN=2048

# ── Activate venv if present ────────────────────────
if [ -f "venv/bin/activate" ]; then
    source "venv/bin/activate"
fi

# ── Run panel ───────────────────────────────────────
exec python web_panel.py \
    --model "$MODEL_PATH" \
    --host "$HOST" \
    --port "$PORT" \
    --device "$DEVICE" \
    --dtype "$DTYPE" \
    --attn "$ATTN" \
    --max-seq-len "$MAX_SEQ_LEN"
