#!/usr/bin/env bash
set -euo pipefail

# ── Qwen3-TTS Web Panel — launch script ──────────────
cd "$(dirname "$0")"

# Model path priority:
# 1) QWEN_MODEL_PATH env var
# 2) Local model folder on /run/media/chapa/480gb (default used by web_panel.py as well)
MODEL_PATH="${QWEN_MODEL_PATH:-/run/media/chapa/480gb/qwen3-local/Qwen3-TTS-12Hz-1.7B-CustomVoice-real}"

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
