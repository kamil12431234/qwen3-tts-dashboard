#!/usr/bin/env bash
set -euo pipefail

# ── Qwen3-TTS Web Panel — launch script ──────────────
cd "$(dirname "$0")"

# ── Config (edit here) ───────────────────────────────
MODEL_PATH="${QWEN_MODEL_PATH:-Qwen/Qwen3-TTS-1.7B}"   # or local path, e.g. /path/to/model
HOST="0.0.0.0"
PORT=8000
DEVICE="cuda"
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
