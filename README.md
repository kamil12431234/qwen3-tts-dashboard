# Qwen3-TTS Web Dashboard

FastAPI-based web dashboard for **Qwen3-TTS** (voice clone, custom voices, voice design) with a modern UI and simple REST API.

## Quick start (local model on /run/media/chapa/480gb/)

1. Enter project directory:
   - cd /run/media/chapa/480gb/qwen3-local/qwen3-local

2. Activate venv (or create it):
   - source venv/bin/activate
   - pip install -r requirements.txt

3. Run with local model:
   - ./start.sh
     or
   - python web_panel.py --model /run/media/chapa/480gb/qwen3-local/Qwen3-TTS-12Hz-1.7B-CustomVoice-real

4. Open in browser:
   - http://localhost:8000

## Environment

- QWEN_MODEL_PATH (optional) — override model path:
  - export QWEN_MODEL_PATH="/path/to/your/model"

Defaults to the local model folder on /run/media/chapa/480gb if not set.

## API endpoints

- GET /                 — Web dashboard UI
- GET /api/health       — Model and runtime info
- POST /api/generate    — Generate speech (form-encoded; see web_panel.py for fields)
- GET /api/audio/{name} — Download generated audio file
- GET /api/outputs      — List recent outputs

## Notes

- Uses faster-qwen3-tts as the backend.
- Designed to run with CUDA (RTX 3070 Ti recommended); CPU mode supported but slow and limited.
- Model weights are large (~3–4 GB) — keep them on a fast disk (e.g., /run/media/chapa/480gb).
