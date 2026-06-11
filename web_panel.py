#!/usr/bin/env python3
"""
Qwen3-TTS Web Dashboard — FastAPI + vanilla HTML/CSS/JS.

Launch:
    source venv/bin/activate
    python web_panel.py --model <MODEL_PATH_OR_ID> [--port 8000]
"""

import argparse
import logging
import base64
import io
import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse


app = FastAPI(title="Qwen3-TTS Dashboard", version="1.0.0")

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("qwen3-tts")

# ---------------------------------------------------------------------------
# Global model handle (lazy-loaded on first request)
# ---------------------------------------------------------------------------
_model: Optional[object] = None
_model_config: dict = {}

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

FILES_DIR = Path("web_files")
FILES_DIR.mkdir(exist_ok=True)


def get_model():
    global _model
    if _model is None:
        raise HTTPException(503, "Model not loaded yet. Start server with --model <path>")
    return _model


# ---------------------------------------------------------------------------
# API: Health / info
# ---------------------------------------------------------------------------
@app.get("/api/health")
async def health():
    m = _model
    return {
        "loaded": m is not None,
        "model_path": _model_config.get("model_path", ""),
        "device": _model_config.get("device", "cuda"),
        "dtype": _model_config.get("dtype", "bf16"),
        "sample_rate": getattr(m, "sample_rate", 24000) if m else 24000,
    }


# ---------------------------------------------------------------------------
# API: Generate TTS
# ---------------------------------------------------------------------------
@app.get("/api/speakers")
async def list_speakers():
    """List supported built-in speakers if available."""
    m = _model
    if not m:
        return {"available": False}
    try:
        speakers = m.model.get_supported_speakers()
        return {"available": True, "speakers": speakers}
    except Exception as e:
        log.warning("Failed to list speakers: %s", e)
        return {"available": False, "error": str(e)}


@app.post("/api/generate")
async def generate_tts(
    text: str = Form(...),
    mode: str = Form("clone"),
    language: str = Form("Auto"),
    temperature: float = Form(0.9),
    top_k: int = Form(50),
    do_sample: bool = Form(True),
    repetition_penalty: float = Form(1.05),
    max_new_tokens: int = Form(2048),
    instruct: str = Form(""),
    speaker: str = Form(""),
    ref_text: str = Form(""),
    xvec_only: bool = Form(False),
    streaming: bool = Form(True),
    ref_audio: Optional[UploadFile] = File(None),
):
    model = get_model()
    start = time.perf_counter()

    # ── Save uploaded reference audio ──
    ref_audio_path: Optional[str] = None
    if ref_audio is not None and ref_audio.filename:
        upload_id = uuid.uuid4().hex[:8]
        ref_audio_path = str(FILES_DIR / f"ref_{upload_id}.wav")
        data = await ref_audio.read()
        with open(ref_audio_path, "wb") as f:
            f.write(data)

    try:
        if mode == "clone":
            if not ref_audio_path:
                raise HTTPException(400, "Reference audio is required for voice clone mode.")
            if not ref_text:
                raise HTTPException(400, "Reference transcript (ref_text) is required for voice clone mode.")

            audio_list, sr = model.generate_voice_clone(
                text=text,
                language=language,
                ref_audio=ref_audio_path,
                ref_text=ref_text,
                temperature=temperature,
                top_k=top_k,
                do_sample=do_sample,
                repetition_penalty=repetition_penalty,
                max_new_tokens=max_new_tokens,
                xvec_only=xvec_only,
            )
            audio = audio_list[0]

        elif mode == "custom":
            if not speaker:
                raise HTTPException(400, "Speaker ID is required for custom voice mode.")

            # Validate speaker exists
            valid_speakers = model.model.get_supported_speakers()
            if speaker not in valid_speakers:
                raise HTTPException(400, f"Unknown speaker '{speaker}'. Available: {', '.join(valid_speakers)}")

            audio_list, sr = model.generate_custom_voice(
                text=text,
                speaker=speaker,
                language=language,
                instruct=instruct or None,
                temperature=temperature,
                top_k=top_k,
                do_sample=do_sample,
                repetition_penalty=repetition_penalty,
                max_new_tokens=max_new_tokens,
            )
            audio = audio_list[0]

        elif mode == "design":
            if not instruct:
                raise HTTPException(400, "Instruction is required for voice design mode.")

            # Check model supports voice design
            tts_type = getattr(model.model.model, "tts_model_type", None)
            if tts_type != "voice_design":
                raise HTTPException(400,
                    f"This model ({tts_type or 'unknown'}) does not support voice design. "
                    "Use Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign instead.")

            audio_list, sr = model.generate_voice_design(
                text=text,
                instruct=instruct,
                language=language,
                temperature=temperature,
                top_k=top_k,
                do_sample=do_sample,
                repetition_penalty=repetition_penalty,
                max_new_tokens=max_new_tokens,
            )
            audio = audio_list[0]

        else:
            raise HTTPException(400, f"Unknown mode: {mode}")

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, f"Generation failed: {e}")

    finally:
        # Cleanup uploaded ref audio
        if ref_audio_path and os.path.exists(ref_audio_path):
            os.remove(ref_audio_path)

    # ── Save output ──
    out_id = uuid.uuid4().hex[:8]
    out_path = OUTPUT_DIR / f"out_{out_id}.wav"
    sf.write(str(out_path), audio, sr)

    total_time = time.perf_counter() - start
    audio_dur = len(audio) / sr
    rtf = audio_dur / total_time if total_time > 0 else 0.0

    # ── Encode audio as base64 data-URI ──
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()

    return {
        "id": out_id,
        "audio_b64": b64,
        "sample_rate": sr,
        "duration_s": round(audio_dur, 2),
        "total_time_s": round(total_time, 2),
        "rtf": round(rtf, 2),
        "text": text,
        "mode": mode,
        "timestamp": time.strftime("%H:%M:%S"),
    }


# ---------------------------------------------------------------------------
# API: List generated files
# ---------------------------------------------------------------------------
@app.get("/api/outputs")
async def list_outputs():
    files = sorted(OUTPUT_DIR.glob("out_*.wav"), key=lambda p: p.stat().st_mtime, reverse=True)
    results = []
    for f in files[:50]:
        try:
            data, sr = sf.read(str(f))
            dur = len(data) / sr if sr else 0
        except Exception:
            dur = 0
        results.append({
            "name": f.name,
            "path": f.name,
            "duration_s": round(dur, 1),
            "mtime": f.stat().st_mtime,
        })
    return results


# ---------------------------------------------------------------------------
# API: Get audio file
# ---------------------------------------------------------------------------
@app.get("/api/audio/{name}")
async def get_audio(name: str):
    path = OUTPUT_DIR / name
    if not path.exists():
        raise HTTPException(404, "File not found")
    data, sr = sf.read(str(path))

    buf = io.BytesIO()
    sf.write(buf, data, sr, format="WAV")
    buf.seek(0)

    return StreamingResponse(buf, media_type="audio/wav",
                             headers={"Content-Disposition": f"inline; filename={name}"})


# ---------------------------------------------------------------------------
# Frontend — single-page dashboard
# ---------------------------------------------------------------------------
HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Qwen3-TTS Dashboard</title>
<style>
:root {
  --bg: #0d1117;
  --surface: #161b22;
  --border: #30363d;
  --text: #e6edf3;
  --text-dim: #8b949e;
  --accent: #58a6ff;
  --accent-hover: #79c0ff;
  --green: #3fb950;
  --red: #f85149;
  --yellow: #d2991d;
  --radius: 8px;
}
* { margin:0; padding:0; box-sizing:border-box; }
body {
  background: var(--bg);
  color: var(--text);
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  line-height: 1.5;
  min-height: 100vh;
}
header {
  background: var(--surface);
  border-bottom: 1px solid var(--border);
  padding: 12px 24px;
  display: flex;
  align-items: center;
  gap: 16px;
}
header h1 { font-size: 18px; font-weight: 600; }
header .badge { font-size: 11px; padding: 2px 8px; border-radius: 12px; }
.badge-ok { background: #1a3a1a; color: var(--green); }
.badge-err { background: #3a1a1a; color: var(--red); }
main {
  display: grid;
  grid-template-columns: 420px 1fr;
  gap: 20px;
  padding: 20px 24px;
  max-width: 1400px;
  margin: 0 auto;
  height: calc(100vh - 70px);
}
.panel {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 20px;
  overflow-y: auto;
}
.panel h2 {
  font-size: 14px;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  color: var(--text-dim);
  margin-bottom: 16px;
  padding-bottom: 8px;
  border-bottom: 1px solid var(--border);
}
.form-group { margin-bottom: 14px; }
.form-group label {
  display: block;
  font-size: 12px;
  font-weight: 500;
  color: var(--text-dim);
  margin-bottom: 4px;
}
.form-group label .required { color: var(--red); }
input, textarea, select {
  width: 100%;
  padding: 8px 10px;
  background: var(--bg);
  border: 1px solid var(--border);
  border-radius: 6px;
  color: var(--text);
  font-size: 13px;
  font-family: inherit;
  outline: none;
  transition: border 0.15s;
}
input:focus, textarea:focus, select:focus { border-color: var(--accent); }
textarea { resize: vertical; min-height: 80px; }
.row { display: flex; gap: 10px; }
.row > * { flex: 1; }
.btn {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 6px;
  padding: 10px 20px;
  border: none;
  border-radius: 6px;
  font-size: 13px;
  font-weight: 600;
  cursor: pointer;
  transition: background 0.15s, opacity 0.15s;
}
.btn:disabled { opacity: 0.5; cursor: not-allowed; }
.btn-primary { background: #238636; color: #fff; width: 100%; }
.btn-primary:hover:not(:disabled) { background: #2ea043; }
.btn-primary.loading { background: #1f6f2b; }
.mode-tabs {
  display: flex;
  gap: 4px;
  margin-bottom: 16px;
  background: var(--bg);
  border-radius: 6px;
  padding: 3px;
}
.mode-tab {
  flex: 1;
  padding: 6px 10px;
  border: none;
  border-radius: 4px;
  font-size: 12px;
  font-weight: 500;
  cursor: pointer;
  background: transparent;
  color: var(--text-dim);
  transition: all 0.15s;
}
.mode-tab.active { background: var(--accent); color: #fff; }
.mode-tab:hover:not(.active) { color: var(--text); background: var(--border); }
.status-line {
  font-size: 12px;
  color: var(--text-dim);
  min-height: 18px;
  margin-top: 10px;
}
.audio-card {
  background: var(--bg);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 12px;
  margin-bottom: 8px;
}
.audio-card .meta {
  font-size: 11px;
  color: var(--text-dim);
  display: flex;
  gap: 12px;
  margin-bottom: 6px;
}
.audio-card audio { width: 100%; height: 32px; margin-top: 4px; }
.audio-card .text-preview {
  font-size: 13px;
  color: var(--text);
  margin-bottom: 4px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.empty-state {
  text-align: center;
  color: var(--text-dim);
  padding: 40px 20px;
  font-size: 13px;
}
.spinner { animation: spin 0.8s linear infinite; }
@keyframes spin { to { transform: rotate(360deg); } }
#file-upload-area {
  border: 2px dashed var(--border);
  border-radius: 6px;
  padding: 16px;
  text-align: center;
  cursor: pointer;
  transition: border-color 0.15s;
  font-size: 12px;
  color: var(--text-dim);
}
#file-upload-area:hover, #file-upload-area.dragover { border-color: var(--accent); color: var(--accent); }
#file-upload-area .filename { color: var(--green); margin-top: 4px; font-weight: 500; }
input[type="range"] {
  -webkit-appearance: none;
  height: 4px;
  background: var(--border);
  border-radius: 2px;
  padding: 0;
  border: none;
}
input[type="range"]::-webkit-slider-thumb {
  -webkit-appearance: none;
  width: 14px; height: 14px;
  background: var(--accent);
  border-radius: 50%;
  cursor: pointer;
}
.range-val { font-size: 11px; color: var(--accent); margin-left: 8px; }
.check-row { display: flex; align-items: center; gap: 8px; }
.check-row input[type="checkbox"] { width: auto; accent-color: var(--accent); }
@media (max-width: 900px) {
  main { grid-template-columns: 1fr; height: auto; }
}
</style>
</head>
<body>

<header>
  <h1>🎙️ Qwen3-TTS Dashboard</h1>
  <span id="model-badge" class="badge badge-ok">checking…</span>
  <span style="flex:1"></span>
  <span style="font-size:12px;color:var(--text-dim)" id="sys-info"></span>
</header>

<main>
  <!-- ── Left panel: Controls ── -->
  <div class="panel" id="left-panel">

    <!-- Mode tabs -->
    <div class="mode-tabs" id="mode-tabs">
      <button type="button" class="mode-tab active" data-mode="clone">🎭 Voice Clone</button>
      <button type="button" class="mode-tab" data-mode="custom">👤 Custom Voice</button>
      <button type="button" class="mode-tab" data-mode="design">✨ Voice Design</button>
    </div>

    <form id="tts-form" novalidate>
      <!-- Mode-specific fields -->
      <div id="field-ref-audio" class="form-group">
        <label><span class="required">*</span> Reference Audio</label>
        <div id="file-upload-area">
          <span>📁 Click or drag .wav file here</span>
          <div class="filename" id="ref-filename"></div>
        </div>
        <input type="file" id="ref-audio-input" accept=".wav,.mp3,.flac,.ogg" style="display:none">
      </div>

      <div id="field-ref-text" class="form-group">
        <label><span class="required">*</span> Reference Transcript</label>
        <input type="text" id="ref-text" placeholder="What the reference audio says…">
      </div>

      <div id="field-speaker" class="form-group" style="display:none">
        <label><span class="required">*</span> Speaker ID</label>
        <input type="text" id="speaker" placeholder="e.g. en_speaker_01">
      </div>

      <div id="field-instruct" class="form-group" style="display:none">
        <label><span class="required">*</span> Voice Instruction</label>
        <textarea id="instruct" placeholder="Describe the voice and style… e.g. A warm and friendly female voice, clear and articulate"></textarea>
      </div>

      <!-- Common fields -->
      <div class="form-group">
        <label><span class="required">*</span> Text to Speak</label>
        <textarea id="text" placeholder="Enter text to synthesize…" rows="4"></textarea>
      </div>

      <div class="form-group">
        <label>Language</label>
        <input type="text" id="language" value="Auto" placeholder="Auto / English / Chinese / …">
      </div>

      <!-- Advanced parameters (collapsible) -->
      <details style="margin-bottom:14px">
        <summary style="cursor:pointer;font-size:12px;color:var(--text-dim);font-weight:500;user-select:none">
          ⚙️ Advanced Parameters
        </summary>
        <div style="padding-top:10px">
          <div class="row">
            <div class="form-group">
              <label>Temperature <span class="range-val" id="temp-val">0.9</span></label>
              <input type="range" id="temperature" min="0.1" max="2.0" step="0.05" value="0.9">
            </div>
            <div class="form-group">
              <label>Top-K <span class="range-val" id="topk-val">50</span></label>
              <input type="range" id="top-k" min="1" max="200" step="1" value="50">
            </div>
          </div>
          <div class="row">
            <div class="form-group">
              <label>Repetition Penalty <span class="range-val" id="rp-val">1.05</span></label>
              <input type="range" id="repetition-penalty" min="1.0" max="2.0" step="0.05" value="1.05">
            </div>
            <div class="form-group">
              <label>Max New Tokens</label>
              <input type="number" id="max-new-tokens" value="2048" min="64" max="8192" step="128">
            </div>
          </div>
          <div class="check-row form-group">
            <input type="checkbox" id="do-sample" checked>
            <label for="do-sample" style="margin:0">Enable Sampling</label>
          </div>
          <div id="field-xvec-only" class="check-row form-group">
            <input type="checkbox" id="xvec-only">
            <label for="xvec-only" style="margin:0">X-Vector Only (no ICL bleed-through)</label>
          </div>
        </div>
      </details>

      <button type="submit" class="btn btn-primary" id="generate-btn">
        🔊 Generate Speech
      </button>
      <div class="status-line" id="status-line"></div>
    </form>
  </div>

  <!-- ── Right panel: Results ── -->
  <div class="panel" id="right-panel">
    <h2>📋 Generated Audio</h2>
    <div id="results-container">
      <div class="empty-state">Generated audio will appear here</div>
    </div>
  </div>
</main>

<script>
// ──────────────────────────────────────────────────
// State
// ──────────────────────────────────────────────────
let currentMode = 'clone';
let refAudioFile = null;

// ──────────────────────────────────────────────────
// Init
// ──────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  checkHealth();
  loadHistory();
  setupModeTabs();
  setupFileUpload();
  setupForm();
  setupSliders();
});

// ──────────────────────────────────────────────────
// Health check
// ──────────────────────────────────────────────────
async function checkHealth() {
  try {
    const r = await fetch('/api/health');
    const h = await r.json();
    const badge = document.getElementById('model-badge');
    if (h.loaded) {
      badge.textContent = '✓ model loaded';
      badge.className = 'badge badge-ok';
      document.getElementById('sys-info').textContent =
        `${h.model_path.split('/').pop() || h.model_path} · ${h.device} · ${h.dtype}`;
    } else {
      badge.textContent = '✗ not loaded';
      badge.className = 'badge badge-err';
    }
  } catch(e) {
    document.getElementById('model-badge').textContent = '✗ offline';
    document.getElementById('model-badge').className = 'badge badge-err';
  }
}

// ──────────────────────────────────────────────────
// Mode switching
// ──────────────────────────────────────────────────
function setupModeTabs() {
  document.querySelectorAll('.mode-tab').forEach(tab => {
    tab.addEventListener('click', () => {
      document.querySelectorAll('.mode-tab').forEach(t => t.classList.remove('active'));
      tab.classList.add('active');
      currentMode = tab.dataset.mode;
      updateModeFields();
    });
  });
  updateModeFields();
}

function updateModeFields() {
  const refAudio = document.getElementById('field-ref-audio');
  const refText = document.getElementById('field-ref-text');
  const speaker = document.getElementById('field-speaker');
  const instruct = document.getElementById('field-instruct');
  const xvecOnly = document.getElementById('field-xvec-only');

  refAudio.style.display = currentMode === 'clone' ? '' : 'none';
  refText.style.display = currentMode === 'clone' ? '' : 'none';
  speaker.style.display = currentMode === 'custom' ? '' : 'none';
  instruct.style.display = (currentMode === 'design' || currentMode === 'custom') ? '' : 'none';
  xvecOnly.style.display = currentMode === 'clone' ? '' : 'none';

  if (currentMode === 'design') {
    document.querySelector('#field-instruct label .required').style.display = '';
  } else if (currentMode === 'custom') {
    document.querySelector('#field-instruct label .required').style.display = 'none';
  }
}

// ──────────────────────────────────────────────────
// File upload
// ──────────────────────────────────────────────────
function setupFileUpload() {
  const area = document.getElementById('file-upload-area');
  const input = document.getElementById('ref-audio-input');
  const label = document.getElementById('ref-filename');

  area.addEventListener('click', () => input.click());

  area.addEventListener('dragover', e => { e.preventDefault(); area.classList.add('dragover'); });
  area.addEventListener('dragleave', () => area.classList.remove('dragover'));
  area.addEventListener('drop', e => {
    e.preventDefault();
    area.classList.remove('dragover');
    if (e.dataTransfer.files.length) {
      input.files = e.dataTransfer.files;
      updateFileLabel();
    }
  });

  input.addEventListener('change', updateFileLabel);

  function updateFileLabel() {
    if (input.files.length) {
      refAudioFile = input.files[0];
      label.textContent = '✓ ' + refAudioFile.name;
    } else {
      refAudioFile = null;
      label.textContent = '';
    }
  }
}

// ──────────────────────────────────────────────────
// Sliders
// ──────────────────────────────────────────────────
function setupSliders() {
  document.getElementById('temperature').addEventListener('input', function() {
    document.getElementById('temp-val').textContent = this.value;
  });
  document.getElementById('top-k').addEventListener('input', function() {
    document.getElementById('topk-val').textContent = this.value;
  });
  document.getElementById('repetition-penalty').addEventListener('input', function() {
    document.getElementById('rp-val').textContent = this.value;
  });
}

// ──────────────────────────────────────────────────
// Form submit
// ──────────────────────────────────────────────────
function setupForm() {
  document.getElementById('tts-form').addEventListener('submit', async e => {
    e.preventDefault();
    await generate();
  });
}

async function generate() {
  const btn = document.getElementById('generate-btn');
  const status = document.getElementById('status-line');
  const text = document.getElementById('text').value.trim();

  if (!text) {
    status.textContent = '⚠️ Please enter text to synthesize.';
    status.style.color = 'var(--red)';
    return;
  }

  btn.disabled = true;
  btn.innerHTML = '<span class="spinner">⏳</span> Generating…';
  btn.classList.add('loading');
  status.textContent = '';

  const fd = new FormData();
  fd.append('text', text);
  fd.append('mode', currentMode);
  fd.append('language', document.getElementById('language').value || 'Auto');
  fd.append('temperature', document.getElementById('temperature').value);
  fd.append('top_k', document.getElementById('top-k').value);
  fd.append('do_sample', document.getElementById('do-sample').checked);
  fd.append('repetition_penalty', document.getElementById('repetition-penalty').value);
  fd.append('max_new_tokens', document.getElementById('max-new-tokens').value);
  fd.append('streaming', 'true');

  if (currentMode === 'clone') {
    fd.append('ref_text', document.getElementById('ref-text').value);
    fd.append('xvec_only', document.getElementById('xvec-only').checked);
    if (refAudioFile) {
      fd.append('ref_audio', refAudioFile);
    }
  }

  if (currentMode === 'custom') {
    fd.append('speaker', document.getElementById('speaker').value);
    fd.append('instruct', document.getElementById('instruct').value);
  }

  if (currentMode === 'design') {
    fd.append('instruct', document.getElementById('instruct').value);
  }

  try {
    const r = await fetch('/api/generate', { method: 'POST', body: fd });
    if (!r.ok) {
      let msg = r.statusText;
      try {
        const err = await r.json();
        if (Array.isArray(err.detail)) {
          msg = err.detail.map(e => e.msg).join('; ');
        } else if (err.detail) {
          msg = String(err.detail);
        }
      } catch (_) { /* not JSON */ }
      throw new Error(msg);
    }
    const result = await r.json();

    // Add to results
    prependResult(result);

    status.textContent = `✓ Done — ${result.duration_s}s audio, RTF ${result.rtf}x`;
    status.style.color = 'var(--green)';
  } catch(err) {
    status.textContent = '⚠️ Error: ' + err.message;
    status.style.color = 'var(--red)';
  } finally {
    btn.disabled = false;
    btn.innerHTML = '🔊 Generate Speech';
    btn.classList.remove('loading');
  }
}

// ──────────────────────────────────────────────────
// Results display
// ──────────────────────────────────────────────────
function prependResult(result) {
  const container = document.getElementById('results-container');
  const empty = container.querySelector('.empty-state');
  if (empty) empty.remove();

  const card = document.createElement('div');
  card.className = 'audio-card';
  card.dataset.id = result.id;

  const audioSrc = 'data:audio/wav;base64,' + result.audio_b64;

  card.innerHTML = `
    <div class="text-preview" title="${escapeHTML(result.text)}">${escapeHTML(result.text)}</div>
    <div class="meta">
      <span>🎵 ${result.duration_s}s</span>
      <span>⚡ RTF ${result.rtf}x</span>
      <span>🏷️ ${result.mode}</span>
      <span>🕐 ${result.timestamp}</span>
    </div>
    <audio controls src="${audioSrc}"></audio>
    <div style="margin-top:6px;display:flex;gap:8px">
      <button class="btn btn-primary" style="font-size:11px;padding:4px 10px"
              onclick="downloadAudio('${result.id}')">💾 Download</button>
    </div>
  `;

  container.insertBefore(card, container.firstChild);

  // Auto-play
  const audio = card.querySelector('audio');
  audio.play().catch(() => {});
}

function escapeHTML(s) {
  const div = document.createElement('div');
  div.textContent = s;
  return div.innerHTML;
}

function downloadAudio(id) {
  const a = document.createElement('a');
  a.href = '/api/audio/out_' + id + '.wav';
  a.download = 'tts_' + id + '.wav';
  a.click();
}

// ──────────────────────────────────────────────────
// History load
// ──────────────────────────────────────────────────
async function loadHistory() {
  try {
    const r = await fetch('/api/outputs');
    if (!r.ok) return;
    const files = await r.json();
    if (files.length === 0) return;

    const container = document.getElementById('results-container');
    const empty = container.querySelector('.empty-state');
    if (empty) empty.remove();

    files.forEach(f => {
      const card = document.createElement('div');
      card.className = 'audio-card';
      card.innerHTML = `
        <div class="text-preview">📁 ${escapeHTML(f.name)}</div>
        <div class="meta">
          <span>🎵 ${f.duration_s}s</span>
        </div>
        <audio controls src="/api/audio/${f.name}"></audio>
        <div style="margin-top:6px;display:flex;gap:8px">
          <button class="btn btn-primary" style="font-size:11px;padding:4px 10px"
                  onclick="downloadAudio('${f.name.replace('out_','').replace('.wav','')}')">💾 Download</button>
        </div>
      `;
      container.appendChild(card);
    });
  } catch(e) {
    // history load failure is non-critical
  }
}
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
async def index():
    return HTML


# ---------------------------------------------------------------------------

# Entry point
def main():
    global _model, _model_config

    parser = argparse.ArgumentParser(description="Qwen3-TTS Web Dashboard")
    parser.add_argument("--model", required=True, help="Model path or HuggingFace ID")
    parser.add_argument("--host", default="0.0.0.0", help="Bind address")
    parser.add_argument("--port", type=int, default=8000, help="Port")
    parser.add_argument("--device", default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"],
                        help="Model dtype")
    parser.add_argument("--attn", default="sdpa",
                        choices=["sdpa", "flash_attention_2"], help="Attention implementation")
    parser.add_argument("--max-seq-len", type=int, default=2048, help="Max sequence length")
    args = parser.parse_args()

    _model_config = {
        "model_path": args.model,
        "device": args.device,
        "dtype": args.dtype,
    }

    log.info("Loading model: %s", args.model)
    log.info("  device=%s  dtype=%s  attn=%s", args.device, args.dtype, args.attn)

    # Convert dtype string → torch.dtype
    if args.dtype == "bf16":
        torch_dtype = torch.bfloat16
    elif args.dtype == "fp16":
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    from faster_qwen3_tts import FasterQwen3TTS

    try:
        _model = FasterQwen3TTS.from_pretrained(
            model_name=args.model,
            device=args.device,
            dtype=torch_dtype,
            attn_implementation=args.attn,
            max_seq_len=args.max_seq_len,
        )
    except Exception as e:
        log.error("Failed to load model: %s", e)
        raise SystemExit(1)

    log.info("Model loaded. Sample rate: %d Hz", _model.sample_rate)
    log.info("Dashboard: http://%s:%d", args.host, args.port)

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()