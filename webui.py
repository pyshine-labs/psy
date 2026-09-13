#!/usr/bin/env python3
"""PSY Music Studio: a single-file FastAPI server that lets you pick models/backends,
enter lyrics+style, watch live progress, and play/download the generated song.

Run:
    run.bat   (Windows)   or   ./run.sh   (Linux/macOS)
Then open http://127.0.0.1:7860
"""
from __future__ import annotations
import io
import json
import os
import re
import sys
import threading
import time
import traceback
import uuid
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse, FileResponse
import uvicorn

ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = ROOT / "outputs"
OUTPUT_ROOT.mkdir(exist_ok=True)

DEFAULT_MODELS = [
    {"id": "m-a-p/YuE2-3B", "label": "YuE2-3B (Hugging Face)", "vae": "m-a-p/YuE2-Vae"},
]
# Local cached models discovered in the HF cache appear in /api/options dynamically.

app = FastAPI(title="PSY Music Studio")

JOBS: dict[str, dict] = {}
JOB_LOCK = threading.Lock()


# --------------------------------------------------------------------------- #
# Stderr capture: yue2.Progress writes "[YuE2] prefix Label: suffix" lines to
# stderr. We swap sys.stderr per-worker-thread so concurrent UI calls don't
# cross streams (we still serialize model jobs below via a global lock).
# --------------------------------------------------------------------------- #
class StreamCapture:
    _LINE_RE = re.compile(r"\[YuE2\]\s+(\S+)\s+([^:]+?):\s*(.*)")

    def __init__(self, job_id: str):
        self.job_id = job_id
        self._linebuf = ""

    def write(self, s: str) -> int:
        self._linebuf += s
        while "\n" in self._linebuf:
            line, self._linebuf = self._linebuf.split("\n", 1)
            self._emit(line)
        return len(s)

    def flush(self) -> None:
        if self._linebuf:
            self._emit(self._linebuf)
            self._linebuf = ""

    def isatty(self) -> bool:
        return False

    def fileno(self):
        raise OSError("StreamCapture has no fileno")

    def _emit(self, line: str) -> None:
        line = line.rstrip()
        if not line:
            return
        m = self._LINE_RE.match(line)
        if m:
            prefix, label, suffix = m.group(1), m.group(2).strip(), m.group(3).strip()
            _push(self.job_id, {"type": "progress", "prefix": prefix, "stage": label, "suffix": suffix})
        else:
            _push(self.job_id, {"type": "log", "line": line})


def _push(job_id: str, event: dict) -> None:
    job = JOBS.get(job_id)
    if job is not None:
        job["queue"].put(event)


# Only one model job can run at a time (the yue2 pipeline is not concurrent).
JOB_RUN_LOCK = threading.Lock()


def worker(job_id: str, params: dict) -> None:
    job = JOBS[job_id]
    cancel_event = job["cancel"]
    capture = StreamCapture(job_id)
    old_stderr = sys.stderr
    sys.stderr = capture
    try:
        with JOB_RUN_LOCK:
            if cancel_event.is_set():
                _push(job_id, {"type": "log", "line": "Cancelled before start"})
                return
            _push(job_id, {"type": "log", "line": "Loading yue2 + torch... (first run is slow)"})
            from yue2 import YuE2Pipeline
            pipe = YuE2Pipeline.from_pretrained(
                params["model"], vae=params["vae"], device=params["device"],
                backend=params["backend"], quantization=params["quantization"],
                memory_budget_gib=params["budget"], offload_ar=params["offload_ar"],
                progress=True,
            )
            try:
                request_kwargs = {"style": params["style"], "lyrics": params["lyrics"], "cot": params["cot"]}
                if params.get("seed") not in (None, ""):
                    request_kwargs["seed"] = int(params["seed"])
                if params.get("cfg_scale") not in (None, ""):
                    request_kwargs["cfg_scale"] = float(params["cfg_scale"])
                if params.get("abc"):
                    request_kwargs["abc"] = params["abc"]
                out_dir = OUTPUT_ROOT / job_id
                out_dir.mkdir(parents=True, exist_ok=True)
                _push(job_id, {"type": "stage", "name": "generate", "status": "running"})
                # The pipeline checks `cancelled` between tokens and between stages;
                # raising InterruptedError is how yue2 signals a graceful cancel.
                song = pipe(**request_kwargs, cancelled=cancel_event.is_set)
                if cancel_event.is_set():
                    raise InterruptedError("Cancelled after completion")
                song.save_artifacts(out_dir)
                audio_path = out_dir / "audio.flac"
                job["audio_path"] = str(audio_path)
                job["out_dir"] = str(out_dir)
                job["truncated"] = song.truncated
                _push(job_id, {"type": "complete", "audio": str(audio_path),
                               "truncated": song.truncated, "output_dir": str(out_dir)})
            finally:
                try:
                    pipe.close()
                except Exception:
                    pass
    except InterruptedError as e:
        _push(job_id, {"type": "log", "line": f"Cancelled: {e}"})
        _push(job_id, {"type": "cancelled", "message": str(e)})
    except Exception as e:
        tb = traceback.format_exc()
        job["error"] = str(e)
        _push(job_id, {"type": "error", "message": str(e), "traceback": tb})
    finally:
        try:
            sys.stderr.flush()
        except Exception:
            pass
        sys.stderr = old_stderr
        _push(job_id, None)  # sentinel: stream ends


# --------------------------------------------------------------------------- #
# API endpoints
# --------------------------------------------------------------------------- #
@app.get("/api/options")
def options():
    import torch
    devices = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            devices.append({"id": f"cuda:{i}", "name": p.name, "vram_gib": round(p.total_memory / 2**30, 1)})
    if not devices:
        devices = [{"id": "cpu", "name": "CPU (very slow)", "vram_gib": 0}]
    # Discover locally cached HF models (best-effort)
    extra_models = []
    hf_home = Path(os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface"))) / "hub"
    if hf_home.exists():
        for d in hf_home.iterdir():
            if d.name.startswith("models--") and "YuE2" in d.name:
                repo = d.name.replace("models--", "").replace("--", "/")
                extra_models.append({"id": repo, "label": f"{repo} (cached)", "vae": "m-a-p/YuE2-Vae"})
    models = DEFAULT_MODELS + extra_models
    return {
        "models": models,
        "backends": [
            {"id": "torch-eager", "label": "torch-eager (native SDPA, recommended on RTX without flash-attn)", "default": True},
            {"id": "torch", "label": "torch (CUDA graphs + flash; needs flash-attn build)", "default": False},
            {"id": "vllm", "label": "vllm (advanced)", "default": False},
        ],
        "quantizations": [
            {"id": "none", "label": "none (bfloat16)", "default": True},
            {"id": "fp8", "label": "fp8 (lower VRAM)", "default": False},
        ],
        "cot_modes": [
            {"id": "full", "label": "full (chords + melody ABC, default)", "default": True},
            {"id": "melody", "label": "melody (melody ABC only)", "default": False},
            {"id": "off", "label": "off (no ABC planning)", "default": False},
        ],
        "devices": devices,
        "default_device": "cuda:0" if any(d["id"].startswith("cuda") for d in devices) else "cpu",
    }


@app.post("/api/generate")
async def generate(request: dict):
    # Accept either JSON body
    body = request  # already parsed by fastapi when used as `request: dict`
    job_id = body.get("id") or uuid.uuid4().hex[:12]
    required = ("style", "lyrics")
    for k in required:
        if not body.get(k):
            return JSONResponse({"error": f"Missing field: {k}"}, status_code=400)
    params = {
        "model": body.get("model", "m-a-p/YuE2-3B"),
        "vae": body.get("vae", "m-a-p/YuE2-Vae"),
        "device": body.get("device", "auto"),
        "backend": body.get("backend", "torch-eager"),
        "quantization": body.get("quantization", "none"),
        "budget": float(body.get("budget", 24)),
        "offload_ar": bool(body.get("offload_ar", False)),
        "style": body["style"],
        "lyrics": body["lyrics"],
        "cot": body.get("cot", "full"),
        "seed": body.get("seed"),
        "cfg_scale": body.get("cfg_scale"),
        "abc": body.get("abc") or None,
    }
    import queue
    with JOB_LOCK:
        if job_id in JOBS and JOBS[job_id]["thread"].is_alive():
            return JSONResponse({"error": "Job already running"}, status_code=409)
        JOBS[job_id] = {
            "queue": queue.Queue(),
            "audio_path": None,
            "out_dir": None,
            "error": None,
            "truncated": None,
            "started_at": time.time(),
            "thread": None,
            "cancel": threading.Event(),
        }
        t = threading.Thread(target=worker, args=(job_id, params), daemon=True)
        JOBS[job_id]["thread"] = t
        t.start()
    return {"job_id": job_id}


@app.post("/api/generate/form")
async def generate_form(request_data: dict):
    # Same as /api/generate but kept for clarity / future form posts
    return await generate(request_data)


@app.get("/api/jobs/{job_id}/stream")
def job_stream(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        return JSONResponse({"error": "Unknown job"}, status_code=404)

    def event_gen():
        q = job["queue"]
        while True:
            try:
                item = q.get(timeout=15)
            except Exception:
                # heartbeat keeps SSE alive
                yield ": ping\n\n"
                continue
            if item is None:
                yield f"data: {json.dumps({'type': 'end'})}\n\n"
                return
            yield f"data: {json.dumps(item)}\n\n"

    headers = {
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
        "Connection": "keep-alive",
    }
    return StreamingResponse(event_gen(), media_type="text/event-stream", headers=headers)


def _kill_thread(thread: threading.Thread) -> bool:
    """Inject InterruptedError into a running thread as a last-resort stop.

    Returns True if the exception was delivered."""
    if not thread.is_alive():
        return False
    import ctypes
    tid = thread.ident
    if tid is None:
        return False
    res = ctypes.pythonapi.PyThreadState_SetAsyncExc(
        ctypes.c_ulong(tid), ctypes.py_object(InterruptedError))
    return res == 1


@app.post("/api/jobs/{job_id}/stop")
def job_stop(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        return JSONResponse({"error": "Unknown job"}, status_code=404)
    if not job.get("thread") or not job["thread"].is_alive():
        return {"ok": True, "was_running": False}
    # 1) Graceful: set the cancel event. The pipeline checks this between
    #    tokens and between stages and raises InterruptedError on its own.
    job["cancel"].set()
    # 2) Give the graceful path 1.5 s to take effect.
    job["thread"].join(timeout=1.5)
    # 3) If still alive (stuck in model loading or a long CUDA kernel),
    #    inject an exception directly into the thread.
    killed = False
    if job["thread"].is_alive():
        killed = _kill_thread(job["thread"])
        if killed:
            job["thread"].join(timeout=2.0)
    return {"ok": True, "was_running": True, "forced_kill": killed,
            "still_alive": job["thread"].is_alive()}


@app.get("/api/jobs/{job_id}/audio")
def job_audio(job_id: str):
    job = JOBS.get(job_id)
    if not job or not job.get("audio_path"):
        return JSONResponse({"error": "Audio not ready"}, status_code=404)
    return FileResponse(job["audio_path"], media_type="audio/flac", filename=f"{job_id}.flac")


@app.get("/api/jobs/{job_id}/download")
def job_download(job_id: str):
    job = JOBS.get(job_id)
    if not job or not job.get("audio_path"):
        return JSONResponse({"error": "Audio not ready"}, status_code=404)
    return FileResponse(job["audio_path"], media_type="audio/flac",
                        filename=f"yue2-{job_id}.flac",
                        headers={"Content-Disposition": f'attachment; filename="yue2-{job_id}.flac"'})


@app.get("/api/jobs/{job_id}/open-folder")
def job_open_folder(job_id: str):
    job = JOBS.get(job_id)
    if not job or not job.get("audio_path"):
        return JSONResponse({"error": "Audio not ready"}, status_code=404)
    path = Path(job["audio_path"])
    # Windows: select the file in Explorer. macOS: open -R. Linux: xdg-open the dir.
    import subprocess
    import sys as _sys
    try:
        if _sys.platform.startswith("win"):
            # explorer.exe /select, requires absolute path with backslashes
            subprocess.Popen(["explorer", "/select,", str(path)])
        elif _sys.platform == "darwin":
            subprocess.Popen(["open", "-R", str(path)])
        else:
            subprocess.Popen(["xdg-open", str(path.parent)])
        return {"ok": True}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/jobs")
def list_jobs():
    out = []
    for jid, job in list(JOBS.items()):
        out.append({
            "id": jid,
            "audio_path": job.get("audio_path"),
            "out_dir": job.get("out_dir"),
            "error": job.get("error"),
            "running": job["thread"].is_alive() if job.get("thread") else False,
            "started_at": job.get("started_at"),
        })
    return {"jobs": out}


# --------------------------------------------------------------------------- #
# HTML UI (single page, vanilla JS, no build step)
# --------------------------------------------------------------------------- #
HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>PSY Music Studio</title>
<style>
  :root{
    --bg:#0e0f15; --panel:#161821; --panel2:#1c1f2b; --border:#2a2e3d;
    --text:#e6e8f0; --muted:#8a90a6; --accent:#8b5cf6; --accent2:#06b6d4;
    --ok:#10b981; --warn:#f59e0b; --err:#ef4444;
  }
  *{box-sizing:border-box}
  body{margin:0;background:linear-gradient(180deg,#0a0b11 0%,#0e0f15 100%);
    color:var(--text);font-family:ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,sans-serif;
    min-height:100vh}
  header{padding:14px 22px;border-bottom:1px solid var(--border);
    background:linear-gradient(90deg,rgba(139,92,246,.08),rgba(6,182,212,.08));
    display:flex;align-items:center;gap:14px}
  header h1{margin:0;font-size:20px;letter-spacing:.3px}
  header .sub{color:var(--muted);font-size:13px;margin-left:auto}
  .logo{width:34px;height:34px;border-radius:9px;
    background:linear-gradient(135deg,var(--accent),var(--accent2));
    display:flex;align-items:center;justify-content:center;font-weight:700;font-size:18px;color:#0a0b11}
  main{display:grid;grid-template-columns:1fr 1fr;gap:18px;padding:18px;max-width:1500px;margin:0 auto}
  @media (max-width:980px){main{grid-template-columns:1fr}}
  .panel{background:var(--panel);border:1px solid var(--border);border-radius:14px;padding:16px}
  .panel h2{margin:0 0 12px;font-size:14px;color:var(--muted);text-transform:uppercase;letter-spacing:.8px}
  label{display:block;font-size:12px;color:var(--muted);margin:10px 0 4px}
  input,textarea,select{width:100%;background:var(--panel2);border:1px solid var(--border);
    border-radius:9px;color:var(--text);padding:9px 11px;font-size:13px;font-family:inherit}
  textarea{min-height:130px;resize:vertical;line-height:1.45}
  input:focus,textarea:focus,select:focus{outline:none;border-color:var(--accent)}
  .grid2{display:grid;grid-template-columns:1fr 1fr;gap:10px}
  .grid3{display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px}
  button{cursor:pointer;border:none;border-radius:9px;padding:10px 16px;font-size:13px;font-weight:600;
    background:linear-gradient(135deg,var(--accent),#7c3aed);color:#fff;transition:transform .06s,filter .2s}
  button:hover{filter:brightness(1.12)}
  button:active{transform:translateY(1px)}
  button.secondary{background:var(--panel2);color:var(--text);border:1px solid var(--border)}
  button:disabled{opacity:.5;cursor:not-allowed}
  .actions{display:flex;gap:10px;margin-top:14px;flex-wrap:wrap}
  #generateBtn{flex:1;font-size:14px;padding:12px 18px}
  .status-pill{display:inline-flex;align-items:center;gap:6px;font-size:12px;color:var(--muted)}
  .dot{width:8px;height:8px;border-radius:50%;background:var(--muted)}
  .dot.running{background:var(--warn);animation:pulse 1.2s infinite}
  .dot.done{background:var(--ok)}
  .dot.err{background:var(--err)}
  @keyframes pulse{0%,100%{opacity:1}50%{opacity:.35}}
  #progressLog{background:#0a0b11;border:1px solid var(--border);border-radius:10px;
    padding:12px 14px;font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
    font-size:12px;line-height:1.55;max-height:340px;overflow-y:auto;white-space:pre-wrap;color:#cbd1e1}
  .log-line{margin:0}
  .stage-line{color:var(--accent2)}
  .stage-line.running{color:var(--warn)}
  .stage-line.done{color:var(--ok)}
  .stage-line.err{color:var(--err)}
  .progress-bar{height:4px;background:var(--panel2);border-radius:2px;margin:6px 0 4px;overflow:hidden}
  .progress-bar > div{height:100%;background:linear-gradient(90deg,var(--accent),var(--accent2));
    width:0;transition:width .3s}
  .hint{font-size:11px;color:var(--muted);margin-top:6px}
  .result{margin-top:14px;padding:14px;background:var(--panel2);border-radius:11px;display:none}
  .result.show{display:block}
  audio{width:100%;margin-top:10px}
  .result-actions{display:flex;gap:10px;flex-wrap:wrap;margin-top:12px}
  .truncated-warn{color:var(--warn);font-size:12px;margin-top:8px}
  .err-box{background:rgba(239,68,68,.08);border:1px solid var(--err);border-radius:10px;
    padding:12px;color:#fecaca;font-size:12px;white-space:pre-wrap;margin-top:10px;display:none}
  .err-box.show{display:block}
  details summary{cursor:pointer;color:var(--muted);font-size:11px;margin-top:6px}
  .badge{display:inline-block;padding:2px 7px;border-radius:5px;font-size:10px;
    background:var(--panel2);border:1px solid var(--border);color:var(--muted);margin-left:6px}
</style>
</head>
<body>
<header>
  <div class="logo">P</div>
  <h1>PSY Music Studio</h1>
  <div class="sub">style + lyrics &rarr; symbolic plan &rarr; song</div>
</header>
<main>
  <section class="panel" id="formPanel">
    <h2>Inputs</h2>
    <label for="genrePreset">Genre Preset <span class="badge">fills style</span></label>
    <select id="genrePreset">
      <option value="">— Pick a preset (or write your own below) —</option>
      <option data-style="English, warm piano pop, expressive female voice, acoustic piano, rounded bass and light drums, lyrical memorable melody, unhurried phrasing, 88 BPM">Pop / Piano (warm, female)</option>
      <option data-style="English, intimate female jazz vocal, relaxed 88 BPM, piano, tenor saxophone, upright bass, brushed drums, no guitar, spacious modern harmony, natural English phrasing">Jazz (intimate, female)</option>
      <option data-style="English, gritty male rock vocal, driving 140 BPM, distorted electric guitar, heavy bass, powerful drums, energetic, anthemic chorus, raw edge">Rock (energetic, male)</option>
      <option data-style="English, smooth male R&amp;B vocal, 95 BPM, deep sub-bass, smooth electric piano, trap hi-hats, sensual, laid-back groove, melodic">R&amp;B / Soul (smooth, male)</option>
      <option data-style="English, soft female folk vocal, acoustic guitar fingerpicking, gentle mandolin, light percussion, warm upright bass, 76 BPM, intimate, storytelling">Folk / Acoustic (soft, female)</option>
      <option data-style="English, powerful female country vocal, 120 BPM, steel guitar, fiddle, acoustic rhythm guitar, driving bass, snare backbeat, heartfelt, twangy">Country (heartfelt, female)</option>
      <option data-style="English, ethereal female electronic vocal, 128 BPM, analog synth pads, pulsing bass, four-on-the-floor kick, shimmering arpeggios, dreamy, atmospheric">Electronic / Synthpop (dreamy, female)</option>
      <option data-style="English, aggressive male hip-hop rap, 90 BPM, deep 808 bass, crisp snare, dark piano loop, hi-hat triplets, hard-hitting, street, confident">Hip-Hop / Rap (aggressive, male)</option>
      <option data-style="Mandarin, 温柔女声, 流行, 钢琴伴奏, 圆润贝斯, 轻柔鼓点, 抒情旋律, 88 BPM">Mandarin Pop (温柔女声)</option>
      <option data-style="English, soaring orchestral vocal, 70 BPM, full string section, french horns, timpani, dramatic choir, epic, cinematic, grand, sweeping">Orchestral / Cinematic (epic)</option>
      <option data-style="English, raw male blues vocal, 100 BPM, electric guitar with overdrive, harmonica, walking bass, shuffle drums, gritty, emotional, 12-bar feel">Blues (gritty, male)</option>
      <option data-style="English, energetic female punk vocal, 170 BPM, distorted power chords, fast bass, driving drums, rebellious, raw, short and fast">Punk (energetic, female)</option>
      <option data-style="English, smooth female bossa nova vocal, 110 BPM, nylon classical guitar, light percussion, upright bass, gentle, romantic, Brazilian, swaying">Bossa Nova (romantic, female)</option>
      <option data-style="English, haunting female gothic vocal, 80 BPM, dark synth pads, deep bass, slow drums, reverb-drenched guitar, melancholic, atmospheric, moody">Gothic / Dark (moody, female)</option>
      <option data-style="English, cheerful male reggae vocal, 85 BPM, skanking guitar, organ shuffle, deep dub bass, one-drop drum pattern, sunny, laid-back, Caribbean">Reggae (sunny, male)</option>
    </select>
    <label for="style">Style / Tags <span class="badge">free-form</span></label>
    <textarea id="style" placeholder="English, warm piano pop, expressive female voice, acoustic piano, 88 BPM">English, warm piano pop, expressive female voice, acoustic piano, rounded bass and light drums, lyrical memorable melody, unhurried phrasing, 88 BPM</textarea>
    <label for="lyrics">Lyrics (use [Verse] / [Chorus] markers)</label>
    <textarea id="lyrics" placeholder="[Verse]&#10;Neon fades along the lane&#10;Footsteps keep the time of rain&#10;&#10;[Chorus]&#10;Let the day come into view&#10;Every road begins with you">[Verse]
Neon fades along the lane
Footsteps keep the time of rain
Fold the night and leave it here
Morning has a sky to clear

[Chorus]
Let the day come into view
Every road begins with you
Hold a little room for light
We will sing beyond the night</textarea>
    <div class="grid2">
      <div>
        <label for="cot">Chain-of-Thought mode</label>
        <select id="cot"></select>
      </div>
      <div>
        <label for="seed">Seed (blank = 831001)</label>
        <input id="seed" type="text" placeholder="831001"/>
      </div>
    </div>
    <div class="grid2">
      <div>
        <label for="cfgScale">CFG scale (blank = default)</label>
        <input id="cfgScale" type="text" placeholder="1.0"/>
      </div>
      <div>
        <label for="abc">External ABC score (optional)</label>
        <input id="abc" type="file" accept=".abc,.txt"/>
      </div>
    </div>

    <h2 style="margin-top:18px">Models &amp; Backend</h2>
    <div class="grid2">
      <div>
        <label for="model">Model</label>
        <select id="model"></select>
      </div>
      <div>
        <label for="vae">VAE</label>
        <select id="vae"></select>
      </div>
    </div>
    <div class="grid3">
      <div>
        <label for="backend">Backend</label>
        <select id="backend"></select>
      </div>
      <div>
        <label for="quantization">Quantization</label>
        <select id="quantization"></select>
      </div>
      <div>
        <label for="device">Device</label>
        <select id="device"></select>
      </div>
    </div>
    <div class="grid2">
      <div>
        <label for="budget">Memory budget (GiB)</label>
        <input id="budget" type="number" value="24" min="4" max="64"/>
      </div>
      <div>
        <label for="offloadAr">Offload AR to CPU</label>
        <select id="offloadAr">
          <option value="false">No</option>
          <option value="true">Yes (low-VRAM)</option>
        </select>
      </div>
    </div>
    <div class="actions">
      <button id="generateBtn">Generate Music</button>
      <button id="stopBtn" class="secondary" disabled>Stop</button>
    </div>
    <div class="hint" id="readyHint">Loading options...</div>
  </section>

  <section class="panel" id="progressPanel">
    <h2>Progress</h2>
    <div class="status-pill"><span id="statusDot" class="dot"></span><span id="statusText">Idle</span></div>
    <div class="progress-bar"><div id="progressFill"></div></div>
    <div id="progressLog">Waiting for a generation to start...</div>
    <div class="err-box" id="errBox"></div>
    <div class="result" id="resultBox">
      <h2 style="margin-top:0">Result</h2>
      <audio id="audioPlayer" controls></audio>
      <div class="truncated-warn" id="truncatedWarn" style="display:none"></div>
      <div class="result-actions">
        <button class="secondary" id="downloadBtn">Download .flac</button>
        <button class="secondary" id="openFolderBtn">Open containing folder</button>
      </div>
    </div>
  </section>
</main>

<script>
const $ = id => document.getElementById(id);
let currentJobId = null;
let eventSource = null;

async function loadOptions(){
  const r = await fetch('/api/options');
  const o = await r.json();
  fillSelect('model', o.models.map(m=>({value:m.id, label:m.label, vae:m.vae})));
  fillSelect('backend', o.backends);
  fillSelect('quantization', o.quantizations);
  fillSelect('cot', o.cot_modes);
  fillSelect('device', o.devices.map(d=>({value:d.id, label:`${d.id} ${d.name?('- '+d.name):''} ${d.vram_gib?('('+d.vram_gib+' GiB)'):''}`})));
  // Default VAE: standard listening. User can edit the input below.
  $('vae').innerHTML = '';
  ['m-a-p/YuE2-Vae','m-a-p/YuE2-Vae-legacy'].forEach(v=>{
    const opt=document.createElement('option'); opt.value=v; opt.textContent=v;
    if(v==='m-a-p/YuE2-Vae') opt.selected=true;
    $('vae').appendChild(opt);
  });
  $('readyHint').textContent = `Ready. ${o.devices.length} device(s) found. Default backend: ${o.backends.find(b=>b.default)?.id||'torch-eager'}.`;
}
function fillSelect(id, items){
  const sel=$(id); sel.innerHTML='';
  items.forEach(it=>{
    const opt=document.createElement('option');
    opt.value=it.value||it.id; opt.textContent=it.label||it.id;
    if(it.default) opt.selected=true;
    if(it.vae) opt.dataset.vae=it.vae;
    sel.appendChild(opt);
  });
}
$('model').addEventListener('change', e=>{
  const opt=e.target.selectedOptions[0];
  if(opt && opt.dataset.vae){ $('vae').value=opt.dataset.vae; }
});

function setStatus(state, text){
  $('statusDot').className='dot '+(state||'');
  $('statusText').textContent=text||'Idle';
}
function logLine(line, cls){
  const el=document.createElement('div');
  el.className='log-line'+(cls?(' '+cls):'');
  el.textContent=line;
  $('progressLog').appendChild(el);
  $('progressLog').scrollTop=$('progressLog').scrollHeight;
}
function clearProgress(){
  $('progressLog').innerHTML='';
  $('progressFill').style.width='0%';
  $('errBox').classList.remove('show');
  $('errBox').textContent='';
  $('resultBox').classList.remove('show');
  $('truncatedWarn').style.display='none';
  $('audioPlayer').src='';
}

async function startGenerate(){
  const abcFile=$('abc').files[0];
  let abcText=null;
  if(abcFile){ abcText=await abcFile.text(); }
  const payload={
    style:$('style').value,
    lyrics:$('lyrics').value,
    cot:$('cot').value,
    seed:$('seed').value||null,
    cfg_scale:$('cfgScale').value||null,
    abc:abcText,
    model:$('model').value,
    vae:$('vae').value,
    backend:$('backend').value,
    quantization:$('quantization').value,
    device:$('device').value,
    budget:parseFloat($('budget').value)||24,
    offload_ar:$('offloadAr').value==='true',
  };
  clearProgress();
  setStatus('running','Generating...');
  $('generateBtn').disabled=true;
  $('stopBtn').disabled=false;
  logLine('Submitting job...');
  try{
    const r=await fetch('/api/generate',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
    const j=await r.json();
    if(!r.ok){ throw new Error(j.error||'Failed to start job'); }
    currentJobId=j.job_id;
    openStream(j.job_id);
  }catch(e){
    setStatus('err','Error');
    $('errBox').classList.add('show');
    $('errBox').textContent=String(e);
    $('generateBtn').disabled=false;
    $('stopBtn').disabled=true;
  }
}

function openStream(jobId){
  if(eventSource) eventSource.close();
  eventSource=new EventSource(`/api/jobs/${jobId}/stream`);
  eventSource.onmessage=ev=>{
    let d; try{ d=JSON.parse(ev.data); }catch{ return; }
    handleEvent(d);
  };
  eventSource.onerror=()=>{
    setStatus('err','Stream disconnected');
  };
}

function handleEvent(d){
  if(d.type==='log'){
    logLine(d.line);
  } else if(d.type==='progress'){
    // Try to extract percent from suffix like "243 tokens | 16.2 tokens/s | elapsed 15.0s" or "3/32 steps (9%)"
    let pct=null;
    const m1=d.suffix.match(/(\d+\/\d+)\s+\S+\s+\((\d+)%\)/);
    if(m1) pct=parseInt(m1[2]);
    if(pct!==null){ $('progressFill').style.width=pct+'%'; }
    const cls = d.prefix==='Completed'?'done':(d.prefix==='Failed'||d.prefix==='Cancelled'?'err':'running');
    logLine(`[${d.prefix}] ${d.stage}: ${d.suffix}`, 'stage-line '+cls);
    if(d.prefix==='Completed') setStatus('done','Stage complete');
    else if(d.prefix==='Failed'||d.prefix==='Cancelled') setStatus('err',d.prefix);
    else setStatus('running',d.stage);
  } else if(d.type==='stage'){
    logLine(`>> ${d.name}: ${d.status}`, 'stage-line running');
    setStatus('running',d.name);
  } else if(d.type==='complete'){
    $('progressFill').style.width='100%';
    setStatus('done','Done');
    logLine(`Generation complete: ${d.audio}`,'stage-line done');
    showResult(d);
    $('generateBtn').disabled=false;
    $('stopBtn').disabled=true;
  } else if(d.type==='error'){
    setStatus('err','Failed');
    $('errBox').classList.add('show');
    $('errBox').textContent=d.message+'\n\n'+(d.traceback||'');
    $('generateBtn').disabled=false;
    $('stopBtn').disabled=true;
  } else if(d.type==='cancelled'){
    setStatus('err','Cancelled');
    logLine('>> '+d.message,'stage-line err');
    $('generateBtn').disabled=false;
    $('stopBtn').disabled=true;
  } else if(d.type==='end'){
    if(eventSource){ eventSource.close(); eventSource=null; }
  }
}

function showResult(d){
  $('audioPlayer').src=`/api/jobs/${currentJobId}/audio?t=${Date.now()}`;
  $('resultBox').classList.add('show');
  if(d.truncated && (d.truncated.abc||d.truncated.semantic)){
    $('truncatedWarn').style.display='block';
    $('truncatedWarn').textContent='Note: output was truncated (abc='+d.truncated.abc+', semantic='+d.truncated.semantic+'). Increase max_tokens in the request or shorten lyrics.';
  }
  $('downloadBtn').onclick=()=>{ window.location=`/api/jobs/${currentJobId}/download`; };
  $('openFolderBtn').onclick=async()=>{
    const r=await fetch(`/api/jobs/${currentJobId}/open-folder`);
    if(!r.ok){ const j=await r.json(); alert('Failed: '+(j.error||r.status)); }
  };
}

$('generateBtn').addEventListener('click', startGenerate);
$('genrePreset').addEventListener('change', e=>{
  const opt=e.target.selectedOptions[0];
  if(opt && opt.dataset.style){ $('style').value=opt.dataset.style; }
});
$('stopBtn').addEventListener('click', async ()=>{
  if(!currentJobId){ return; }
  $('stopBtn').disabled=true;
  setStatus('err','Stopping...');
  try{
    const r=await fetch(`/api/jobs/${currentJobId}/stop`,{method:'POST'});
    const j=await r.json();
    if(j.forced_kill){
      logLine('>> Thread killed (was stuck in a non-interruptible step)','stage-line err');
    }
    if(j.still_alive){
      logLine('>> Warning: thread still alive, may block next run','stage-line err');
    }
  }catch(e){
    logLine('>> Stop request failed: '+e,'stage-line err');
  }
  if(eventSource){ eventSource.close(); eventSource=null; }
  setStatus('err','Stopped');
  $('generateBtn').disabled=false;
});

loadOptions();
</script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
def index():
    return HTML


@app.get("/api")
def api_root():
    return {"name": "PSY Music Studio", "endpoints": ["/api/options", "/api/generate",
              "/api/jobs/{id}/stream", "/api/jobs/{id}/audio", "/api/jobs/{id}/download",
              "/api/jobs/{id}/open-folder", "/api/jobs"]}


if __name__ == "__main__":
    import argparse
    import asyncio
    import sys as _sys
    # On Windows, the default Proactor event loop raises cosmetic
    # ConnectionResetError [WinError 10054] when SSE clients close the stream.
    # The Selector loop avoids the Proactor pipe transport entirely.
    if _sys.platform.startswith("win"):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    p = argparse.ArgumentParser(description="PSY Music Studio")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=7860)
    args = p.parse_args()
    print(f"\n  PSY Music Studio:  http://{args.host}:{args.port}\n", flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
