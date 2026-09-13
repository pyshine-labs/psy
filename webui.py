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
  .grid4{display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:10px}
  .chips{display:flex;flex-wrap:wrap;gap:6px;margin-top:4px}
  .chip{display:inline-flex;align-items:center;gap:4px;padding:5px 11px;border-radius:20px;
    background:var(--panel2);border:1px solid var(--border);color:var(--muted);font-size:12px;
    cursor:pointer;user-select:none;transition:all .15s}
  .chip:hover{border-color:var(--accent);color:var(--text)}
  .chip.active{background:linear-gradient(135deg,var(--accent),#7c3aed);
    border-color:transparent;color:#fff}
  .chip .x{font-size:14px;line-height:1;margin-left:2px;opacity:.6}
  .chip.active .x{opacity:.9}
  #stylePreview{background:#0a0b11;border:1px solid var(--border);border-radius:9px;
    color:var(--text);padding:10px 12px;font-size:12px;line-height:1.5;min-height:48px;
    white-space:pre-wrap;font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
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

    <!-- Genre Preset (one-click fills everything below) -->
    <label for="genrePreset">Genre Preset <span class="badge">fills all</span></label>
    <select id="genrePreset">
      <option value="">— Pick a preset (or customise below) —</option>
      <option data-preset='{"lang":"English","gender":"female","genre":"pop","instruments":["acoustic piano","rounded bass","light drums"],"tempo":88,"mood":["warm","lyrical memorable melody","unhurried phrasing"]}'>Pop / Piano (warm, female)</option>
      <option data-preset='{"lang":"English","gender":"female","genre":"jazz","instruments":["acoustic piano","tenor saxophone","upright bass","brushed drums"],"tempo":88,"mood":["intimate","spacious modern harmony","natural phrasing"]}'>Jazz (intimate, female)</option>
      <option data-preset='{"lang":"English","gender":"male","genre":"rock","instruments":["distorted electric guitar","heavy bass","driving drums"],"tempo":140,"mood":["energetic","anthemic","raw"]}'>Rock (energetic, male)</option>
      <option data-preset='{"lang":"English","gender":"male","genre":"R&B","instruments":["deep sub-bass","smooth electric piano","trap hi-hats"],"tempo":95,"mood":["smooth","sensual","laid-back"]}'>R&B / Soul (smooth, male)</option>
      <option data-preset='{"lang":"English","gender":"female","genre":"folk","instruments":["acoustic guitar fingerpicking","mandolin","light percussion","warm upright bass"],"tempo":76,"mood":["soft","intimate","heartfelt"]}'>Folk / Acoustic (soft, female)</option>
      <option data-preset='{"lang":"English","gender":"female","genre":"country","instruments":["steel guitar","fiddle","acoustic rhythm guitar","driving bass"],"tempo":120,"mood":["powerful","heartfelt"]}'>Country (heartfelt, female)</option>
      <option data-preset='{"lang":"English","gender":"female","genre":"electronic","instruments":["analog synth pads","pulsing bass","four-on-the-floor kick","shimmering arpeggios"],"tempo":128,"mood":["ethereal","dreamy","atmospheric"]}'>Electronic / Synthpop (dreamy, female)</option>
      <option data-preset='{"lang":"English","gender":"male","genre":"hip-hop","instruments":["deep 808 bass","crisp snare","dark piano loop","hi-hat triplets"],"tempo":90,"mood":["aggressive","street","confident"]}'>Hip-Hop / Rap (aggressive, male)</option>
      <option data-preset='{"lang":"Mandarin","gender":"female","genre":"pop","instruments":["\u94a2\u7434\u4f34\u594f","\u5706\u6da6\u8d1d\u65af","\u8f7b\u67d4\u9f13\u70b9"],"tempo":88,"mood":["\u6e29\u67d4","\u6292\u60c5\u65cb\u5f8b"]}'>Mandarin Pop (\u6e29\u67d4\u5973\u58f0)</option>
      <option data-preset='{"lang":"English","gender":"female","genre":"orchestral","instruments":["full string section","french horns","timpani","dramatic choir"],"tempo":70,"mood":["soaring","epic","cinematic","grand"]}'>Orchestral / Cinematic (epic)</option>
      <option data-preset='{"lang":"English","gender":"male","genre":"blues","instruments":["electric guitar with overdrive","harmonica","walking bass","shuffle drums"],"tempo":100,"mood":["raw","gritty","heartfelt"]}'>Blues (gritty, male)</option>
      <option data-preset='{"lang":"English","gender":"female","genre":"punk","instruments":["distorted power chords","fast bass","driving drums"],"tempo":170,"mood":["energetic","rebellious","raw"]}'>Punk (energetic, female)</option>
      <option data-preset='{"lang":"English","gender":"female","genre":"bossa nova","instruments":["nylon classical guitar","light percussion","upright bass"],"tempo":110,"mood":["smooth","romantic","atmospheric"]}'>Bossa Nova (romantic, female)</option>
      <option data-preset='{"lang":"English","gender":"female","genre":"gothic","instruments":["dark synth pads","deep bass","slow drums","reverb-drenched guitar"],"tempo":80,"mood":["haunting","melancholic","atmospheric"]}'>Gothic / Dark (moody, female)</option>
      <option data-preset='{"lang":"English","gender":"male","genre":"reggae","instruments":["skanking guitar","organ shuffle","deep dub bass","one-drop drum pattern"],"tempo":85,"mood":["cheerful","laid-back"]}'>Reggae (sunny, male)</option>
    </select>

    <!-- Structured style builder -->
    <div class="grid4">
      <div>
        <label for="selLang">Language</label>
        <select id="selLang">
          <option>English</option><option>Mandarin</option><option>Spanish</option>
          <option>French</option><option>Japanese</option><option>Korean</option>
          <option>Portuguese</option><option>German</option><option>Hindi</option>
          <option>Urdu</option><option>Arabic</option><option>Russian</option>
        </select>
      </div>
      <div>
        <label for="selGender">Vocal</label>
        <select id="selGender">
          <option value="female">Female</option>
          <option value="male">Male</option>
          <option value="duet">Duet (M+F)</option>
          <option value="choir">Choir</option>
        </select>
      </div>
      <div>
        <label for="selGenre">Genre</label>
        <select id="selGenre">
          <option value="pop">Pop</option><option value="jazz">Jazz</option>
          <option value="rock">Rock</option><option value="R&B">R&B</option>
          <option value="folk">Folk</option><option value="country">Country</option>
          <option value="electronic">Electronic</option><option value="hip-hop">Hip-Hop</option>
          <option value="orchestral">Orchestral</option><option value="blues">Blues</option>
          <option value="punk">Punk</option><option value="bossa nova">Bossa Nova</option>
          <option value="gothic">Gothic</option><option value="reggae">Reggae</option>
          <option value="metal">Metal</option><option value="disco">Disco</option>
          <option value="soul">Soul</option><option value="funk">Funk</option>
          <option value="ambient">Ambient</option>
        </select>
      </div>
      <div>
        <label for="selTempo">Tempo</label>
        <select id="selTempo">
          <option value="60">60 — Ballad</option>
          <option value="70">70 — Slow</option>
          <option value="76">76 — Easy</option>
          <option value="80">80 — Mid</option>
          <option value="85">85 — Groove</option>
          <option value="88" selected>88 — Warm</option>
          <option value="95">95 — Smooth</option>
          <option value="100">100 — Upbeat</option>
          <option value="110">110 — Swing</option>
          <option value="120">120 — Pop</option>
          <option value="128">128 — Dance</option>
          <option value="140">140 — Rock</option>
          <option value="150">150 — Fast</option>
          <option value="170">170 — Punk</option>
          <option value="180">180 — Extreme</option>
        </select>
      </div>
    </div>

    <label>Instruments <span class="badge">click to toggle</span></label>
    <div class="chips" id="instrumentChips">
      <span class="chip" data-ins="acoustic piano">acoustic piano</span>
      <span class="chip" data-ins="electric guitar">electric guitar</span>
      <span class="chip" data-ins="distorted electric guitar">distorted guitar</span>
      <span class="chip" data-ins="nylon guitar">nylon guitar</span>
      <span class="chip" data-ins="steel guitar">steel guitar</span>
      <span class="chip" data-ins="upright bass">upright bass</span>
      <span class="chip" data-ins="rounded bass">rounded bass</span>
      <span class="chip" data-ins="deep sub-bass">sub-bass</span>
      <span class="chip" data-ins="808 bass">808 bass</span>
      <span class="chip" data-ins="light drums">light drums</span>
      <span class="chip" data-ins="brushed drums">brushed drums</span>
      <span class="chip" data-ins="driving drums">driving drums</span>
      <span class="chip" data-ins="trap hi-hats">trap hi-hats</span>
      <span class="chip" data-ins="string section">strings</span>
      <span class="chip" data-ins="french horns">french horns</span>
      <span class="chip" data-ins="timpani">timpani</span>
      <span class="chip" data-ins="choir">choir</span>
      <span class="chip" data-ins="tenor saxophone">tenor sax</span>
      <span class="chip" data-ins="harmonica">harmonica</span>
      <span class="chip" data-ins="fiddle">fiddle</span>
      <span class="chip" data-ins="mandolin">mandolin</span>
      <span class="chip" data-ins="synth pads">synth pads</span>
      <span class="chip" data-ins="arpeggios">arpeggios</span>
      <span class="chip" data-ins="organ">organ</span>
      <span class="chip" data-ins="marimba">marimba</span>
      <span class="chip" data-ins="trumpet">trumpet</span>
      <span class="chip" data-ins="flute">flute</span>
    </div>

    <label>Mood / Phrasing <span class="badge">click to toggle</span></label>
    <div class="chips" id="moodChips">
      <span class="chip" data-mood="warm">warm</span>
      <span class="chip" data-mood="intimate">intimate</span>
      <span class="chip" data-mood="energetic">energetic</span>
      <span class="chip" data-mood="ethereal">ethereal</span>
      <span class="chip" data-mood="melancholic">melancholic</span>
      <span class="chip" data-mood="anthemic">anthemic</span>
      <span class="chip" data-mood="dreamy">dreamy</span>
      <span class="chip" data-mood="gritty">gritty</span>
      <span class="chip" data-mood="romantic">romantic</span>
      <span class="chip" data-mood="epic">epic</span>
      <span class="chip" data-mood="raw">raw</span>
      <span class="chip" data-mood="cheerful">cheerful</span>
      <span class="chip" data-mood="haunting">haunting</span>
      <span class="chip" data-mood="sensual">sensual</span>
      <span class="chip" data-mood="aggressive">aggressive</span>
      <span class="chip" data-mood="laid-back">laid-back</span>
      <span class="chip" data-mood="unhurried phrasing">unhurried</span>
      <span class="chip" data-mood="lyrical memorable melody">lyrical melody</span>
      <span class="chip" data-mood="spacious modern harmony">spacious harmony</span>
      <span class="chip" data-mood="natural phrasing">natural phrasing</span>
      <span class="chip" data-mood="heartfelt">heartfelt</span>
      <span class="chip" data-mood="cinematic">cinematic</span>
      <span class="chip" data-mood="atmospheric">atmospheric</span>
      <span class="chip" data-mood="street">street</span>
      <span class="chip" data-mood="rebellious">rebellious</span>
    </div>

    <label for="stylePreview">Style / Tags <span class="badge">auto-generated</span></label>
    <div id="stylePreview"></div>
    <input type="hidden" id="style"/>
    <textarea id="styleEdit" placeholder="Type your own style from scratch..." style="display:none;min-height:80px"></textarea>
    <button class="secondary" id="toggleEditBtn" style="margin-top:8px;font-size:11px;padding:6px 12px">✎ Edit manually</button>

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

// ---- Style Builder ----
const GENDER_WORD = {female:'female voice', male:'male voice', duet:'duet male and female vocals', choir:'choir vocals'};
let manualEdit = false;

function buildStyle() {
  if (manualEdit) return;  // don't overwrite when user is typing manually
  const lang = $('selLang').value;
  const gender = $('selGender').value;
  const genre = $('selGenre').value;
  const tempo = $('selTempo').value;
  const instruments = [...document.querySelectorAll('#instrumentChips .chip.active')].map(c => c.dataset.ins);
  const moods = [...document.querySelectorAll('#moodChips .chip.active')].map(c => c.dataset.mood);
  const parts = [lang];
  if (GENDER_WORD[gender]) parts.push(GENDER_WORD[gender]);
  parts.push(genre);
  parts.push(...instruments);
  parts.push(...moods);
  parts.push(tempo + ' BPM');
  const styleStr = parts.join(', ');
  $('stylePreview').textContent = styleStr;
  $('style').value = styleStr;
}

// Update on any control change
['selLang','selGender','selGenre','selTempo'].forEach(id => {
  $(id).addEventListener('change', buildStyle);
});
// Chip toggling
function wireChips(containerId) {
  $(containerId).addEventListener('click', e => {
    const chip = e.target.closest('.chip');
    if (!chip) return;
    chip.classList.toggle('active');
    buildStyle();
  });
}
wireChips('instrumentChips');
wireChips('moodChips');

// Preset fills all controls
$('genrePreset').addEventListener('change', e => {
  const opt = e.target.selectedOptions[0];
  if (!opt || !opt.dataset.preset) return;
  const p = JSON.parse(opt.dataset.preset);
  $('selLang').value = p.lang;
  $('selGender').value = p.gender;
  $('selGenre').value = p.genre;
  $('selTempo').value = String(p.tempo);
  // Clear all chips first
  document.querySelectorAll('.chip.active').forEach(c => c.classList.remove('active'));
  // Activate instrument chips
  (p.instruments || []).forEach(ins => {
    const chip = document.querySelector(`#instrumentChips .chip[data-ins="${ins}"]`);
    if (chip) chip.classList.add('active');
  });
  // Activate mood chips
  (p.mood || []).forEach(m => {
    const chip = document.querySelector(`#moodChips .chip[data-mood="${m}"]`);
    if (chip) chip.classList.add('active');
  });
  buildStyle();
});

// Toggle manual edit mode
$('toggleEditBtn').addEventListener('click', () => {
  manualEdit = !manualEdit;
  if (manualEdit) {
    $('styleEdit').style.display = '';
    $('styleEdit').value = $('style').value || $('stylePreview').textContent;
    $('stylePreview').style.opacity = '0.4';
    $('toggleEditBtn').textContent = '\u2714 Use builder';
  } else {
    $('styleEdit').style.display = 'none';
    $('stylePreview').style.opacity = '';
    $('toggleEditBtn').textContent = '\u270e Edit manually';
    buildStyle();
  }
});
// When manually editing, sync hidden field
$('styleEdit').addEventListener('input', () => {
  $('style').value = $('styleEdit').value;
});

// Build initial style
buildStyle();
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
