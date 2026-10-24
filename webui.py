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
PROFILE_DIR = ROOT / "profiles"
PROFILE_DIR.mkdir(exist_ok=True)
ASSETS_DIR = ROOT / "assests"

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
            import torch

            def build_pipe(quantization, offload_ar, budget):
                return YuE2Pipeline.from_pretrained(
                    params["model"], vae=params["vae"], device=params["device"],
                    backend=params["backend"], quantization=quantization,
                    memory_budget_gib=budget, offload_ar=offload_ar,
                    progress=True,
                )

            def run_song(pipe):
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
                return song, out_dir

            on_cuda = str(params["device"]).startswith("cuda")
            if on_cuda:
                free_gib = torch.cuda.mem_get_info()[0] / 2**30
                if free_gib < 10.5 and params["quantization"] == "none":
                    _push(job_id, {"type": "log", "line":
                        f"Free VRAM is low ({free_gib:.1f} GiB). If generation runs out of "
                        "memory it will auto-retry with fp8 weights + AR modules offloaded to system RAM."})
            # First attempt uses the user's settings; on CUDA OOM the job is
            # retried with fp8 (bf16 originals kept in system RAM) and AR
            # modules offloaded to RAM instead of failing the whole job.
            attempts = [(params["quantization"], params["offload_ar"], params["budget"])]
            if on_cuda:
                fallback_q = params["quantization"]
                if fallback_q == "none":
                    try:
                        if torch.cuda.get_device_capability() >= (8, 9):
                            fallback_q = "fp8"
                    except Exception:
                        pass
                attempts.append((fallback_q, True, min(float(params["budget"]), 12.0)))
            song = out_dir = None
            for index, (quant, offload, budget) in enumerate(attempts):
                pipe = build_pipe(quant, offload, budget)
                try:
                    song, out_dir = run_song(pipe)
                    break
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    if index + 1 >= len(attempts):
                        raise RuntimeError(
                            "CUDA out of memory even with fp8 + AR offload to system RAM. "
                            "Close VRAM-heavy apps (video editor, browser) or shorten the lyrics."
                        ) from None
                    _push(job_id, {"type": "log", "line":
                        "CUDA out of memory - retrying with fp8 weights + AR modules offloaded to system RAM..."})
                finally:
                    try:
                        pipe.close()
                    except Exception:
                        pass
            audio_path = out_dir / "audio.flac"
            job["audio_path"] = str(audio_path)
            job["out_dir"] = str(out_dir)
            job["truncated"] = song.truncated
            _push(job_id, {"type": "complete", "audio": str(audio_path),
                           "truncated": song.truncated, "output_dir": str(out_dir)})
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
@app.get("/logo")
def logo():
    return FileResponse(ASSETS_DIR / "pyshine_logo.png", media_type="image/png")


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
# Profiles: named JSON snapshots of the full form state, shared across devices.
# --------------------------------------------------------------------------- #
_PROFILE_NAME_RE = re.compile(r"[A-Za-z0-9_\- \u4e00-\u9fff]{1,64}")


def _profile_path(name: str):
    name = (name or "").strip()
    if not _PROFILE_NAME_RE.fullmatch(name):
        return None
    return PROFILE_DIR / f"{name}.json"


@app.get("/api/profiles")
def list_profiles():
    files = sorted(PROFILE_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return {"profiles": [p.stem for p in files]}


@app.post("/api/profiles/{name}")
async def save_profile(name: str, request: dict):
    path = _profile_path(name)
    data = (request or {}).get("data")
    if path is None:
        return JSONResponse({"error": "Invalid profile name"}, status_code=400)
    if not isinstance(data, dict):
        return JSONResponse({"error": "Missing profile data"}, status_code=400)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"ok": True, "name": path.stem}


@app.get("/api/profiles/{name}")
def load_profile(name: str):
    path = _profile_path(name)
    if path is None or not path.is_file():
        return JSONResponse({"error": "Profile not found"}, status_code=404)
    return JSONResponse(json.loads(path.read_text(encoding="utf-8")))


@app.delete("/api/profiles/{name}")
def delete_profile(name: str):
    path = _profile_path(name)
    if path is None or not path.is_file():
        return JSONResponse({"error": "Profile not found"}, status_code=404)
    path.unlink()
    return {"ok": True}


# --------------------------------------------------------------------------- #
# GPU monitor + VRAM killer (no torch import here: keeps server VRAM-free)
# --------------------------------------------------------------------------- #
try:
    import shutil as _shutil
    _NVIDIA_SMI = _shutil.which("nvidia-smi")
except Exception:
    _NVIDIA_SMI = None


def _gpu_processes():
    """Processes currently holding GPU memory (pid, name, MiB or None).

    On Windows/WDDM drivers per-process memory is often reported as [N/A].
    """
    import subprocess
    out = subprocess.run(
        [_NVIDIA_SMI, "--query-compute-apps=pid,process_name,used_memory",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=8).stdout
    procs = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.strip().rsplit(",", 2)]
        if len(parts) != 3:
            continue
        pid_s, name, mem_s = parts
        try:
            pid = int(pid_s)
        except ValueError:
            continue
        try:
            mem = float(mem_s)
        except ValueError:
            mem = None  # [N/A] on WDDM
        procs.append({"pid": pid, "name": name or "unknown", "mem": mem})
    return procs


def _is_compute_process(name: str) -> bool:
    """True for interpreter-style compute processes we may force-kill.

    Desktop/GUI apps (explorer, browsers, overlay tools, ...) hold GPU
    memory too but must never be terminated; on Windows/WDDM the actual
    VRAM-heavy ML workers are python interpreters.
    """
    base = (name or "").replace("\\", "/").rsplit("/", 1)[-1].lower()
    return base.startswith(("python", "pythonw"))


@app.get("/api/gpu")
def gpu_stats():
    if not _NVIDIA_SMI:
        return {"available": False, "reason": "nvidia-smi not found"}
    import subprocess
    try:
        lines = subprocess.run(
            [_NVIDIA_SMI, "--query-gpu=name,memory.used,memory.total,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=8).stdout.strip().splitlines()
        if not lines:
            return {"available": False, "reason": "no GPU reported"}
        vals = [v.strip() for v in lines[0].split(",")]
        used = float(vals[1]) if len(vals) > 1 and vals[1] else 0.0
        total = float(vals[2]) if len(vals) > 2 and vals[2] else 0.0
        util = vals[3] if len(vals) > 3 and vals[3] else None
    except Exception as exc:
        return {"available": False, "reason": f"nvidia-smi failed: {exc}"}
    me = os.getpid()
    try:
        procs = _gpu_processes()
    except Exception:
        procs = []
    for p in procs:
        p["self"] = (p["pid"] == me)
        p["killable"] = _is_compute_process(p["name"])
    return {"available": True, "name": vals[0], "mem_used": used,
            "mem_total": total,
            "util": int(util) if util and util.isdigit() else None,
            "processes": procs, "self_pid": me}


@app.post("/api/gpu/kill")
def gpu_kill(nuke: bool = False):
    """Free VRAM: cancel running generations, clear this server's CUDA
    cache, kill other python compute processes. nuke=True additionally
    stops this server itself (releases its whole CUDA context; re-run
    run.bat to start again)."""
    # 1) Ask running generations to cancel (pipeline checks between tokens)
    cancelled_jobs = 0
    try:
        for j in list(JOBS.values()):
            ev = j.get("cancel")
            th = j.get("thread")
            if ev is not None and getattr(th, "is_alive", lambda: False)():
                ev.set()
                cancelled_jobs += 1
    except Exception:
        pass
    # 2) Self-cleanup: release cached CUDA blocks held by finished jobs.
    #    Only touch torch if it is already imported — importing it fresh
    #    would create a new CUDA context (~0.5 GiB) and defeat the purpose.
    freed_cache = None
    busy = False
    try:
        busy = any(getattr(j.get("thread"), "is_alive", lambda: False)()
                   for j in JOBS.values())
    except Exception:
        pass
    if not busy:
        torch = sys.modules.get("torch")
        if torch is not None:
            try:
                if torch.cuda.is_available():
                    import gc as _gc
                    before = torch.cuda.memory_reserved() / 2**20
                    _gc.collect()
                    torch.cuda.empty_cache()
                    freed_cache = round(before - torch.cuda.memory_reserved() / 2**20, 1)
            except Exception:
                freed_cache = None
    # 3) Kill other python compute processes.
    if not _NVIDIA_SMI:
        return JSONResponse({"error": "nvidia-smi not available"}, status_code=400)
    protected = {os.getpid(), os.getppid()}
    try:
        procs = _gpu_processes()
    except Exception as exc:
        return JSONResponse({"error": f"nvidia-smi failed: {exc}"}, status_code=500)
    killed, failed, skipped = [], [], []
    for p in procs:
        pid = p["pid"]
        if pid in protected:
            skipped.append({**p, "reason": "protected"})
        elif not _is_compute_process(p["name"]):
            skipped.append({**p, "reason": "not a compute process"})
        else:
            try:
                import subprocess
                if os.name == "nt":
                    subprocess.run(["taskkill", "/F", "/PID", str(pid)],
                                   capture_output=True, timeout=10, check=True)
                else:
                    os.kill(pid, 9)
                killed.append(p)
            except Exception as exc:
                failed.append({**p, "error": str(exc)[:120]})
    # 4) Nuke mode: hard-exit this server so its CUDA context is released.
    #    Delayed so the HTTP response reaches the browser first.
    if nuke:
        def _suicide():
            try:
                time.sleep(1.5)
            finally:
                os._exit(0)
        threading.Thread(target=_suicide, daemon=True).start()
    return {"killed": killed, "failed": failed, "skipped": skipped,
            "freed_cache_mib": freed_cache, "busy": busy,
            "cancelled_jobs": cancelled_jobs, "nuke": nuke}


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
  .logo{width:42px;height:42px;border-radius:9px;
    background:linear-gradient(135deg,#1e1f29,#0a0b11);overflow:hidden;
    display:flex;align-items:center;justify-content:center;flex-shrink:0}
  .logo img{width:100%;height:100%;object-fit:cover}
  main{display:grid;grid-template-columns:1fr 1fr;gap:18px;padding:18px;max-width:1500px;margin:0 auto}
  @media (max-width:980px){main{grid-template-columns:1fr}}
  .panel{background:var(--panel);border:1px solid var(--border);border-radius:14px;padding:16px}
  .panel h2{margin:0 0 12px;font-size:14px;color:var(--muted);text-transform:uppercase;letter-spacing:.8px}
  label{display:block;font-size:12px;color:var(--muted);margin:10px 0 4px}
  .hint{font-size:11px;color:var(--muted);opacity:.75;margin:3px 0 0;line-height:1.4}
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
  <div class="logo"><img src="/logo" alt="PSY Music Studio" /></div>
  <h1>PSY Music Studio</h1>
  <div class="sub" data-i18n="headerSub">style + lyrics &rarr; symbolic plan &rarr; song</div>
  <select id="uiLang" style="width:auto;flex-shrink:0" title="UI language">
    <option value="en">English</option>
    <option value="zh">中文</option>
  </select>
</header>
<main>
  <section class="panel" id="formPanel">
    <h2 data-i18n="inputs">Inputs</h2>

    <!-- Genre Preset (one-click fills everything below) -->
    <label for="genrePreset"><span data-i18n="genrePreset">Genre Preset</span> <span class="badge" data-i18n="badgeFillsAll">fills all</span></label>
    <select id="genrePreset">
      <option value="" data-i18n="presetPlaceholder">— Pick a preset (or customise below) —</option>
      <option data-zh="流行 / 钢琴（温暖女声）" data-preset='{"lang":"English","gender":"female","genre":"pop","instruments":["acoustic piano","rounded bass","light drums"],"tempo":88,"mood":["warm","lyrical memorable melody","unhurried phrasing"]}'>Pop / Piano (warm, female)</option>
      <option data-zh="爵士（亲密女声）" data-preset='{"lang":"English","gender":"female","genre":"jazz","instruments":["acoustic piano","tenor saxophone","upright bass","brushed drums"],"tempo":88,"mood":["intimate","spacious modern harmony","natural phrasing"]}'>Jazz (intimate, female)</option>
      <option data-zh="摇滚（活力男声）" data-preset='{"lang":"English","gender":"male","genre":"rock","instruments":["distorted electric guitar","rounded bass","driving drums"],"tempo":140,"mood":["energetic","anthemic","raw"]}'>Rock (energetic, male)</option>
      <option data-zh="节奏布鲁斯 / 灵魂（丝滑男声）" data-preset='{"lang":"English","gender":"male","genre":"R&B","instruments":["deep sub-bass","acoustic piano","trap hi-hats"],"tempo":95,"mood":["sensual","laid-back","warm"]}'>R&B / Soul (smooth, male)</option>
      <option data-zh="民谣 / 原声（轻柔女声）" data-preset='{"lang":"English","gender":"female","genre":"folk","instruments":["steel guitar","mandolin","light drums","upright bass"],"tempo":76,"mood":["intimate","heartfelt","warm"]}'>Folk / Acoustic (soft, female)</option>
      <option data-zh="乡村（真挚女声）" data-preset='{"lang":"English","gender":"female","genre":"country","instruments":["steel guitar","fiddle","electric guitar","light drums"],"tempo":120,"mood":["anthemic","heartfelt"]}'>Country (heartfelt, female)</option>
      <option data-zh="电子 / 合成器流行（梦幻女声）" data-preset='{"lang":"English","gender":"female","genre":"electronic","instruments":["synth pads","deep sub-bass","driving drums","arpeggios"],"tempo":128,"mood":["ethereal","dreamy","atmospheric"]}'>Electronic / Synthpop (dreamy, female)</option>
      <option data-zh="嘻哈 / 说唱（凶猛男声）" data-preset='{"lang":"English","gender":"male","genre":"hip-hop","instruments":["808 bass","trap hi-hats","acoustic piano"],"tempo":90,"mood":["aggressive","street"]}'>Hip-Hop / Rap (aggressive, male)</option>
      <option data-zh="华语流行（温柔女声）" data-preset='{"lang":"Mandarin","gender":"female","genre":"pop","instruments":["acoustic piano","rounded bass","light drums"],"tempo":88,"mood":["warm","lyrical memorable melody"]}'>Mandarin Pop (温柔女声)</option>
      <option data-zh="管弦 / 电影配乐（史诗）" data-preset='{"lang":"English","gender":"female","genre":"orchestral","instruments":["string section","french horns","timpani","choir"],"tempo":70,"mood":["epic","cinematic","anthemic"]}'>Orchestral / Cinematic (epic)</option>
      <option data-zh="布鲁斯（粗粝男声）" data-preset='{"lang":"English","gender":"male","genre":"blues","instruments":["electric guitar","harmonica","upright bass","brushed drums"],"tempo":100,"mood":["raw","gritty","heartfelt"]}'>Blues (gritty, male)</option>
      <option data-zh="朋克（活力女声）" data-preset='{"lang":"English","gender":"female","genre":"punk","instruments":["distorted electric guitar","driving drums"],"tempo":170,"mood":["energetic","rebellious","raw"]}'>Punk (energetic, female)</option>
      <option data-zh="波萨诺瓦（浪漫女声）" data-preset='{"lang":"English","gender":"female","genre":"bossa nova","instruments":["nylon guitar","light drums","upright bass"],"tempo":110,"mood":["romantic","atmospheric","laid-back"]}'>Bossa Nova (romantic, female)</option>
      <option data-zh="哥特 / 暗黑（忧郁女声）" data-preset='{"lang":"English","gender":"female","genre":"gothic","instruments":["synth pads","deep sub-bass","light drums","electric guitar"],"tempo":80,"mood":["haunting","melancholic","atmospheric"]}'>Gothic / Dark (moody, female)</option>
      <option data-zh="雷鬼（阳光男声）" data-preset='{"lang":"English","gender":"male","genre":"reggae","instruments":["electric guitar","organ","rounded bass","light drums"],"tempo":85,"mood":["cheerful","laid-back"]}'>Reggae (sunny, male)</option>
    </select>

    <!-- Song Template: loads a complete sample song (style + lyrics) -->
    <label for="songTemplate"><span data-i18n="labelSongTpl">Song Template</span> <span class="badge" data-i18n="badgeSampleLyrics">sample lyrics</span></label>
    <select id="songTemplate">
      <option value="" data-i18n="tplPlaceholder">— Load a full sample song into Style + Lyrics —</option>
      <option value="pop" data-zh="流行 · 霓虹夜（女声）">Pop · Neon Nights (female)</option>
      <option value="rock" data-zh="摇滚 · 不屈（男声）">Rock · Unbroken (male)</option>
      <option value="hiphop" data-zh="嘻哈 · 崛起（男声）">Hip-Hop · Rise (male)</option>
      <option value="edm" data-zh="电子 · 复活夜（女声）">EDM · Come Alive (female)</option>
      <option value="ballad" data-zh="抒情 · 静静时光（女声钢琴）">Ballad · Quiet Hours (female piano)</option>
      <option value="mandarin" data-zh="华语流行 · 把歌唱给你听">华语流行 · 把歌唱给你听 (Mandarin)</option>
      <option value="jazz" data-zh="爵士 · 再跳一曲（女声）">Jazz · One More Dance (female)</option>
      <option value="cinematic" data-zh="史诗 · 黎明（合唱）">Cinematic · Rise of Dawn (choir)</option>
      <option value="duet" data-zh="对唱 · 并肩同行（男女）">Duet · Side by Side (male + female)</option>
    </select>

    <!-- Structured style builder -->
    <div class="grid4">
      <div>
        <label for="selLang" data-i18n="labelLang">Language</label>
        <select id="selLang">
          <option value="English" data-zh="英语">English</option>
          <option value="Mandarin" data-zh="普通话">Mandarin</option>
          <option value="Spanish" data-zh="西班牙语">Spanish</option>
          <option value="French" data-zh="法语">French</option>
          <option value="Japanese" data-zh="日语">Japanese</option>
          <option value="Korean" data-zh="韩语">Korean</option>
          <option value="Portuguese" data-zh="葡萄牙语">Portuguese</option>
          <option value="German" data-zh="德语">German</option>
          <option value="Hindi" data-zh="印地语">Hindi</option>
          <option value="Urdu" data-zh="乌尔都语">Urdu</option>
          <option value="Arabic" data-zh="阿拉伯语">Arabic</option>
          <option value="Russian" data-zh="俄语">Russian</option>
        </select>
        <p class="hint" data-i18n="hintLang">Language the model sings in</p>
      </div>
      <div>
        <label for="selGender" data-i18n="labelVocal">Vocal</label>
        <select id="selGender">
          <option value="female" data-i18n="vocalFemale">Female</option>
          <option value="male" data-i18n="vocalMale">Male</option>
          <option value="duet" data-i18n="vocalDuet">Duet (M+F)</option>
          <option value="choir" data-i18n="vocalChoir">Choir</option>
        </select>
        <p class="hint" data-i18n="hintVocal">Who sings: solo female / male, duet or choir</p>
      </div>
      <div>
        <label for="selGenre" data-i18n="labelGenre">Genre</label>
        <select id="selGenre">
          <option value="pop" data-zh="流行">Pop</option><option value="jazz" data-zh="爵士">Jazz</option>
          <option value="rock" data-zh="摇滚">Rock</option><option value="R&B" data-zh="节奏布鲁斯">R&B</option>
          <option value="folk" data-zh="民谣">Folk</option><option value="country" data-zh="乡村">Country</option>
          <option value="electronic" data-zh="电子">Electronic</option><option value="hip-hop" data-zh="嘻哈">Hip-Hop</option>
          <option value="orchestral" data-zh="管弦">Orchestral</option><option value="blues" data-zh="布鲁斯">Blues</option>
          <option value="punk" data-zh="朋克">Punk</option><option value="bossa nova" data-zh="波萨诺瓦">Bossa Nova</option>
          <option value="gothic" data-zh="哥特">Gothic</option><option value="reggae" data-zh="雷鬼">Reggae</option>
          <option value="metal" data-zh="金属">Metal</option><option value="disco" data-zh="迪斯科">Disco</option>
          <option value="soul" data-zh="灵魂">Soul</option><option value="funk" data-zh="放克">Funk</option>
          <option value="ambient" data-zh="氛围">Ambient</option>
        </select>
        <p class="hint" data-i18n="hintGenre">Overall music style of the song</p>
      </div>
      <div>
        <label for="selTempo" data-i18n="labelTempo">Tempo</label>
        <select id="selTempo">
          <option value="60" data-zh="60 — 叙事曲">60 — Ballad</option>
          <option value="70" data-zh="70 — 慢速">70 — Slow</option>
          <option value="76" data-zh="76 — 轻松">76 — Easy</option>
          <option value="80" data-zh="80 — 中速">80 — Mid</option>
          <option value="85" data-zh="85 — 律动">85 — Groove</option>
          <option value="88" selected data-zh="88 — 温暖">88 — Warm</option>
          <option value="95" data-zh="95 — 丝滑">95 — Smooth</option>
          <option value="100" data-zh="100 — 轻快">100 — Upbeat</option>
          <option value="110" data-zh="110 — 摇摆">110 — Swing</option>
          <option value="120" data-zh="120 — 流行">120 — Pop</option>
          <option value="128" data-zh="128 — 舞曲">128 — Dance</option>
          <option value="140" data-zh="140 — 摇滚">140 — Rock</option>
          <option value="150" data-zh="150 — 快速">150 — Fast</option>
          <option value="170" data-zh="170 — 朋克">170 — Punk</option>
          <option value="180" data-zh="180 — 极限">180 — Extreme</option>
        </select>
        <p class="hint" data-i18n="hintTempo">Beats per minute — ballads 60–80, pop 88–120, dance ~128, rock 130–150, punk 160–180</p>
      </div>
    </div>

    <label><span data-i18n="labelInstruments">Instruments</span> <span class="badge" data-i18n="badgeToggle">click to toggle</span></label>
    <p class="hint" data-i18n="hintInstr">Pick 2–4 instruments; English tags work best</p>
    <div class="chips" id="instrumentChips">
      <span class="chip" data-zh="钢琴" data-ins="acoustic piano">acoustic piano</span>
      <span class="chip" data-zh="电吉他" data-ins="electric guitar">electric guitar</span>
      <span class="chip" data-zh="失真吉他" data-ins="distorted electric guitar">distorted guitar</span>
      <span class="chip" data-zh="尼龙弦吉他" data-ins="nylon guitar">nylon guitar</span>
      <span class="chip" data-zh="钢弦吉他" data-ins="steel guitar">steel guitar</span>
      <span class="chip" data-zh="低音提琴" data-ins="upright bass">upright bass</span>
      <span class="chip" data-zh="圆润贝斯" data-ins="rounded bass">rounded bass</span>
      <span class="chip" data-zh="低音贝斯" data-ins="deep sub-bass">sub-bass</span>
      <span class="chip" data-zh="808贝斯" data-ins="808 bass">808 bass</span>
      <span class="chip" data-zh="轻鼓点" data-ins="light drums">light drums</span>
      <span class="chip" data-zh="刷鼓" data-ins="brushed drums">brushed drums</span>
      <span class="chip" data-zh="强劲鼓点" data-ins="driving drums">driving drums</span>
      <span class="chip" data-zh="嘻哈镲片" data-ins="trap hi-hats">trap hi-hats</span>
      <span class="chip" data-zh="弦乐组" data-ins="string section">strings</span>
      <span class="chip" data-zh="圆号" data-ins="french horns">french horns</span>
      <span class="chip" data-zh="定音鼓" data-ins="timpani">timpani</span>
      <span class="chip" data-zh="合唱" data-ins="choir">choir</span>
      <span class="chip" data-zh="次中音萨克斯" data-ins="tenor saxophone">tenor sax</span>
      <span class="chip" data-zh="口琴" data-ins="harmonica">harmonica</span>
      <span class="chip" data-zh="小提琴" data-ins="fiddle">fiddle</span>
      <span class="chip" data-zh="曼陀林" data-ins="mandolin">mandolin</span>
      <span class="chip" data-zh="合成器铺底" data-ins="synth pads">synth pads</span>
      <span class="chip" data-zh="琶音" data-ins="arpeggios">arpeggios</span>
      <span class="chip" data-zh="风琴" data-ins="organ">organ</span>
      <span class="chip" data-zh="马林巴" data-ins="marimba">marimba</span>
      <span class="chip" data-zh="小号" data-ins="trumpet">trumpet</span>
      <span class="chip" data-zh="长笛" data-ins="flute">flute</span>
    </div>

    <label><span data-i18n="labelMood">Mood / Phrasing</span> <span class="badge" data-i18n="badgeToggle">click to toggle</span></label>
    <p class="hint" data-i18n="hintMood">Feeling and singing style of the performance</p>
    <div class="chips" id="moodChips">
      <span class="chip" data-zh="温暖" data-mood="warm">warm</span>
      <span class="chip" data-zh="亲密" data-mood="intimate">intimate</span>
      <span class="chip" data-zh="活力" data-mood="energetic">energetic</span>
      <span class="chip" data-zh="空灵" data-mood="ethereal">ethereal</span>
      <span class="chip" data-zh="忧郁" data-mood="melancholic">melancholic</span>
      <span class="chip" data-zh="颂歌式" data-mood="anthemic">anthemic</span>
      <span class="chip" data-zh="梦幻" data-mood="dreamy">dreamy</span>
      <span class="chip" data-zh="粗粝" data-mood="gritty">gritty</span>
      <span class="chip" data-zh="浪漫" data-mood="romantic">romantic</span>
      <span class="chip" data-zh="史诗" data-mood="epic">epic</span>
      <span class="chip" data-zh="原始" data-mood="raw">raw</span>
      <span class="chip" data-zh="欢快" data-mood="cheerful">cheerful</span>
      <span class="chip" data-zh="萦绕" data-mood="haunting">haunting</span>
      <span class="chip" data-zh="感性" data-mood="sensual">sensual</span>
      <span class="chip" data-zh="凶猛" data-mood="aggressive">aggressive</span>
      <span class="chip" data-zh="慵懒" data-mood="laid-back">laid-back</span>
      <span class="chip" data-zh="从容" data-mood="unhurried phrasing">unhurried</span>
      <span class="chip" data-zh="抒情旋律" data-mood="lyrical memorable melody">lyrical melody</span>
      <span class="chip" data-zh="宽阔和声" data-mood="spacious modern harmony">spacious harmony</span>
      <span class="chip" data-zh="自然唱腔" data-mood="natural phrasing">natural phrasing</span>
      <span class="chip" data-zh="真挚" data-mood="heartfelt">heartfelt</span>
      <span class="chip" data-zh="电影感" data-mood="cinematic">cinematic</span>
      <span class="chip" data-zh="氛围感" data-mood="atmospheric">atmospheric</span>
      <span class="chip" data-zh="街头" data-mood="street">street</span>
      <span class="chip" data-zh="反叛" data-mood="rebellious">rebellious</span>
    </div>

    <label for="stylePreview"><span data-i18n="labelStyle">Style / Tags</span> <span class="badge" data-i18n="badgeAuto">auto-generated</span></label>
    <p class="hint" data-i18n="hintStyle">Auto-built from the options above — click the ✏️ button to edit it manually</p>
    <div id="stylePreview"></div>
    <input type="hidden" id="style"/>
    <textarea id="styleEdit" data-i18n-ph="phStyleEdit" placeholder="Type your own style from scratch..." style="display:none;min-height:80px"></textarea>
    <button class="secondary" id="toggleEditBtn" style="margin-top:8px;font-size:11px;padding:6px 12px">✎ Edit manually</button>

    <label for="lyrics" data-i18n="labelLyrics">Lyrics (use [Verse] / [Chorus] markers)</label>
    <p class="hint" data-i18n="hintLyrics">Structure the song with markers: [Verse] / [Chorus] / [Bridge] / [Outro] — try a Song Template for a full example</p>
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
        <label for="cot" data-i18n="labelCot">Chain-of-Thought mode</label>
        <select id="cot"></select>
      </div>
      <div>
        <label for="seed" data-i18n="labelSeed">Seed (blank = 831001)</label>
        <input id="seed" type="text" placeholder="831001"/>
      </div>
    </div>
    <div class="grid2">
      <div>
        <label for="cfgScale" data-i18n="labelCfg">CFG scale (blank = default)</label>
        <input id="cfgScale" type="text" placeholder="1.0"/>
      </div>
      <div>
        <label for="abc" data-i18n="labelAbc">External ABC score (optional)</label>
        <input id="abc" type="file" accept=".abc,.txt"/>
      </div>
    </div>

    <h2 style="margin-top:18px" data-i18n="hModels">Models &amp; Backend</h2>
    <div class="grid2">
      <div>
        <label for="model" data-i18n="labelModel">Model</label>
        <select id="model"></select>
      </div>
      <div>
        <label for="vae" data-i18n="labelVae">VAE</label>
        <select id="vae"></select>
      </div>
    </div>
    <div class="grid3">
      <div>
        <label for="backend" data-i18n="labelBackend">Backend</label>
        <select id="backend"></select>
      </div>
      <div>
        <label for="quantization" data-i18n="labelQuant">Quantization</label>
        <select id="quantization"></select>
      </div>
      <div>
        <label for="device" data-i18n="labelDevice">Device</label>
        <select id="device"></select>
      </div>
    </div>
    <div class="grid2">
      <div>
        <label for="budget" data-i18n="labelBudget">Memory budget (GiB)</label>
        <input id="budget" type="number" value="24" min="4" max="64"/>
      </div>
      <div>
        <label for="offloadAr" data-i18n="labelOffload">Offload AR to CPU</label>
        <select id="offloadAr">
          <option value="false" data-i18n="optNo">No</option>
          <option value="true" data-i18n="optYesLowVram">Yes (low-VRAM)</option>
        </select>
      </div>
    </div>

    <h2 style="margin-top:18px" data-i18n="hProfile">Profile save / load</h2>
    <div class="grid2">
      <div>
        <input id="profileName" type="text" data-i18n-ph="phProfileName" placeholder="Profile name"/>
      </div>
      <div>
        <select id="profileList"></select>
      </div>
    </div>
    <div class="grid3" style="margin-top:8px">
      <button class="secondary" id="profileSaveBtn" data-i18n="btnSave">Save</button>
      <button class="secondary" id="profileLoadBtn" data-i18n="btnLoad">Load</button>
      <button class="secondary" id="profileDeleteBtn" data-i18n="btnDelete">Delete</button>
    </div>

    <h2 style="margin-top:18px" data-i18n="hGpu">GPU Monitor</h2>
    <div class="hint" id="gpuName">—</div>
    <div class="progress-bar" id="gpuBar" style="height:8px;margin-top:6px"><div id="gpuBarFill" style="width:0%"></div></div>
    <div class="grid2" style="margin-top:4px">
      <div class="hint" id="gpuMemText">—</div>
      <div class="hint" id="gpuUtilText" style="text-align:right">—</div>
    </div>
    <div id="gpuProcList" class="hint" style="margin-top:6px"></div>
    <button class="secondary" id="gpuKillBtn" style="margin-top:8px" data-i18n="btnKillGpu">Free VRAM (kill GPU processes)</button>
    <button class="secondary" id="gpuNukeBtn" style="margin-top:6px;border-color:#ef4444;color:#f87171" data-i18n="btnNukeGpu">Nuke all VRAM (stops this app)</button>
    <p class="hint" data-i18n="hintKillGpu">Ends other python compute processes holding GPU memory (desktop apps are never touched; this WebUI is protected).</p>

    <div class="actions">
      <button id="generateBtn" data-i18n="btnGenerate">Generate Music</button>
      <button id="stopBtn" class="secondary" disabled data-i18n="btnStop">Stop</button>
    </div>
    <div class="hint" id="readyHint" data-i18n="loadingOptions">Loading options...</div>
  </section>

  <section class="panel" id="progressPanel">
    <h2 data-i18n="hProgress">Progress</h2>
    <div class="status-pill"><span id="statusDot" class="dot"></span><span id="statusText" data-i18n="idle">Idle</span></div>
    <div class="progress-bar"><div id="progressFill"></div></div>
    <div id="progressLog" data-i18n="waitingLog">Waiting for a generation to start...</div>
    <div class="err-box" id="errBox"></div>
    <div class="result" id="resultBox">
      <h2 style="margin-top:0" data-i18n="hResult">Result</h2>
      <audio id="audioPlayer" controls></audio>
      <div class="truncated-warn" id="truncatedWarn" style="display:none"></div>
      <div class="result-actions">
        <button class="secondary" id="downloadBtn" data-i18n="btnDownload">Download .flac</button>
        <button class="secondary" id="openFolderBtn" data-i18n="btnOpenFolder">Open containing folder</button>
      </div>
    </div>
  </section>
</main>

<script>
const $ = id => document.getElementById(id);
let currentJobId = null;
let eventSource = null;
let uiLang = 'en';

// ---- i18n: add more languages by adding another key here (e.g. es: {...}) ----
const I18N = {
  en: {
    headerSub:'style + lyrics \u2192 symbolic plan \u2192 song',
    inputs:'Inputs', genrePreset:'Genre Preset', badgeFillsAll:'fills all',
    presetPlaceholder:'\u2014 Pick a preset (or customise below) \u2014',
    labelSongTpl:'Song Template', badgeSampleLyrics:'sample lyrics',
    tplPlaceholder:'\u2014 Load a full sample song into Style + Lyrics \u2014',
    labelLang:'Language', labelVocal:'Vocal', labelGenre:'Genre', labelTempo:'Tempo',
    vocalFemale:'Female', vocalMale:'Male', vocalDuet:'Duet (M+F)', vocalChoir:'Choir',
    labelInstruments:'Instruments', badgeToggle:'click to toggle',
    labelMood:'Mood / Phrasing', labelStyle:'Style / Tags', badgeAuto:'auto-generated',
    phStyleEdit:'Type your own style from scratch...',
    btnEditManually:'\u270e Edit manually', btnUseBuilder:'\u2714 Use builder',
    labelLyrics:'Lyrics (use [Verse] / [Chorus] markers)',
    hintLang:'Language the model sings in', hintVocal:'Who sings: solo female / male, duet or choir',
    hintGenre:'Overall music style of the song',
    hintTempo:'Beats per minute — ballads 60–80, pop 88–120, dance ~128, rock 130–150, punk 160–180',
    hintInstr:'Pick 2\u20134 instruments; English tags work best',
    hintMood:'Feeling and singing style of the performance',
    hintStyle:'Auto-built from the options above \u2014 click the \u270f\ufe0f button to edit it manually',
    hintLyrics:'Structure the song with markers: [Verse] / [Chorus] / [Bridge] / [Outro] \u2014 try a Song Template for a full example',
    hGpu:'GPU Monitor', btnKillGpu:'Free VRAM (kill GPU processes)',
    hintKillGpu:'Ends other python compute processes holding GPU memory (desktop apps are never touched; this WebUI is protected).',
    gpuUsed:'used', gpuUtil:'GPU util', gpuNoProcs:'No processes using the GPU',
    gpuSelf:'this app', gpuKillConfirm:'Cancel any running generation, free this app\u2019s VRAM cache, and kill other python GPU processes?',
    gpuKilled:'Killed:', gpuFailed:'failed:', gpuUnavailable:'GPU stats unavailable',
    gpuFreed:'VRAM cache freed:', gpuCancelled:'Generation(s) cancelled:', gpuNothingToKill:'No other python GPU processes found \u2014 remaining VRAM is held by desktop apps (close them manually).',
    btnNukeGpu:'Nuke all VRAM (stops this app)',
    gpuNukeConfirm:'This cancels the running song, frees ALL VRAM held by this WebUI (the app will close), and kills other python GPU processes. Re-run run.bat afterwards. Continue?',
    gpuNukeMsg:'All VRAM released. Re-run run.bat to start the WebUI again.',
    labelCot:'Chain-of-Thought mode', labelSeed:'Seed (blank = 831001)',
    labelCfg:'CFG scale (blank = default)', labelAbc:'External ABC score (optional)',
    hModels:'Models & Backend', labelModel:'Model', labelVae:'VAE', labelBackend:'Backend',
    labelQuant:'Quantization', labelDevice:'Device', labelBudget:'Memory budget (GiB)',
    labelOffload:'Offload AR to CPU', optNo:'No', optYesLowVram:'Yes (low-VRAM)',
    hProfile:'Profile save / load', phProfileName:'Profile name',
    profilePlaceholder:'\u2014 saved profiles \u2014',
    btnSave:'Save', btnLoad:'Load', btnDelete:'Delete',
    btnGenerate:'Generate Music', btnStop:'Stop',
    loadingOptions:'Loading options...',
    hProgress:'Progress', idle:'Idle', waitingLog:'Waiting for a generation to start...',
    hResult:'Result', btnDownload:'Download .flac', btnOpenFolder:'Open containing folder',
    generating:'Generating...', error:'Error', done:'Done', failed:'Failed',
    cancelled:'Cancelled', stopping:'Stopping...', stopped:'Stopped',
    streamDisconnected:'Stream disconnected', stageComplete:'Stage complete',
    submitting:'Submitting job...', completeLog:'Generation complete: ',
    readyHint:'Ready. {n} device(s) found. Default backend: {b}.',
    threadKilled:'>> Thread killed (was stuck in a non-interruptible step)',
    threadAlive:'>> Warning: thread still alive, may block next run',
    stopFailed:'>> Stop request failed: ', alertFailed:'Failed: ',
    truncatedWarn:'Note: output was truncated (abc={abc}, semantic={sem}). Increase max_tokens in the request or shorten lyrics.',
    profileNameRequired:'Enter a profile name first',
    profilePickFirst:'Pick a profile from the list first',
    saveFailed:'Save failed', loadFailed:'Load failed',
    confirmDelete:"Delete profile '{name}'?",
  },
  zh: {
    headerSub:'风格 + 歌词 \u2192 符号规划 \u2192 歌曲',
    inputs:'输入', genrePreset:'流派预设', badgeFillsAll:'一键填充',
    presetPlaceholder:'\u2014 选择预设（或在下方自定义）\u2014',
    labelSongTpl:'歌曲模板', badgeSampleLyrics:'示例歌词',
    tplPlaceholder:'\u2014 加载完整示例歌曲（风格 + 歌词）\u2014',
    labelLang:'语言', labelVocal:'人声', labelGenre:'流派', labelTempo:'速度',
    vocalFemale:'女声', vocalMale:'男声', vocalDuet:'对唱（男+女）', vocalChoir:'合唱',
    labelInstruments:'乐器', badgeToggle:'点击选择',
    labelMood:'情绪 / 演唱', labelStyle:'风格 / 标签', badgeAuto:'自动生成',
    phStyleEdit:'从头输入你的自定义风格...',
    btnEditManually:'\u270e 手动编辑', btnUseBuilder:'\u2714 使用构建器',
    labelLyrics:'歌词（使用 [Verse] / [Chorus] 标记）',
    hintLang:'模型演唱的语言', hintVocal:'演唱者：女声 / 男声独唱、对唱或合唱',
    hintGenre:'歌曲的整体音乐风格',
    hintTempo:'每分钟节拍：叙事曲 60\u201380，流行 88\u2013120，舞曲约 128，摇滚 130\u2013150，朋克 160\u2013180',
    hintInstr:'选择 2\u20134 种乐器；英文标签效果最佳',
    hintMood:'演唱的情绪与风格',
    hintStyle:'由上方选项自动生成 \u2014 点击 \u270f\ufe0f 按钮可手动编辑',
    hintLyrics:'使用 [Verse] / [Chorus] / [Bridge] / [Outro] 标记构建歌曲段落 \u2014 可尝试歌曲模板查看完整示例',
    hGpu:'GPU 监控', btnKillGpu:'一键释放显存（结束 GPU 进程）',
    hintKillGpu:'结束其他占用显存的 python 计算进程（桌面应用不受影响；本 WebUI 受保护）。',
    gpuUsed:'已用', gpuUtil:'GPU 利用率', gpuNoProcs:'当前没有进程占用 GPU',
    gpuSelf:'本应用', gpuKillConfirm:'中止正在生成的歌曲、释放本应用显存缓存，并结束其他 python GPU 进程？',
    gpuKilled:'已结束：', gpuFailed:'失败：', gpuUnavailable:'无法获取 GPU 信息',
    gpuCancelled:'已中止生成任务：', gpuFreed:'显存缓存已释放：', gpuNothingToKill:'未发现其他 python GPU 进程 \u2014 剩余显存由桌面应用占用（请手动关闭）。',
    btnNukeGpu:'彻底释放显存（会关闭本应用）',
    gpuNukeConfirm:'将中止正在生成的歌曲、释放本 WebUI 占用的全部显存（应用会关闭），并结束其他 python GPU 进程。之后重新运行 run.bat 即可。继续？',
    gpuNukeMsg:'显存已全部释放。重新运行 run.bat 可再次启动 WebUI。',
    labelCot:'思维链模式', labelSeed:'随机种子（留空 = 831001）',
    labelCfg:'CFG 强度（留空 = 默认）', labelAbc:'外部 ABC 乐谱（可选）',
    hModels:'模型与后端', labelModel:'模型', labelVae:'VAE 解码器', labelBackend:'后端',
    labelQuant:'量化', labelDevice:'设备', labelBudget:'显存预算 (GiB)',
    labelOffload:'AR 卸载到内存', optNo:'否', optYesLowVram:'是（低显存）',
    hProfile:'配置保存 / 加载', phProfileName:'配置名称',
    profilePlaceholder:'\u2014 已保存的配置 \u2014',
    btnSave:'保存', btnLoad:'加载', btnDelete:'删除',
    btnGenerate:'生成音乐', btnStop:'停止',
    loadingOptions:'正在加载选项...',
    hProgress:'进度', idle:'空闲', waitingLog:'等待生成任务开始...',
    hResult:'结果', btnDownload:'下载 .flac', btnOpenFolder:'打开所在文件夹',
    generating:'生成中...', error:'错误', done:'完成', failed:'失败',
    cancelled:'已取消', stopping:'正在停止...', stopped:'已停止',
    streamDisconnected:'连接已断开', stageComplete:'阶段完成',
    submitting:'正在提交任务...', completeLog:'生成完成：',
    readyHint:'就绪。找到 {n} 个设备。默认后端：{b}。',
    threadKilled:'>> 线程已强制终止（卡在不可中断的步骤）',
    threadAlive:'>> 警告：线程仍存活，可能阻塞下一次运行',
    stopFailed:'>> 停止请求失败：', alertFailed:'失败：',
    truncatedWarn:'注意：输出被截断（abc={abc}，semantic={sem}）。请增大请求中的 max_tokens 或缩短歌词。',
    profileNameRequired:'请先输入配置名称',
    profilePickFirst:'请先从列表选择一个配置',
    saveFailed:'保存失败', loadFailed:'加载失败',
    confirmDelete:"确定删除配置“{name}”？",
  },
};
function t(key){
  return (I18N[uiLang] && I18N[uiLang][key]) || I18N.en[key] || key;
}
// Elements that show Chinese text when uiLang=zh but keep their English
// value/model tag (chips, song-language / tempo / genre / preset options).
function applyDataZh(){
  document.querySelectorAll('[data-zh]').forEach(el=>{
    if(el.dataset.en === undefined) el.dataset.en = el.textContent;
    el.textContent = uiLang === 'zh' ? el.dataset.zh : el.dataset.en;
  });
}
// Options filled from /api/options (cot / quantization). Backend & device
// intentionally stay English (technical terms, per compatibility).
const OPT_I18N = {
  cot: { full:'完整（和弦+旋律 ABC，默认）', melody:'仅旋律 ABC', off:'关闭（不生成 ABC）' },
  quantization: { none:'无（bfloat16）', fp8:'fp8（更低显存）' },
};
function applySelectOptionLang(id){
  const map = OPT_I18N[id];
  if(!map) return;
  [...$(id).options].forEach(opt=>{
    opt.textContent = uiLang === 'zh' ? (map[opt.value] || opt.dataset.labelEn) : opt.dataset.labelEn;
  });
}
function applyLang(lang){
  uiLang = I18N[lang] ? lang : 'en';
  document.documentElement.lang = uiLang === 'zh' ? 'zh-CN' : 'en';
  document.querySelectorAll('[data-i18n]').forEach(el=>{
    const v = I18N[uiLang][el.dataset.i18n];
    if(v !== undefined) el.textContent = v;
  });
  document.querySelectorAll('[data-i18n-ph]').forEach(el=>{
    const v = I18N[uiLang][el.dataset.i18nPh];
    if(v !== undefined) el.placeholder = v;
  });
  applyDataZh();
  ['cot','quantization'].forEach(applySelectOptionLang);
  $('toggleEditBtn').textContent = manualEdit ? t('btnUseBuilder') : t('btnEditManually');
  const phOpt = $('profileList').querySelector('option[value=""]');
  if(phOpt) phOpt.textContent = t('profilePlaceholder');
  localStorage.setItem('yue_ui_lang', uiLang);
}
$('uiLang').addEventListener('change', e => applyLang(e.target.value));

// ---- Form state: collect / apply (used by profiles + autosave) ----
function collectState(){
  return {
    uiLang, lang: $('selLang').value, gender: $('selGender').value,
    genre: $('selGenre').value, tempo: $('selTempo').value,
    instruments: [...document.querySelectorAll('#instrumentChips .chip.active')].map(c=>c.dataset.ins),
    moods: [...document.querySelectorAll('#moodChips .chip.active')].map(c=>c.dataset.mood),
    manualEdit, style: $('style').value, styleEdit: $('styleEdit').value,
    lyrics: $('lyrics').value, cot: $('cot').value,
    seed: $('seed').value, cfgScale: $('cfgScale').value,
    model: $('model').value, vae: $('vae').value, backend: $('backend').value,
    quantization: $('quantization').value, device: $('device').value,
    budget: $('budget').value, offloadAr: $('offloadAr').value,
  };
}
function applyState(s){
  if(!s || typeof s !== 'object') return;
  if(s.uiLang && I18N[s.uiLang]){ $('uiLang').value = s.uiLang; applyLang(s.uiLang); }
  if(s.lang) $('selLang').value = s.lang;
  if(s.gender) $('selGender').value = s.gender;
  if(s.genre) $('selGenre').value = s.genre;
  if(s.tempo) $('selTempo').value = String(s.tempo);
  document.querySelectorAll('.chip.active').forEach(c=>c.classList.remove('active'));
  (s.instruments||[]).forEach(ins=>{
    const c = document.querySelector(`#instrumentChips .chip[data-ins="${ins}"]`);
    if(c) c.classList.add('active');
  });
  (s.moods||[]).forEach(m=>{
    const c = document.querySelector(`#moodChips .chip[data-mood="${m}"]`);
    if(c) c.classList.add('active');
  });
  if(typeof s.manualEdit === 'boolean' && manualEdit !== s.manualEdit) $('toggleEditBtn').click();
  if(s.styleEdit !== undefined) $('styleEdit').value = s.styleEdit;
  if(s.style !== undefined) $('style').value = s.style;
  if(s.manualEdit) $('stylePreview').textContent = s.style || '';
  else buildStyle();
  if(s.lyrics !== undefined) $('lyrics').value = s.lyrics;
  if(s.cot) $('cot').value = s.cot;
  if(s.seed !== undefined && s.seed !== null) $('seed').value = s.seed;
  if(s.cfgScale !== undefined && s.cfgScale !== null) $('cfgScale').value = s.cfgScale;
  if(s.model){
    $('model').value = s.model;
    const opt = $('model').selectedOptions[0];
    if(opt && opt.dataset.vae && !s.vae) $('vae').value = opt.dataset.vae;
  }
  if(s.vae) $('vae').value = s.vae;
  if(s.backend) $('backend').value = s.backend;
  if(s.quantization) $('quantization').value = s.quantization;
  if(s.device) $('device').value = s.device;
  if(s.budget) $('budget').value = s.budget;
  if(s.offloadAr) $('offloadAr').value = String(s.offloadAr);
}

// ---- Server-side profiles ----
async function refreshProfiles(){
  try{
    const r = await fetch('/api/profiles');
    const j = await r.json();
    const sel = $('profileList');
    sel.innerHTML = '';
    const ph = document.createElement('option');
    ph.value = ''; ph.textContent = t('profilePlaceholder');
    sel.appendChild(ph);
    (j.profiles||[]).forEach(n=>{
      const o = document.createElement('option');
      o.value = n; o.textContent = n;
      sel.appendChild(o);
    });
  }catch{}
}
$('profileSaveBtn').addEventListener('click', async ()=>{
  const name = $('profileName').value.trim();
  if(!name){ alert(t('profileNameRequired')); return; }
  try{
    const r = await fetch('/api/profiles/'+encodeURIComponent(name),{
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({data: collectState()})
    });
    if(!r.ok){ const j = await r.json(); alert(t('saveFailed') + ': ' + (j.error||r.status)); return; }
    await refreshProfiles();
    $('profileList').value = name;
  }catch(e){ alert(t('saveFailed') + ': ' + e); }
});
$('profileLoadBtn').addEventListener('click', async ()=>{
  const name = $('profileList').value;
  if(!name){ alert(t('profilePickFirst')); return; }
  try{
    const r = await fetch('/api/profiles/'+encodeURIComponent(name));
    if(!r.ok){ alert(t('loadFailed')); return; }
    const j = await r.json();
    applyState(j.data || j);
    localStorage.setItem('yue_last_state', JSON.stringify(collectState()));
  }catch(e){ alert(t('loadFailed') + ': ' + e); }
});
$('profileDeleteBtn').addEventListener('click', async ()=>{
  const name = $('profileList').value;
  if(!name){ alert(t('profilePickFirst')); return; }
  if(!confirm(t('confirmDelete').replace('{name}', name))) return;
  try{
    await fetch('/api/profiles/'+encodeURIComponent(name), {method:'DELETE'});
    if($('profileName').value.trim() === name) $('profileName').value = '';
    await refreshProfiles();
  }catch{}
});

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
  $('readyHint').textContent = t('readyHint')
    .replace('{n}', o.devices.length)
    .replace('{b}', o.backends.find(b=>b.default)?.id||'torch-eager');
}
function fillSelect(id, items){
  const sel=$(id); sel.innerHTML='';
  items.forEach(it=>{
    const opt=document.createElement('option');
    opt.value=it.value||it.id; opt.textContent=it.label||it.id;
    opt.dataset.labelEn=it.label||it.id;
    if(it.default) opt.selected=true;
    if(it.vae) opt.dataset.vae=it.vae;
    sel.appendChild(opt);
  });
  applySelectOptionLang(id);
}
$('model').addEventListener('change', e=>{
  const opt=e.target.selectedOptions[0];
  if(opt && opt.dataset.vae){ $('vae').value=opt.dataset.vae; }
});

function setStatus(state, text){
  $('statusDot').className='dot '+(state||'');
  $('statusText').textContent=text||t('idle');
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
  setStatus('running',t('generating'));
  $('generateBtn').disabled=true;
  $('stopBtn').disabled=false;
  logLine(t('submitting'));
  try{
    localStorage.setItem('yue_last_state', JSON.stringify(collectState()));
    const r=await fetch('/api/generate',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
    const j=await r.json();
    if(!r.ok){ throw new Error(j.error||'Failed to start job'); }
    currentJobId=j.job_id;
    openStream(j.job_id);
  }catch(e){
    setStatus('err',t('error'));
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
    setStatus('err',t('streamDisconnected'));
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
    if(d.prefix==='Completed') setStatus('done',t('stageComplete'));
    else if(d.prefix==='Failed'||d.prefix==='Cancelled') setStatus('err',d.prefix);
    else setStatus('running',d.stage);
  } else if(d.type==='stage'){
    logLine(`>> ${d.name}: ${d.status}`, 'stage-line running');
    setStatus('running',d.name);
  } else if(d.type==='complete'){
    $('progressFill').style.width='100%';
    setStatus('done',t('done'));
    logLine(t('completeLog')+d.audio,'stage-line done');
    showResult(d);
    $('generateBtn').disabled=false;
    $('stopBtn').disabled=true;
  } else if(d.type==='error'){
    setStatus('err',t('failed'));
    $('errBox').classList.add('show');
    $('errBox').textContent=d.message+'\n\n'+(d.traceback||'');
    $('generateBtn').disabled=false;
    $('stopBtn').disabled=true;
  } else if(d.type==='cancelled'){
    setStatus('err',t('cancelled'));
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
    $('truncatedWarn').textContent=t('truncatedWarn')
      .replace('{abc}',d.truncated.abc).replace('{sem}',d.truncated.semantic);
  }
  $('downloadBtn').onclick=()=>{ window.location=`/api/jobs/${currentJobId}/download`; };
  $('openFolderBtn').onclick=async()=>{
    const r=await fetch(`/api/jobs/${currentJobId}/open-folder`);
    if(!r.ok){ const j=await r.json(); alert(t('alertFailed')+(j.error||r.status)); }
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
    $('toggleEditBtn').textContent = t('btnUseBuilder');
  } else {
    $('styleEdit').style.display = 'none';
    $('stylePreview').style.opacity = '';
    $('toggleEditBtn').textContent = t('btnEditManually');
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
  setStatus('err',t('stopping'));
  try{
    const r=await fetch(`/api/jobs/${currentJobId}/stop`,{method:'POST'});
    const j=await r.json();
    if(j.forced_kill){
      logLine(t('threadKilled'),'stage-line err');
    }
    if(j.still_alive){
      logLine(t('threadAlive'),'stage-line err');
    }
  }catch(e){
    logLine(t('stopFailed')+e,'stage-line err');
  }
  if(eventSource){ eventSource.close(); eventSource=null; }
  setStatus('err',t('stopped'));
  $('generateBtn').disabled=false;
});

// ---- Song templates: complete sample songs (one click fills style + lyrics) ----
const SONG_TEMPLATES = {
  pop: {
    style:{lang:'English',gender:'female',genre:'pop',tempo:100,
      instruments:['acoustic piano','rounded bass','light drums'],
      mood:['warm','lyrical memorable melody']},
    lyrics:`[Verse]
City lights are blinking slow tonight,
Every window tells a story in the glow.

[Chorus]
We are dancing on the edge of the night,
Holding on to every spark of light,
Turn the music up and let it go,
This is all we need, this is all we know.

[Verse]
Morning came and painted gold the street,
Every shadow moved along beneath our feet.

[Chorus]
We are dancing on the edge of the night,
Holding on to every spark of light,
Turn the music up and let it go,
This is all we need, this is all we know.

[Outro]
One more chorus in the morning glow.`
  },
  rock: {
    style:{lang:'English',gender:'male',genre:'rock',tempo:140,
      instruments:['distorted electric guitar','driving drums'],
      mood:['energetic','anthemic','raw']},
    lyrics:`[Verse]
They said sit down, keep your voice low,
But the fire in my chest has nowhere left to go.

[Chorus]
We are the thunder, we are the flame,
We will not beg and we will not tame,
Break the silence, break the chain,
Roar like a storm through the rain.

[Verse]
Every wall they built is coming down,
Every scar I own is my crown.

[Chorus]
We are the thunder, we are the flame,
We will not beg and we will not tame,
Break the silence, break the chain,
Roar like a storm through the rain.

[Bridge]
Softer now, one vow, one name...

[Chorus]
We are the thunder, we are the flame,
Break the silence, break the chain!`
  },
  hiphop: {
    style:{lang:'English',gender:'male',genre:'hip-hop',tempo:90,
      instruments:['808 bass','trap hi-hats','acoustic piano'],
      mood:['aggressive','street']},
    lyrics:`[Verse]
Woke up with a vision, pockets full of static,
Every closed door taught me to be automatic,
Counted every no like a stack of paper,
Turned the whole grind into my own savior.

[Chorus]
From the bottom to the light, watch me rise,
Concrete flowers grow where the dark never dies.

[Verse]
Second round, same fight, colder shoulders,
Made a ladder out of every boulder.

[Chorus]
From the bottom to the light, watch me rise,
Concrete flowers grow where the dark never dies.`
  },
  edm: {
    style:{lang:'English',gender:'female',genre:'electronic',tempo:128,
      instruments:['synth pads','deep sub-bass','arpeggios'],
      mood:['ethereal','dreamy','atmospheric']},
    lyrics:`[Verse]
Hands up, the bass is a heartbeat,
Colors are moving beneath our feet.

[Chorus]
Run into the light, tonight we come alive,
Louder than the dark, together we will drive,
Hold the moment, hold it high,
We come alive, alive tonight.

[Verse]
Signals are spinning, the sky's a screen,
Everything glitters in ultraviolet green.

[Chorus]
Run into the light, tonight we come alive,
Louder than the dark, together we will drive,
Hold the moment, hold it high,
We come alive, alive tonight.`
  },
  ballad: {
    style:{lang:'English',gender:'female',genre:'pop',tempo:72,
      instruments:['acoustic piano','string section'],
      mood:['intimate','heartfelt','unhurried phrasing']},
    lyrics:`[Verse]
Your coffee cup is still beside the door,
The way you left it, like the night before.

[Chorus]
And I miss you in the quiet,
In the ordinary hours,
In the kitchen light at midnight,
In the smell of dying flowers,
I miss you in the quiet.

[Verse]
Your sweater still is hanging by the stair,
I cannot teach my hands to leave it there.

[Chorus]
And I miss you in the quiet,
In the ordinary hours,
In the kitchen light at midnight,
In the smell of dying flowers,
I miss you in the quiet.

[Bridge]
Maybe love is just the shape of what remains.

[Chorus]
And I miss you in the quiet,
In the ordinary hours.`
  },
  mandarin: {
    style:{lang:'Mandarin',gender:'female',genre:'pop',tempo:88,
      instruments:['acoustic piano','rounded bass','light drums'],
      mood:['warm','lyrical memorable melody']},
    lyrics:`[Verse]
霓虹灯下影子拉得很长，
旧街的风还带着一点凉。

[Chorus]
把这首歌轻轻地唱，
唱给远方想念的人听，
无论世界变得多匆忙，
心里的光从不曾熄灭。

[Verse]
时间带走了青春的模样，
带不走心里那道温柔的光。

[Chorus]
把这首歌轻轻地唱，
唱给远方想念的人听，
无论世界变得多匆忙，
心里的光从不曾熄灭。

[Outro]
晚安，好梦，明天再见。`
  },
  jazz: {
    style:{lang:'English',gender:'female',genre:'jazz',tempo:95,
      instruments:['acoustic piano','tenor saxophone','upright bass','brushed drums'],
      mood:['intimate','spacious modern harmony','natural phrasing']},
    lyrics:`[Verse]
Midnight pours like honey through the bar,
The saxophone knows exactly who we are.

[Chorus]
Slow down, let the record spin,
One more dance before the morning comes in,
The city can wait, the world can spin,
Slow down, and let the night begin.

[Verse]
Your laughter trades its secrets with the smoke,
Two strangers in a song that hasn't spoke.

[Chorus]
Slow down, let the record spin,
One more dance before the morning comes in.`
  },
  cinematic: {
    style:{lang:'English',gender:'choir',genre:'orchestral',tempo:70,
      instruments:['string section','french horns','timpani','choir'],
      mood:['epic','cinematic']},
    lyrics:`[Verse]
Beyond the mountains, beyond the sea,
A thousand voices call to me.

[Chorus]
Rise, oh rise, the dawn is near,
Carry the fire through the fear,
Hand in hand we chart the way,
Night is old but breaks to day.

[Verse]
Every banner torn, every promise kept,
The road remembers every step.

[Chorus]
Rise, oh rise, the dawn is near,
Carry the fire through the fear,
Hand in hand we chart the way,
Night is old but breaks to day.`
  },
  duet: {
    style:{lang:'English',gender:'duet',genre:'pop',tempo:96,
      instruments:['steel guitar','rounded bass','light drums'],
      mood:['warm','heartfelt']},
    lyrics:`[Verse]
You called me when the sky was gray,
I heard your voice and found my way.

[Chorus]
Two hearts on one road, side by side,
Through every storm and every tide,
You take my hand, I take the fall,
Side by side through it all.

[Verse]
We built a lamp from every fight,
A little warmth against the night.

[Chorus]
Two hearts on one road, side by side,
Through every storm and every tide,
You take my hand, I take the fall,
Side by side through it all.`
  },
};
$('songTemplate').addEventListener('change', e=>{
  const tpl = SONG_TEMPLATES[e.target.value];
  if(!tpl){ return; }
  const s = tpl.style;
  $('selLang').value = s.lang;
  $('selGender').value = s.gender;
  $('selGenre').value = s.genre;
  $('selTempo').value = String(s.tempo);
  document.querySelectorAll('.chip.active').forEach(c=>c.classList.remove('active'));
  (s.instruments||[]).forEach(ins=>{
    const c = document.querySelector(`#instrumentChips .chip[data-ins="${ins}"]`);
    if(c) c.classList.add('active');
  });
  (s.mood||[]).forEach(m=>{
    const c = document.querySelector(`#moodChips .chip[data-mood="${m}"]`);
    if(c) c.classList.add('active');
  });
  if(manualEdit) $('toggleEditBtn').click(); // back to builder so chips regenerate
  buildStyle();
  $('genrePreset').value = '';
  $('lyrics').value = tpl.lyrics;
  localStorage.setItem('yue_last_state', JSON.stringify(collectState()));
});

// ---- GPU monitor (live) + VRAM killer ----
function fmtGB(mib){ return (mib/1024).toFixed(1); }
function gpuColor(frac){ return frac<0.6 ? '#22c55e' : (frac<0.85 ? '#eab308' : '#ef4444'); }
async function refreshGpu(){
  if(document.hidden) return;
  let g;
  try{ g = await (await fetch('/api/gpu')).json(); }catch(e){ return; }
  if(!g.available){ $('gpuName').textContent = g.reason || t('gpuUnavailable'); return; }
  $('gpuName').textContent = g.name;
  const frac = g.mem_total>0 ? g.mem_used/g.mem_total : 0;
  $('gpuBarFill').style.width = (frac*100).toFixed(1)+'%';
  $('gpuBarFill').style.background = gpuColor(frac);
  $('gpuMemText').textContent = fmtGB(g.mem_used)+' / '+fmtGB(g.mem_total)+' GiB '+t('gpuUsed')+' ('+(frac*100).toFixed(0)+'%)';
  $('gpuUtilText').textContent = (g.util!=null) ? t('gpuUtil')+' '+g.util+'%' : '';
  const el = $('gpuProcList');
  if(!g.processes || !g.processes.length){ el.textContent = t('gpuNoProcs'); }
  else{
    el.innerHTML = g.processes.map(p =>
      '<div>'+(p.killable&&!p.self?'⚡ ':'· ')+'PID '+p.pid+' · '+p.name+
      (p.self?' <b>('+t('gpuSelf')+')</b>':'')+' — '+
      (p.mem!=null ? fmtGB(p.mem)+' GiB' : 'n/a')+'</div>'
    ).join('');
  }
}
$('gpuKillBtn').addEventListener('click', async ()=>{
  if(!confirm(t('gpuKillConfirm'))) return;
  const btn = $('gpuKillBtn'); btn.disabled = true;
  try{
      const r = await (await fetch('/api/gpu/kill',{method:'POST'})).json();
      if(r.error){ alert(r.error); }
      else{
        const k=(r.killed||[]).length, f=(r.failed||[]).length, c=(r.cancelled_jobs||0);
        let msg = t('gpuKilled')+' '+k+(f?(' · '+t('gpuFailed')+' '+f):'');
        if(c>0){ msg += '\n'+t('gpuCancelled')+' '+c; }
        if(r.freed_cache_mib!=null && r.freed_cache_mib>0){ msg += '\n'+t('gpuFreed')+' '+r.freed_cache_mib+' MiB'; }
        if(k===0 && !(r.freed_cache_mib>0)){ msg += '\n'+t('gpuNothingToKill'); }
        alert(msg);
      }
    }catch(e){ alert('Kill request failed: '+e); }
  btn.disabled = false;
  refreshGpu(); setTimeout(refreshGpu, 2000);
});
$('gpuNukeBtn').addEventListener('click', async ()=>{
  if(!confirm(t('gpuNukeConfirm'))) return;
  const btn = $('gpuNukeBtn'); btn.disabled = true;
  try{ await fetch('/api/gpu/kill?nuke=1',{method:'POST'}); }catch(e){}
  alert(t('gpuNukeMsg'));
  btn.disabled = false;
});
setInterval(refreshGpu, 2500);
refreshGpu();

// ---- Init: restore UI language + last state, then load options/profiles ----
(function initLang(){
  const saved = localStorage.getItem('yue_ui_lang');
  const lang = saved || ((navigator.language||'en').toLowerCase().startsWith('zh') ? 'zh' : 'en');
  $('uiLang').value = I18N[lang] ? lang : 'en';
  applyLang($('uiLang').value);
})();
loadOptions().then(()=>{
  const savedState = localStorage.getItem('yue_last_state');
  if(savedState){ try{ applyState(JSON.parse(savedState)); }catch{} }
  refreshProfiles();
});
</script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
def index():
    # no-cache: UI updates reach LAN devices without a hard refresh
    return HTMLResponse(HTML, headers={"Cache-Control": "no-cache"})


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
