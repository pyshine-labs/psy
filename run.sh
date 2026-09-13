#!/usr/bin/env bash
# PSY Music Studio launcher (Linux/macOS)
# Automatically finds the venv python and starts the WebUI.

set -e
cd "$(dirname "$0")"

# Pick the first available python (local venv > known venv > system)
if [ -x ".venv/bin/python" ]; then
    PY=".venv/bin/python"
elif [ -x "/e/voice-assitant/.venv/Scripts/python.exe" ]; then
    PY="/e/voice-assitant/.venv/Scripts/python.exe"   # Git-Bash on Windows
elif [ -x "/mnt/e/voice-assitant/.venv/bin/python" ]; then
    PY="/mnt/e/voice-assitant/.venv/bin/python"
elif command -v python >/dev/null 2>&1; then
    PY="python"
else
    echo "[error] No python found. Create a venv first:  python -m venv .venv"
    exit 1
fi

echo
echo "  PSY Music Studio"
echo "  python: $PY"
echo "  URL:     http://127.0.0.1:7860"
echo

# Open browser after short delay (best-effort, non-fatal)
(
    sleep 3
    if command -v xdg-open >/dev/null 2>&1; then xdg-open http://127.0.0.1:7860
    elif command -v open >/dev/null 2>&1; then open http://127.0.0.1:7860
    fi
) >/dev/null 2>&1 &

# Forward any extra args, e.g.  ./run.sh --port 8000
exec "$PY" webui.py "$@"
