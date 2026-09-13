@echo off
REM PSY Music Studio launcher (Windows)
REM Automatically finds the venv python and starts the WebUI.

setlocal enabledelayedexpansion
cd /d "%~dp0"

REM Pick the first available python (local venv > known venv > system)
set "PY="
if exist ".venv\Scripts\python.exe" set "PY=.venv\Scripts\python.exe"
if not defined PY if exist "E:\voice-assitant\.venv\Scripts\python.exe" set "PY=E:\voice-assitant\.venv\Scripts\python.exe"
if not defined PY (
  where python >nul 2>nul
  if not errorlevel 1 set "PY=python"
)
if not defined PY (
  echo [error] No python found. Create a venv first:  python -m venv .venv
  pause
  exit /b 1
)

echo.
echo   PSY Music Studio
echo   python: %PY%
echo   URL:     http://127.0.0.1:7860
echo.

REM Open browser after short delay (in background)
start "" cmd /c "timeout /t 3 >nul && start http://127.0.0.1:7860"

REM Forward any extra args, e.g.  run.bat --port 8000
"%PY%" webui.py %*
endlocal
