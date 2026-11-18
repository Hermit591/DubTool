@echo off
chcp 65001 >nul
set PYTHONUTF8=1
cd /d "%~dp0"
set "PY=python"
where python >nul 2>&1 || (
  echo [ERROR] Python not found. Install Python 3.10+ and check "Add to PATH", then rerun.
  pause
  exit /b 1
)
echo [1/2] Installing Python packages ...
"%PY%" -m pip install -q numpy scipy soundfile librosa edge-tts "setuptools<81"
if errorlevel 1 (
  echo pip install failed. Check your network / pip source.
  pause
  exit /b 1
)
echo [2/2] Checking ffmpeg ...
where ffmpeg >nul 2>&1 && (
  echo ffmpeg: OK ^(found in PATH^)
) || (
  echo ffmpeg: NOT FOUND in PATH.
  echo   Install ffmpeg and add to PATH, OR put ffmpeg.exe next to this bat,
  echo   OR set the "ffmpeg" field in config.json to its absolute path.
)
echo.
echo Done. Now run "启动.bat".
pause
