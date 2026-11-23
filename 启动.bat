@echo off
set PYTHONUTF8=1
cd /d "%~dp0"
set "PY=python"
where python >nul 2>nul || (
  echo Python not found in PATH. Run the installer BAT first.
  pause
  exit /b 1
)
echo Starting Video Dubbing Tool ...
"%PY%" app.py
if errorlevel 1 (
  echo.
  echo Startup failed. Please run the installer BAT in this folder first.
  pause
)
