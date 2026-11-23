@echo off
set PYTHONUTF8=1
cd /d "%~dp0"
set "PY=D:\Python3.10\python.exe"
if not exist "%PY%" set "PY=python"
echo Starting Video Dubbing Tool ...
"%PY%" app.py
if errorlevel 1 (
  echo.
  echo Startup failed. Please run the installer BAT in D:\DubTool first.
  pause
)
