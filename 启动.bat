@echo off
chcp 65001 >nul
set PYTHONUTF8=1
cd /d "%~dp0"
set "PY=python"
where python >nul 2>&1 || (
  echo [ERROR] Python not found. Install Python 3.10+ and check "Add to PATH".
  pause
  exit /b 1
)
echo Starting DubTool ...
"%PY%" app.py
if errorlevel 1 (
  echo.
  echo Startup failed. Run "安装依赖.bat" first.
  pause
)
