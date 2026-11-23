@echo off
set PYTHONUTF8=1
cd /d "%~dp0"
set "PY=python"
where python >nul 2>nul || (
  echo Python not found in PATH. Install Python 3.10+ and add it to PATH first.
  pause
  exit /b 1
)

echo [1/3] Installing python packages ...
"%PY%" -m pip install -q numpy scipy soundfile librosa sherpa-onnx edge-tts "setuptools<81"
if errorlevel 1 (
  echo pip install failed. Check your network.
  pause
  exit /b 1
)

echo [2/3] Checking ffmpeg and local models ...
"%PY%" -c "import os,sys;sys.path.insert(0,os.getcwd());from core import config;c=config.load();ff=config.resolve_ffmpeg(c);print('ffmpeg:',ff,'OK' if os.path.exists(ff) else 'MISSING');print('ASR model:',os.path.exists(os.path.join(c['sensevoice_dir'],'model.int8.onnx')));print('TTS model:',os.path.exists(os.path.join(c['aishell3_dir'],'vits-aishell3.onnx')))"

echo [3/3] Checking voice library ...
"%PY%" -c "import os,sys;sys.path.insert(0,os.getcwd());from core import voices;d=voices.library();print('voices:',len(d),'male:',sum(1 for x in d if x['gender']=='male'),'female:',sum(1 for x in d if x['gender']=='female'))"

echo.
echo Done. Now run start.bat (or the desktop launcher).
echo Online voices need internet; local voices + local ASR work fully offline.
pause
