@echo off
REM One-click setup for Windows: double-click this file, or run it from a terminal.
REM Creates .venv, installs PyTorch with CUDA (NVIDIA GPU) and the analyzer's requirements.
cd /d "%~dp0"

set PY=
where py >nul 2>nul && py -3.12 --version >nul 2>nul && set "PY=py -3.12"
if not defined PY where py >nul 2>nul && set "PY=py -3"
if not defined PY where python >nul 2>nul && set "PY=python"
if not defined PY (
  echo [X] Python not found. Install Python 3.12 from https://www.python.org/downloads/
  echo     and tick "Add python.exe to PATH" in the installer, then run this again.
  goto :fail
)
echo Using: %PY%
%PY% --version

if not exist .venv\Scripts\python.exe (
  echo Creating virtual environment .venv ...
  %PY% -m venv .venv || goto :fail
)
set "VPY=.venv\Scripts\python.exe"

"%VPY%" -m pip install --upgrade pip || goto :fail

echo.
echo Installing PyTorch with CUDA (about 3 GB, takes a while) ...
"%VPY%" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128 || goto :torchfail

echo.
echo Installing analyzer requirements ...
"%VPY%" -m pip install -r requirements.txt || goto :fail

echo.
"%VPY%" -c "import torch; ok=torch.cuda.is_available(); print('GPU OK:', torch.cuda.get_device_name(0)) if ok else print('[!] CUDA not available - update the NVIDIA driver (GeForce Experience / nvidia.com) and run this again')"
"%VPY%" -m pytest -q tests

echo.
echo Done. To use it, open a terminal in this folder and run:
echo     .venv\Scripts\activate.bat
echo     python scripts\analyze.py data\videos\match.mp4 --max-seconds 60 --out runs\test
pause
exit /b 0

:torchfail
echo.
echo [X] PyTorch install failed. Most common cause: Python version too new for PyTorch.
echo     Install Python 3.12 from python.org, delete the .venv folder, and run this again.
:fail
echo.
echo [X] Setup failed - copy the red/error lines above and send them to Claude.
pause
exit /b 1
