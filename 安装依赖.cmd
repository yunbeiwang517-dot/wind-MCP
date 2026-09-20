@echo off
setlocal
cd /d "%~dp0"
set "PY=py -3"
%PY% -c "import sys" >nul 2>nul
if errorlevel 1 set "PY="
if not defined PY (
  set "PY=python"
  python -c "import sys" >nul 2>nul
  if errorlevel 1 set "PY="
)
if not defined PY (
  echo ERROR: Python 3 was not found.
  echo Please install Python 3.10 or later and enable Add Python to PATH.
  pause
  exit /b 1
)
echo Installing MCP dependencies...
%PY% -m pip install -r "%~dp0requirements.txt"
if errorlevel 1 (
  echo ERROR: Dependency installation failed.
  pause
  exit /b 1
)
echo Dependencies installed successfully.
pause
endlocal
