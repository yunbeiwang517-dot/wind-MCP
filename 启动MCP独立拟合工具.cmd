@echo off
setlocal
cd /d "%~dp0"
if exist "__pycache__" rmdir /s /q "__pycache__"
if exist "mcp_algorithms\__pycache__" rmdir /s /q "mcp_algorithms\__pycache__"
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
%PY% -c "import PySide6,pandas,numpy,openpyxl" >nul 2>nul
if errorlevel 1 (
  echo Required Python packages are missing. Installing now...
  %PY% -m pip install -r "%~dp0requirements.txt"
  if errorlevel 1 (
    echo ERROR: Dependency installation failed.
    pause
    exit /b 1
  )
)
%PY% "%~dp0mcp_app.py"
set "RC=%errorlevel%"
if not "%RC%"=="0" (
  echo.
  echo ERROR: MCP application exited with code %RC%.
  pause
)
endlocal
