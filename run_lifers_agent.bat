@echo off
setlocal

set "ROOT=%~dp0"
set "APP=%ROOT%lifers"
if not exist "%APP%\scripts\run_agent.py" set "APP=%ROOT%lifers_brain"

if not exist "%APP%\scripts\run_agent.py" (
  echo Missing Lifers app folder lifers/lifers_brain: "%APP%"
  exit /b 1
)

cd /d "%APP%"
set PYTHONUTF8=1
set SANDBOX=1
set MODEL=transformer

set "PYEXE=python"
if exist "%APP%\.venv\Scripts\python.exe" set "PYEXE=%APP%\.venv\Scripts\python.exe"
"%PYEXE%" .\scripts\run_agent.py

endlocal

