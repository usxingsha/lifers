@echo off
setlocal
REM lifers / 终身监禁者 — HTTP 网关默认 http://127.0.0.1:55555
set "ROOT=%~dp0.."
cd /d "%ROOT%"
set "PYTHONPATH=%ROOT%"
if not defined LIFERS_ROOT set "LIFERS_ROOT=%ROOT%"
python "%ROOT%\scripts\lifers_gate.py" --host 127.0.0.1 --port 55555 %*
endlocal
