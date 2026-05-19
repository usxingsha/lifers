@echo off
REM Lifers — 本地 AI 终端助手
REM 优先本地 CLI，本地无权重时回退到 Kali 远程

setlocal

REM Try local Python lifers first
set "LOCAL_PYTHON="
if exist "%~dp0lifers\.venv\Scripts\python.exe" (
    set "LOCAL_PYTHON=%~dp0lifers\.venv\Scripts\python.exe"
) else (
    for /f "tokens=*" %%p in ('where python 2^>nul') do (
        set "LOCAL_PYTHON=%%p"
        goto :found_python
    )
)
:found_python

if defined LOCAL_PYTHON (
    "%LOCAL_PYTHON%" -m lifers.scripts.cli %*
    if %errorlevel% equ 0 goto :done
)

REM Fallback to remote Kali
echo [Lifers] 本地未就绪，通过 SSH 连接到 Kali...
ssh -o ControlMaster=no -o StrictHostKeyChecking=no -t kali@192.168.234.152 "lifers %*"

:done
endlocal
