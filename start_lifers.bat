@echo off
REM Lifers 一键启动 — 自动检测环境，后台静默运行所有服务
setlocal enabledelayedexpansion

REM 自动检测 LIFERS 根目录
set LIFERS_HOME=%~dp0
set LIFERS_ROOT=%LIFERS_HOME%lifers

REM 自动检测 Python
set PYTHON=
if exist "%LIFERS_HOME%.venv\Scripts\python.exe" (
    set PYTHON=%LIFERS_HOME%.venv\Scripts\python.exe
    echo [Lifers] 使用虚拟环境: .venv
) else (
    for /f "tokens=*" %%p in ('where python 2^>nul') do set PYTHON=%%p
    if "!PYTHON!"=="" (
        for /f "tokens=*" %%p in ('where py 2^>nul') do set PYTHON=%%p
    )
)
if "%PYTHON%"=="" (
    echo [错误] 未找到 Python。请先运行 install.bat
    pause
    exit /b 1
)
echo [Lifers] Python: %PYTHON%

set LIFERS_SILENT=1
set PYTHONPATH=%LIFERS_ROOT%
set LOGS=%LIFERS_HOME%logs

if not exist "%LOGS%" mkdir "%LOGS%"

REM 检查 Gate 端口是否可用
netstat -ano 2>nul | findstr "127.0.0.1:55555" >nul
if %errorlevel% equ 0 (
    echo [Lifers] Gate 端口 55555 已被占用，尝试复用...
) else (
    echo [Lifers] 启动 Gate 服务...
    start "Lifers Gate" /MIN "%PYTHON%" -u -m lifers.scripts.lifers_gate --silent --port 55555 > "%LOGS%\gate_stdout.log" 2>&1
    timeout /t 2 >nul
)

REM 检查 GUI 端口是否可用
netstat -ano 2>nul | findstr "127.0.0.1:18765" >nul
if %errorlevel% equ 0 (
    echo [Lifers] GUI 端口 18765 已被占用，尝试复用...
) else (
    echo [Lifers] 启动 Web UI...
    start "Lifers GUI" /MIN "%PYTHON%" -u "%LIFERS_ROOT%\scripts\start_gui_host.py" --brain-root "%LIFERS_ROOT%" --port 18765 > "%LOGS%\gui_stdout.log" 2>&1
    timeout /t 3 >nul
)

REM 监控守护
echo [Lifers] 启动监控守护...
start "Lifers Monitor" /MIN "%PYTHON%" -u -m lifers.scripts.monitor_all_pillars --interval 120 > "%LOGS%\monitor_stdout.log" 2>&1

timeout /t 2 >nul

echo.
echo [Lifers] 服务已启动:
echo   Gate:   http://127.0.0.1:55555
echo   Web UI: http://127.0.0.1:18765
echo   日志:   %LOGS%
echo.

REM 打开浏览器
start http://127.0.0.1:18765

endlocal
exit /b 0
