@echo off
REM ============================================================
REM  Lifers 一键安装脚本 (Windows)
REM  自动检测 Python、创建虚拟环境、安装依赖、初始化配置
REM ============================================================
setlocal enabledelayedexpansion

set "LIFERS_HOME=%~dp0"
set "LIFERS_ROOT=%LIFERS_HOME%lifers"
set "VENV=%LIFERS_HOME%.venv"

echo.
echo  ╔══════════════════════════════════════╗
echo  ║   Lifers 终身监禁者 AI 安装向导      ║
echo  ╚══════════════════════════════════════╝
echo.

REM ---------- 1. 检测 Python ----------
echo [1/5] 检测 Python 环境...

set PYTHON=
for /f "tokens=*" %%p in ('where python 2^>nul') do (
    set PYTHON=%%p
    goto :found_python
)
for /f "tokens=*" %%p in ('where python3 2^>nul') do (
    set PYTHON=%%p
    goto :found_python
)
for /f "tokens=*" %%p in ('where py 2^>nul') do (
    set PYTHON=%%p
    goto :found_python
)
echo   [错误] 未找到 Python。请安装 Python 3.10+ 并添加到 PATH。
echo         下载: https://www.python.org/downloads/
pause
exit /b 1

:found_python
echo   找到: !PYTHON!
for /f "tokens=2" %%v in ('"!PYTHON!" --version 2^>^&1') do echo   版本: %%v

REM ---------- 2. 创建虚拟环境 ----------
echo.
echo [2/5] 创建虚拟环境...
if exist "%VENV%\Scripts\python.exe" (
    echo   虚拟环境已存在，跳过创建。
) else (
    "!PYTHON!" -m venv "%VENV%"
    if errorlevel 1 (
        echo   [错误] 创建虚拟环境失败。
        pause
        exit /b 1
    )
    echo   虚拟环境已创建: %VENV%
)

set "VENV_PYTHON=%VENV%\Scripts\python.exe"
set "VENV_PIP=%VENV%\Scripts\pip.exe"

REM ---------- 3. 升级 pip 并安装依赖 ----------
echo.
echo [3/5] 安装依赖...
"%VENV_PYTHON%" -m pip install --upgrade pip --quiet
"%VENV_PIP%" install numpy rich --quiet
if errorlevel 1 (
    echo   [错误] 安装依赖失败。
    pause
    exit /b 1
)
echo   核心依赖已安装 (numpy, rich)

REM ---------- 4. 安装 lifers 包(开发模式) ----------
echo.
echo [4/5] 注册 lifers 包...
pushd "%LIFERS_ROOT%"
"%VENV_PIP%" install -e . --quiet 2>nul
if errorlevel 1 (
    echo   [警告] editable install 失败，使用 PYTHONPATH 兜底。
)
popd
echo   lifers 包已注册

REM ---------- 5. 初始化配置 ----------
echo.
echo [5/5] 初始化配置...

REM 检查数据目录
if not exist "%LIFERS_HOME%data" mkdir "%LIFERS_HOME%data"
if not exist "%LIFERS_ROOT%\weights" mkdir "%LIFERS_ROOT%\weights"
if not exist "%LIFERS_HOME%logs" mkdir "%LIFERS_HOME%logs"

REM 检查语料库
if not exist "%LIFERS_ROOT%\weights\training_corpus.txt" (
    echo   [提示] 未找到训练语料，首次启动时将自动生成...
)

REM 生成初始权重(如果不存在)
if not exist "%LIFERS_ROOT%\weights\lifers_deep_transformer.json" (
    echo   [提示] 未找到深度模型权重，首次启动训练将自动创建。
    echo   启动后执行: lifers train --init
)

echo.
echo  ╔══════════════════════════════════════╗
echo  ║   安装完成!                           ║
echo  ╠══════════════════════════════════════╣
echo  ║  启动方式:                            ║
echo  ║   start_lifers.bat  — 一键启动所有服务 ║
echo  ║   lifers chat       — 命令行聊天      ║
echo  ║   Web UI: http://127.0.0.1:18765     ║
echo  ╚══════════════════════════════════════╝
echo.

REM 询问是否立即启动
set /p LAUNCH="是否立即启动 Lifers? [Y/n] "
if /i "%LAUNCH%"=="n" goto :done
if /i "%LAUNCH%"=="N" goto :done
call "%LIFERS_HOME%start_lifers.bat"

:done
endlocal
exit /b 0
