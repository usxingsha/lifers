@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

:: ============================================================
:: Lifers 抗断电/崩溃 自动续接训练脚本
:: 挂后台运行，崩溃/断网/断电后自动从最新权重续接
:: ============================================================

set "ROOT=%~dp0.."
cd /d "%ROOT%"

set "CTL=weights\.train_control"
set "LOG=weights\_resilient_loop.log"
set "CRASH_LOG=weights\_crash_history.log"
set "PYTHONPATH=%ROOT%"

:: 确保 python 可用
where python >nul 2>&1
if %errorlevel% neq 0 (
    echo [%date% %time%] 错误: 找不到 python >> "%LOG%"
    exit /b 1
)

:: 默认无限训练
if "%LIFERS_ESCALATE_UNLIMITED%"=="" set "LIFERS_ESCALATE_UNLIMITED=1"
if "%LIFERS_RAMP_MAX_ITERS%"=="" set "LIFERS_RAMP_MAX_ITERS=999999"
if "%LIFERS_ESCALATE_RESUME%"=="" set "LIFERS_ESCALATE_RESUME=1"
if "%PYTHONUNBUFFERED%"=="" set "PYTHONUNBUFFERED=1"

echo [%date% %time%] ===== Lifers 弹性训练启动 ===== >> "%LOG%"
echo   ROOT=%ROOT% >> "%LOG%"
echo   LIFERS_ESCALATE_UNLIMITED=%LIFERS_ESCALATE_UNLIMITED% >> "%LOG%"
echo   LIFERS_RAMP_MAX_ITERS=%LIFERS_RAMP_MAX_ITERS% >> "%LOG%"

:: 确保 control 文件存在
if not exist "%CTL%" echo run > "%CTL%"

set "CRASH_COUNT=0"
set "BACKOFF=2"

:loop
    :: 检查 control 是否为 stop
    if exist "%CTL%" (
        set /p MODE=<"%CTL%"
        if /i "!MODE!"=="stop" (
            echo [%date% %time%] control=stop — 退出循环 >> "%LOG%"
            goto :done
        )
        if /i "!MODE!"=="pause" (
            echo [%date% %time%] control=pause — 等待中... >> "%LOG%"
            timeout /t 10 /nobreak >nul
            goto :loop
        )
    )

    echo [%date% %time%] 启动训练 (第 !CRASH_COUNT! 次连续崩溃) >> "%LOG%"

    python scripts/train_lifers_escalate.py >> "%LOG%" 2>&1
    set "EXIT_CODE=!errorlevel!"

    if !EXIT_CODE! equ 0 (
        echo [%date% %time%] 训练正常结束 (exit 0) >> "%LOG%"
        set "CRASH_COUNT=0"
        set "BACKOFF=2"
        :: 检查是否应该继续（control 是否为 stop）
        if exist "%CTL%" (
            set /p MODE=<"%CTL%"
            if /i "!MODE!"=="stop" goto :done
        )
        timeout /t 3 /nobreak >nul
        goto :loop
    )

    :: 崩溃 / 异常退出
    set /a CRASH_COUNT+=1
    echo [%date% %time%] 崩溃! exit=!EXIT_CODE! 连续崩溃: !CRASH_COUNT! >> "%CRASH_LOG%"

    :: 渐进退避: 2s, 4s, 8s, 16s, 30s (最大)
    if !CRASH_COUNT! gtr 5 set "BACKOFF=30"
    if !CRASH_COUNT! equ 5 set "BACKOFF=30"
    if !CRASH_COUNT! equ 4 set "BACKOFF=16"
    if !CRASH_COUNT! equ 3 set "BACKOFF=8"
    if !CRASH_COUNT! equ 2 set "BACKOFF=4"

    echo [%date% %time%] !BACKOFF! 秒后重试... >> "%LOG%"
    timeout /t !BACKOFF! /nobreak >nul
    goto :loop

:done
    echo [%date% %time%] 训练循环退出 >> "%LOG%"
    exit /b 0
