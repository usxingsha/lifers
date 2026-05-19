@echo off
:: ============================================================
:: 注册 Windows 计划任务 — 开机自动启动 Lifers 弹性训练
:: 需以管理员身份运行！
:: ============================================================
chcp 65001 >nul

set "ROOT=%~dp0.."
cd /d "%ROOT%"
set "ROOT=%CD%"

set "TASK_NAME=LifersTrainResilient"
set "PS1=%ROOT%\scripts\run_train_resilient.ps1"

echo 注册计划任务: %TASK_NAME%
echo 脚本路径: %PS1%

:: 删除旧任务（如果存在）
schtasks /Delete /TN "%TASK_NAME%" /F >nul 2>&1

:: 创建新任务：开机启动 + 每隔 10 分钟检查是否在运行（自动拉起）
schtasks /Create /TN "%TASK_NAME%" /SC ONSTART /RU "%USERNAME%" ^
  /TR "powershell.exe -ExecutionPolicy Bypass -WindowStyle Hidden -File \"%PS1%\"" ^
  /F /RL HIGHEST /DELAY 0001:00

if %errorlevel% equ 0 (
    echo [成功] 任务已注册。训练将在系统启动后自动运行。
    echo 管理: schtasks /Run /TN "%TASK_NAME%"  (手动启动)
    echo        schtasks /End /TN "%TASK_NAME%"  (手动停止)
    echo        schtasks /Delete /TN "%TASK_NAME%" (删除)
) else (
    echo [失败] 请以管理员身份运行此脚本！
)

pause
