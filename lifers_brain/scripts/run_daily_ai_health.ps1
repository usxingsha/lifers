#Requires -Version 5.1
<#
.SYNOPSIS
  运行每日 AI 健康评测（逻辑自洽/全链检查/桥接延迟/上下文吞吐），写入 lifers_brain/state/daily_health/。

.EXAMPLE
  cd lifers_brain\scripts
  .\run_daily_ai_health.ps1
  $env:LIFERS_DAILY_HEALTH_AUTO_REMEDIATE='1'; .\run_daily_ai_health.ps1

.NOTES
  计划任务（每天一次，例如 04:30）：
  schtasks /Create /TN "LifersDailyAIHealth" /TR "powershell -NoProfile -ExecutionPolicy Bypass -File C:\path\to\lifers\lifers_brain\scripts\run_daily_ai_health.ps1" /SC DAILY /ST 04:30 /RL LIMITED
#>
param(
    [switch] $AutoRemediate
)

$ErrorActionPreference = "Stop"
$BrainRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Py = Join-Path $BrainRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $Py)) {
    $Py = "python"
}

$env:LIFERS_ROOT = $BrainRoot
if ($AutoRemediate) {
    $env:LIFERS_DAILY_HEALTH_AUTO_REMEDIATE = "1"
}

& $Py (Join-Path $BrainRoot "eval\daily_ai_health.py")
exit $LASTEXITCODE
