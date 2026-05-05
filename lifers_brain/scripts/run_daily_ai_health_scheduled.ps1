#Requires -Version 5.1
<#
.SYNOPSIS
  供计划任务调用：跑每日 AI 健康并把输出追加到 state/daily_health/scheduled.log
#>
$ErrorActionPreference = "Stop"
$BrainRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$LogDir = Join-Path $BrainRoot "state\daily_health"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$Log = Join-Path $LogDir "scheduled.log"

Add-Content -Path $Log -Value "===== $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') =====" -Encoding UTF8

$Py = Join-Path $BrainRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $Py)) {
    $Py = "python"
}
$env:LIFERS_ROOT = $BrainRoot

$output = & $Py (Join-Path $BrainRoot "eval\daily_ai_health.py") 2>&1
$code = $LASTEXITCODE
$output | ForEach-Object { Add-Content -Path $Log -Value $_ -Encoding UTF8 }
exit $code
