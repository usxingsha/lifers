#Requires -Version 5.1
<#
.SYNOPSIS
  注册当前用户的 Windows 计划任务：每天运行 Lifers 每日 AI 健康检查。

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File .\register_daily_ai_health_task.ps1
  powershell -ExecutionPolicy Bypass -File .\register_daily_ai_health_task.ps1 -DailyAt "05:15"

.NOTES
  无需管理员（当前用户任务）。卸载：Unregister-ScheduledTask -TaskName 'LifersDailyAIHealth' -Confirm:$false
#>
param(
    [string] $DailyAt = "04:30",
    [switch] $Unregister
)

$ErrorActionPreference = "Stop"
$ScriptPath = Join-Path $PSScriptRoot "run_daily_ai_health_scheduled.ps1"
if (-not (Test-Path $ScriptPath)) {
    throw "missing $ScriptPath"
}
$TaskName = "LifersDailyAIHealth"

if ($Unregister) {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue
    Write-Host "Removed task $TaskName"
    exit 0
}

# Parse HH:mm for Daily trigger
$parts = $DailyAt -split ":"
$h = [int]$parts[0]
$m = [int]$parts[1]
$today = Get-Date -Hour $h -Minute $m -Second 0

$Action = New-ScheduledTaskAction -Execute "powershell.exe" `
    -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$ScriptPath`""

$Trigger = New-ScheduledTaskTrigger -Daily -At $today

$Settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -ExecutionTimeLimit (New-TimeSpan -Hours 2)

Register-ScheduledTask -TaskName $TaskName -Action $Action -Trigger $Trigger -Settings $Settings `
    -Description "Lifers: daily_ai_health -> lifers_brain/state/daily_health/" | Out-Null

$BrainRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Write-Host "Registered task '$TaskName' daily at $DailyAt (current user)."
Write-Host "Log: $(Join-Path $BrainRoot 'state\daily_health\scheduled.log')"
