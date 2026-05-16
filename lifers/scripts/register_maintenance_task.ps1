#Requires -Version 5.1
<#
.SYNOPSIS
  注册 Windows 计划任务：每日运行 run_maintenance_all.ps1（git pull + Kali 同步 + 健康检查）。

.EXAMPLE
  # 以管理员 PowerShell 运行（创建计划任务需要权限）：
  cd lifers\scripts
  .\register_maintenance_task.ps1
  .\register_maintenance_task.ps1 -DailyAt "05:30"
#>
param(
    [string] $DailyAt = "05:00",
    [switch] $SkipGit,
    [switch] $SkipSync
)

$ErrorActionPreference = "Stop"
$ScriptPath = Join-Path $PSScriptRoot "run_maintenance_all.ps1"
if (-not (Test-Path $ScriptPath)) { throw "missing $ScriptPath" }

$TaskName = "LifersMaintenanceAll"
$argTail = ""
if ($SkipGit) { $argTail += " -SkipGit" }
if ($SkipSync) { $argTail += " -SkipSync" }
$args = "-NoProfile -ExecutionPolicy Bypass -File `"$ScriptPath`"$argTail"

$action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument $args
$trigger = New-ScheduledTaskTrigger -Daily -At $DailyAt
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable
Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger -Settings $settings -Force | Out-Null
Write-Host "Registered scheduled task '$TaskName' daily at $DailyAt"
Write-Host "  -> $args"
