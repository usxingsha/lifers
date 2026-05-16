#Requires -Version 5.1
<#
.SYNOPSIS
  One-shot: optional git pull, Kali weight sync, daily AI health (optional remediate).

.PARAMETER SkipGit
  Skip git pull (use when working tree is dirty).

.PARAMETER SkipSync
  Skip scp sync from Kali.

.PARAMETER AutoRemediate
  Pass through to daily_ai_health workspace_custom merge.

.EXAMPLE
  cd lifers\scripts
  .\run_maintenance_all.ps1
  .\run_maintenance_all.ps1 -SkipGit
#>
param(
    [switch] $SkipGit,
    [switch] $SkipSync,
    [switch] $AutoRemediate
)

$ErrorActionPreference = "Stop"
$ScriptsDir = $PSScriptRoot
$BrainRoot = (Resolve-Path (Join-Path $ScriptsDir "..")).Path
$RepoRoot = (Resolve-Path (Join-Path $BrainRoot "..")).Path
$env:LIFERS_ROOT = $BrainRoot
$env:PYTHONUTF8 = "1"

function Write-Step([string] $m) {
    Write-Host ""
    Write-Host "=== $m ===" -ForegroundColor Cyan
}

if (-not $SkipGit) {
    Write-Step "git pull ($RepoRoot)"
    Push-Location $RepoRoot
    try {
        $st = git status --porcelain 2>$null
        if ($st) {
            Write-Warning "Dirty working tree: skipped git pull. Commit/stash or use -SkipGit."
        } else {
            git pull --rebase origin main
            if ($LASTEXITCODE -ne 0) {
                throw "git pull failed"
            }
        }
    } finally {
        Pop-Location
    }
}

if (-not $SkipSync) {
    Write-Step "sync weights from Kali"
    & (Join-Path $ScriptsDir "sync_weights_from_kali.ps1")
    if ($LASTEXITCODE -ne 0) {
        throw "sync_weights_from_kali failed"
    }
}

Write-Step "daily AI health"
$daily = Join-Path $ScriptsDir "run_daily_ai_health.ps1"
if ($AutoRemediate) {
    & $daily -AutoRemediate
} else {
    & $daily
}
$healthExit = $LASTEXITCODE

Write-Step "done"
$latestPath = Join-Path $BrainRoot "state\daily_health\latest.json"
if (Test-Path $latestPath) {
    Write-Host ('Report: ' + $latestPath)
}
exit $healthExit
