# rename_portable_folder_rs_to_lifers.ps1
# Renames the portable root folder from "rs" to "lifers" safely.
# Run from Desktop\curku: powershell -File rs\tools\rename_portable_folder_rs_to_lifers.ps1

$CURKU  = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$OLD    = Join-Path $CURKU "rs"
$NEW    = Join-Path $CURKU "lifers"

Write-Host "=== Rename rs -> lifers ===" -ForegroundColor Cyan
Write-Host "From: $OLD"
Write-Host "To  : $NEW"

if (-not (Test-Path $OLD)) {
    Write-Host "Folder 'rs' not found — already renamed or wrong directory." -ForegroundColor Green
    exit 0
}
if (Test-Path $NEW) {
    Write-Error "'lifers' already exists at $NEW — rename aborted."
    exit 1
}

try {
    Rename-Item -Path $OLD -NewName "lifers"
    Write-Host "Renamed successfully." -ForegroundColor Green
    Write-Host "Please reopen the workspace from: $NEW\lifers.code-workspace"
} catch {
    Write-Error "Rename failed (close all editors first): $_"
    exit 1
}
