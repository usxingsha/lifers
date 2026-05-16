# link_lifers_app.ps1
# Creates a directory junction from lifers → lifers (legacy compat)
# Run as: powershell -File tools\link_lifers_app.ps1

$ROOT   = Split-Path $PSScriptRoot -Parent
$TARGET = Join-Path $ROOT "lifers"
$LINK   = Join-Path $ROOT "lifers"

Write-Host "=== Lifers Junction Linker ===" -ForegroundColor Cyan
Write-Host "Target : $TARGET"
Write-Host "Link   : $LINK"

if (-not (Test-Path $TARGET)) {
    Write-Error "Target does not exist: $TARGET"
    exit 1
}

if (Test-Path $LINK) {
    $item = Get-Item $LINK -Force
    if ($item.Attributes -band [System.IO.FileAttributes]::ReparsePoint) {
        Write-Host "Junction already exists at $LINK" -ForegroundColor Green
        exit 0
    } else {
        Write-Warning "$LINK exists and is NOT a junction — skipping to avoid data loss"
        exit 1
    }
}

try {
    New-Item -ItemType Junction -Path $LINK -Target $TARGET | Out-Null
    Write-Host "Junction created: $LINK -> $TARGET" -ForegroundColor Green
} catch {
    Write-Error "Failed to create junction: $_"
    exit 1
}
