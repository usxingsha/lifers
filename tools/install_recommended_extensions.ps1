#Requires -Version 5.1
# Install extensions listed in .vscode/extensions.json -> recommendations into:
#   - data/extensions (portable)
#   - %USERPROFILE%/.cursor/extensions (system Cursor without portable flags)
# Requires Cursor CLI (Cursor.exe --install-extension). Offline: use -SkipMarketplace in bootstrap.
param(
    [switch]$SkipMarketplace,
    [switch]$PortableOnly
)

$ErrorActionPreference = 'Continue'
$portableRoot = Split-Path -Parent $PSScriptRoot
$extJson = Join-Path $portableRoot '.vscode\extensions.json'
if (-not (Test-Path -LiteralPath $extJson)) {
    Write-Host "Skip: no $extJson"
    exit 0
}

$json = Get-Content -LiteralPath $extJson -Encoding UTF8 -Raw | ConvertFrom-Json
$ids = @($json.recommendations)
if (-not $ids -or $ids.Count -eq 0) {
    Write-Host 'No recommendations in extensions.json'
    exit 0
}

if ($SkipMarketplace) {
    Write-Host 'SkipMarketplace: not installing marketplace extensions.'
    exit 0
}

function Find-CursorExe {
    $pf86 = ${env:ProgramFiles(x86)}
    @(
        (Join-Path $portableRoot 'shell\Cursor\Cursor.exe'),
        (Join-Path $env:LOCALAPPDATA 'Programs\cursor\Cursor.exe'),
        (Join-Path $env:ProgramFiles 'Cursor\Cursor.exe'),
        (Join-Path $pf86 'Cursor\Cursor.exe')
    ) | Where-Object { $_ -and (Test-Path -LiteralPath $_) } | Select-Object -First 1
}

function Ensure-Dir([string]$p) {
    $parent = Split-Path -Parent $p
    if ($parent -and -not (Test-Path -LiteralPath $parent)) {
        New-Item -ItemType Directory -Path $parent -Force | Out-Null
    }
}

function Install-Into {
    param(
        [string]$CursorExe,
        [string]$ExtDir,
        [string]$UserDataDir,
        [string[]]$ExtensionIds
    )
    Ensure-Dir $ExtDir
    Ensure-Dir $UserDataDir
    foreach ($id in $ExtensionIds) {
        Write-Host "Installing $id -> $ExtDir"
        $proc = Start-Process -FilePath $CursorExe `
            -ArgumentList @(
            '--extensions-dir', $ExtDir,
            '--user-data-dir', $UserDataDir,
            '--install-extension', $id,
            '--force'
        ) `
            -Wait -PassThru -NoNewWindow
        if ($proc.ExitCode -ne 0) {
            Write-Warning "Exit $($proc.ExitCode) for $id (CLI may be unsupported on this Cursor build)."
        }
    }
}

$cursor = Find-CursorExe
if (-not $cursor) {
    Write-Warning 'Cursor.exe not found. Install Cursor or place portable build at shell\Cursor\Cursor.exe'
    Write-Host 'You can still install recommendations from the Extensions view when the workspace opens.'
    exit 0
}

$portableExt = Join-Path $portableRoot 'data\extensions'
$portableUd = Join-Path $portableRoot 'data\cursor-user-data'

Install-Into -CursorExe $cursor -ExtDir $portableExt -UserDataDir $portableUd -ExtensionIds $ids

if (-not $PortableOnly) {
    $userExt = Join-Path $env:USERPROFILE '.cursor\extensions'
    $userUd = Join-Path $env:USERPROFILE '.cursor'
    if (-not (Test-Path -LiteralPath $userUd)) {
        New-Item -ItemType Directory -Path $userUd -Force | Out-Null
    }
    Install-Into -CursorExe $cursor -ExtDir $userExt -UserDataDir $userUd -ExtensionIds $ids
}

Write-Host 'Recommended extensions install pass finished.'
