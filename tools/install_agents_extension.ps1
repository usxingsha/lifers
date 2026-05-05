#Requires -Version 5.1
# Copy lifers-agents-ui to extension dirs.
# VSCodium (this repo): portable launcher rs\run_lifers_vscodium.bat uses --extensions-dir rs\data\extensions.
# System VSCodium / OSS Code: %USERPROFILE%\.vscode-oss\extensions
# Use -EditorTargets Vscodium to deploy only those two (skip Cursor / VS Code market paths).
param(
    [switch]$VerifyOnly,
    [ValidateSet('All', 'Vscodium')]
    [string]$EditorTargets = 'All'
)

$ErrorActionPreference = 'Stop'
$rsRoot = Split-Path -Parent $PSScriptRoot
if ($env:LIFERS_EXT_INSTALL_TARGETS -match '^(?i)vscodium$') {
    $EditorTargets = 'Vscodium'
}
$candidate = Join-Path $rsRoot 'lifers\extensions\lifers-agents-ui'
$legacy = Join-Path $rsRoot 'lifers_brain\extensions\lifers-agents-ui'
$src = if (Test-Path -LiteralPath (Join-Path $candidate 'package.json')) { $candidate } else { $legacy }
$pkgPath = Join-Path $src 'package.json'
if (-not (Test-Path -LiteralPath $pkgPath)) { throw "Missing $pkgPath" }

$utf8 = New-Object System.Text.UTF8Encoding $false
$pkgRaw = [IO.File]::ReadAllText($pkgPath, $utf8)
if ($pkgRaw -notmatch '"version"\s*:\s*"([^"]+)"') { throw "cannot parse version in $pkgPath" }
$pkgVersion = $Matches[1]
$pkgPublisher = if ($pkgRaw -match '"publisher"\s*:\s*"([^"]+)"') { $Matches[1] } else { 'lifers' }
$pkgName = if ($pkgRaw -match '"name"\s*:\s*"([^"]+)"') { $Matches[1] } else { 'lifers-agents-ui' }
$destFolder = "$pkgPublisher.$pkgName-$pkgVersion"

function Remove-Old-Lifers-Packages([string]$extParent) {
    if (-not (Test-Path -LiteralPath $extParent)) { return }
    Get-ChildItem -LiteralPath $extParent -Directory -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -like 'lifers.lifers-agents-ui-*' -and $_.Name -ne $destFolder } |
        ForEach-Object {
            Remove-Item -LiteralPath $_.FullName -Recurse -Force -ErrorAction SilentlyContinue
            Write-Host "Removed old $($_.Name)"
        }
}

function Copy-Ext([string]$dest) {
    $parent = Split-Path -Parent $dest
    if (-not (Test-Path -LiteralPath $parent)) { New-Item -ItemType Directory -Path $parent -Force | Out-Null }
    if (Test-Path -LiteralPath $dest) { Remove-Item -LiteralPath $dest -Recurse -Force }
    Copy-Item -LiteralPath $src -Destination $dest -Recurse -Force
    Write-Host "OK $dest"
}

<#
  VS Code family --extensions-dir reads extensions.json; manual folder copies do not update it.
  Stale entries (e.g. lifers-0.3.0 removed while json still points there) = extension "vanished".
#>
function Sync-LifersInExtensionsJson {
    param(
        [Parameter(Mandatory = $true)][string]$ExtensionsRoot,
        [Parameter(Mandatory = $true)][string]$DestFolderName,
        [Parameter(Mandatory = $true)][string]$Version
    )

    $extPath = Join-Path $ExtensionsRoot $DestFolderName
    if (-not (Test-Path -LiteralPath (Join-Path $extPath 'package.json'))) {
        Write-Warning "Skip extensions.json sync — missing $extPath"
        return
    }

    $full = (Resolve-Path -LiteralPath $extPath).Path
    $fsWin = $full -replace '/', '\'
    $external = ([Uri]$full).AbsoluteUri.TrimEnd('/')
    $pathProp = $full -replace '\\', '/'
    if ($pathProp -match '^([A-Za-z]):') {
        $pathProp = '/' + $Matches[1].ToLower() + $pathProp.Substring(2)
    }

    $entry = [ordered]@{
        identifier         = @{ id = 'lifers.lifers-agents-ui' }
        version            = $Version
        location           = [ordered]@{
            '$mid'   = 1
            fsPath   = $fsWin
            _sep     = 1
            external = $external
            path     = $pathProp
            scheme   = 'file'
        }
        relativeLocation = $DestFolderName
    }

    $jsonPath = Join-Path $ExtensionsRoot 'extensions.json'
    $rest = @()
    if (Test-Path -LiteralPath $jsonPath) {
        try {
            $raw = Get-Content -LiteralPath $jsonPath -Raw -Encoding UTF8
            if ($raw.Length -gt 0 -and [int][char]$raw[0] -eq 0xFEFF) { $raw = $raw.Substring(1) }
            $parsed = $raw | ConvertFrom-Json
            $arr = @($parsed)
            if ($parsed -isnot [System.Array]) { $arr = @($parsed) }
            $rest = @( $arr | Where-Object { $_ -and $_.identifier -and $_.identifier.id -ne 'lifers.lifers-agents-ui' } )
        }
        catch {
            $rest = @()
        }
    }

    $merged = @($rest) + @($entry)
    $segments = foreach ($m in $merged) {
        ConvertTo-Json -InputObject $m -Depth 12 -Compress
    }
    ('[' + ($segments -join ',') + ']') | Set-Content -LiteralPath $jsonPath -Encoding UTF8
    Write-Host "OK extensions.json ($ExtensionsRoot)"
}

function Get-ExtParents {
    param([string]$Mode, [string]$RsRoot)
    $port = Join-Path $RsRoot 'data\extensions'
    $oss = Join-Path $env:USERPROFILE '.vscode-oss\extensions'
    $vscode = Join-Path $env:USERPROFILE '.vscode\extensions'
    $cursor = Join-Path $env:USERPROFILE '.cursor\extensions'
    if ($Mode -eq 'Vscodium') {
        return @($port, $oss)
    }
    return @($port, $oss, $vscode, $cursor)
}

function Test-Ext {
    param([string]$Mode, [string]$RsRoot, [string]$DestFolderName)
    $ok = $false
    foreach ($parent in (Get-ExtParents -Mode $Mode -RsRoot $RsRoot)) {
        $d = Join-Path $parent $DestFolderName
        if ((Test-Path (Join-Path $d 'package.json')) -and (Test-Path (Join-Path $d 'extension.js'))) {
            Write-Host "FOUND $d"
            $ok = $true
        }
    }
    if (-not $ok) {
        throw "Extension not found in expected dirs for -EditorTargets $Mode. Run without -VerifyOnly."
    }
}

if ($VerifyOnly) {
    Test-Ext -Mode $EditorTargets -RsRoot $rsRoot -DestFolderName $destFolder
    Write-Host 'VERIFY OK'
    exit 0
}

$extParents = Get-ExtParents -Mode $EditorTargets -RsRoot $rsRoot
foreach ($extParent in $extParents) {
    Remove-Old-Lifers-Packages $extParent
}

foreach ($extParent in $extParents) {
    Copy-Ext (Join-Path $extParent $destFolder)
}

foreach ($extRoot in $extParents) {
    Sync-LifersInExtensionsJson -ExtensionsRoot $extRoot -DestFolderName $destFolder -Version $pkgVersion
}

Write-Host 'Done. VSCodium: Ctrl+Shift+P -> Developer: Reload Window.'
Write-Host '  Portable: rs\run_lifers_vscodium.bat uses rs\data\extensions for extensions.'
if ($EditorTargets -eq 'All') {
    Write-Host '  (Also copied to .vscode\extensions and .cursor\extensions.)'
}
Test-Ext -Mode $EditorTargets -RsRoot $rsRoot -DestFolderName $destFolder
