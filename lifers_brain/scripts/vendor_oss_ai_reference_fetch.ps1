#Requires -Version 5.1
<#
.SYNOPSIS
  按 oss_ai_ecosystem_manifest.json 中标记为 optional_shallow_clone 的条目，浅克隆到 third_party/_refs/（默认被 .gitignore，避免误把巨型上游整树推入私仓）。

.EXAMPLE
  cd lifers_brain\scripts
  .\vendor_oss_ai_reference_fetch.ps1 -IncludeIds nanoGPT
#>
param(
    [string[]] $IncludeIds = @()
)

$ErrorActionPreference = "Stop"
$Brain = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Manifest = Join-Path $Brain "config\oss_ai_ecosystem_manifest.json"
if (-not (Test-Path $Manifest)) { throw "missing $Manifest" }

$raw = Get-Content -LiteralPath $Manifest -Raw -Encoding UTF8
$j = $raw | ConvertFrom-Json
$outRoot = Join-Path (Split-Path $Brain -Parent) "third_party\_refs"
New-Item -ItemType Directory -Force -Path $outRoot | Out-Null

$cloneable = @{}
foreach ($p in $j.projects) {
    if ($p.integration -eq "optional_shallow_clone" -and $p.github -match '^https://github.com/') {
        $cloneable[$p.id] = $p.github
    }
}

if ($IncludeIds.Count -eq 0) {
    Write-Host "No -IncludeIds. Optional clone ids:" -ForegroundColor Cyan
    $cloneable.Keys | Sort-Object | ForEach-Object { Write-Host "  $_ -> $($cloneable[$_])" }
    Write-Host "Example: .\vendor_oss_ai_reference_fetch.ps1 -IncludeIds nanoGPT"
    exit 0
}

foreach ($id in $IncludeIds) {
    if (-not $cloneable.ContainsKey($id)) {
        Write-Warning "Skip $id (not optional_shallow_clone in manifest)"
        continue
    }
    $url = $cloneable[$id]
    $dest = Join-Path $outRoot $id
    if (Test-Path $dest) {
        Write-Host "Exists: $dest (remove manually to re-fetch)"
        continue
    }
    Write-Host "git clone --depth 1 $url -> $dest"
    git clone --depth 1 $url $dest
    if ($LASTEXITCODE -ne 0) { throw "git clone failed for $id" }
}
Write-Host "Done. third_party\_refs is gitignored by default."
