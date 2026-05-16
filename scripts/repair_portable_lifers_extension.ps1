#Requires -Version 5.1
# rs\data\extensions：删除旧 Lifers 目录、同步扩展、用 Python 合并写入 extensions.json（与其它扩展并存时更安全）。
param([string] $RsRoot = "")
$ErrorActionPreference = "Stop"
if (-not $RsRoot) {
  $RsRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
}
$DataExt = Join-Path $RsRoot "data\extensions"
$ExtSrc = Join-Path $RsRoot "lifers\extensions\lifers-agents-ui"
$repairPy = Join-Path $RsRoot "lifers\scripts\repair_lifers_extensions_index.py"
if (!(Test-Path (Join-Path $ExtSrc "package.json"))) { throw "missing $ExtSrc" }
if (!(Test-Path $repairPy)) { throw "missing $repairPy" }

$utf8 = New-Object System.Text.UTF8Encoding $false
$raw = [IO.File]::ReadAllText((Join-Path $ExtSrc "package.json"), $utf8)
if ($raw -notmatch '"version"\s*:\s*"([^"]+)"') { throw "no version in package.json" }
$ver = $Matches[1]
$bundle = "lifers.lifers-agents-ui-$ver"
$dest = Join-Path $DataExt $bundle

$pyCmd = @(Get-Command python -ErrorAction SilentlyContinue; Get-Command python3 -ErrorAction SilentlyContinue) | Select-Object -First 1
if (-not $pyCmd) { throw "need python for repair_lifers_extensions_index.py" }

New-Item -ItemType Directory -Force -Path $DataExt | Out-Null
Get-ChildItem -LiteralPath $DataExt -Directory -ErrorAction SilentlyContinue |
  Where-Object { $_.Name -like "lifers.lifers-agents-ui-*" } |
  ForEach-Object { Remove-Item -LiteralPath $_.FullName -Recurse -Force; Write-Host "removed $($_.Name)" }

& robocopy.exe $ExtSrc $dest /MIR /NFL /NDL /NJH /NJS /nc /ns /np
if ($LASTEXITCODE -ge 8) { throw "robocopy failed $LASTEXITCODE" }

& $pyCmd.Source $repairPy $DataExt $ver
if ($LASTEXITCODE -ne 0) { throw "repair index failed" }
Write-Host "OK portable $bundle"
