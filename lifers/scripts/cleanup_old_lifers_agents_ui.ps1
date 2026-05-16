#Requires -Version 5.1
<#
.SYNOPSIS
  删除本机 .vscode / .vscode-oss / .cursor 下所有 lifers.lifers-agents-ui-*，仅保留 package.json 指定版本目录（若不存在则会在下次 sync 时由 robocopy 创建）。
#>
param([string] $BrainRoot = "")

$ErrorActionPreference = "Stop"
if (-not $BrainRoot) {
  $BrainRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
}
$extSrc = Join-Path $BrainRoot "extensions\lifers-agents-ui"
$pkgPath = Join-Path $extSrc "package.json"
$utf8 = New-Object System.Text.UTF8Encoding $false
$raw = [IO.File]::ReadAllText($pkgPath, $utf8)
if ($raw -notmatch '"version"\s*:\s*"([^"]+)"') { throw "cannot parse version from $pkgPath" }
$ver = $Matches[1]
$keep = "lifers.lifers-agents-ui-$ver"
Write-Host "Removing all lifers.lifers-agents-ui-* except $keep"

$roots = @(
  (Join-Path $env:USERPROFILE ".vscode\extensions"),
  (Join-Path $env:USERPROFILE ".vscode-oss\extensions"),
  (Join-Path $env:USERPROFILE ".cursor\extensions")
)
foreach ($root in $roots) {
  if (!(Test-Path $root)) { continue }
  Get-ChildItem -LiteralPath $root -Directory -ErrorAction SilentlyContinue |
    Where-Object { $_.Name -like "lifers.lifers-agents-ui-*" -and $_.Name -ne $keep } |
    ForEach-Object {
      Remove-Item -LiteralPath $_.FullName -Recurse -Force
      Write-Host "removed $($_.FullName)"
    }
}
