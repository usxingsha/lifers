#Requires -Version 5.1
<#
  将 sibling 目录 ..\lifers（独立应用仓）接到便携根下的 lifers junction（NTFS）。
  便携根文件夹历史上曾名为 rs；不改变 lifers_brain。

  用法（在便携根目录执行）:
    powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\link_lifers_app.ps1

  自定义 lifers 路径（可选）:
    $env:LIFERS_APP_ROOT = "C:\Users\...\curku\lifers"
#>
$ErrorActionPreference = "Stop"
$rsRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$defaultTarget = Join-Path $rsRoot "..\lifers"
if ($env:LIFERS_APP_ROOT) {
  $target = $env:LIFERS_APP_ROOT.TrimEnd('\', '/')
} else {
  try {
    $target = (Resolve-Path -LiteralPath $defaultTarget).Path
  } catch {
    $target = $null
  }
}
if (-not $target -or -not (Test-Path -LiteralPath $target)) {
  Write-Error "Lifers app folder not found. Expected: $defaultTarget  Set env:LIFERS_APP_ROOT to override."
}
$linkPath = Join-Path $rsRoot "lifers"

if (Test-Path -LiteralPath $linkPath) {
  $item = Get-Item -LiteralPath $linkPath -ErrorAction SilentlyContinue
  if ($item.Attributes -band [IO.FileAttributes]::ReparsePoint) {
    Write-Host "Already linked: $linkPath -> $target"
    if (Get-Command python -ErrorAction SilentlyContinue) {
      & python "$rsRoot\tools\materialize_integrated_workspace.py"
    }
    exit 0
  }
  Write-Error "Path exists and is not a junction: $linkPath  Remove or rename it first."
}

$mk = "cmd.exe /c mklink /J `"$linkPath`" `"$target`""
Invoke-Expression $mk | Out-Null
if (-not (Test-Path -LiteralPath $linkPath)) {
  Write-Error "mklink failed. Try running shell as Administrator, or enable Developer Mode for symlink privileges."
}
Write-Host "OK: $linkPath -> $target"
if (Get-Command python -ErrorAction SilentlyContinue) {
  & python "$rsRoot\tools\materialize_integrated_workspace.py"
}
