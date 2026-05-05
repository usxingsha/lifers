#Requires -Version 5.1
<#
  将独立 Lifers 桌面应用仓接到便携根下的 lifers junction（NTFS），供扩展/壳查找。
  便携根目录建议命名为 lifers（与 GitHub 一致）；不改变 lifers_brain。

  兄弟目录候选（首个存在且不等于便携根本身则用）：lifers-app、lifers-desktop、lifers。
  若便携根文件夹已名为 lifers，则 ..\lifers 会与自身重合，脚本会自动跳过该项。

  用法（在便携根目录执行）:
    powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\link_lifers_app.ps1

  自定义应用仓路径:
    $env:LIFERS_APP_ROOT = "D:\path\to\lifers-desktop-repo"
#>
$ErrorActionPreference = "Stop"
$rsRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$parentDir = Split-Path -Parent $rsRoot
$rsResolved = (Resolve-Path -LiteralPath $rsRoot).Path
$candidates = @(
  (Join-Path $parentDir "lifers-app"),
  (Join-Path $parentDir "lifers-desktop"),
  (Join-Path $parentDir "lifers")
)
if ($env:LIFERS_APP_ROOT) {
  $target = $env:LIFERS_APP_ROOT.TrimEnd('\', '/')
} else {
  $target = $null
  foreach ($c in $candidates) {
    if (-not (Test-Path -LiteralPath $c)) { continue }
    try {
      $cr = (Resolve-Path -LiteralPath $c).Path
    } catch {
      continue
    }
    if ($cr -ieq $rsResolved) { continue }
    $target = $cr
    break
  }
}
if (-not $target -or -not (Test-Path -LiteralPath $target)) {
  Write-Error "Lifers desktop app folder not found under $parentDir (tried lifers-app, lifers-desktop, lifers excluding portable root). Set env:LIFERS_APP_ROOT to override."
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
