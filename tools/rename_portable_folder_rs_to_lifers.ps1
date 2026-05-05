#Requires -Version 5.1
<#
  将父目录下的便携根文件夹 rs 重命名为 lifers（与 GitHub 仓库根目录名一致）。
  请先关闭 Cursor/VS Code 等占用该路径的程序后再运行。

  用法（推荐）:
    powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\rename_portable_folder_rs_to_lifers.ps1

  可选参数:
    -ParentPath "C:\Users\...\Desktop\curku"
#>
param(
  [string] $ParentPath = ""
)
$ErrorActionPreference = "Stop"
if (-not $ParentPath) {
  $ParentPath = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
}
$src = Join-Path $ParentPath "rs"
$dst = Join-Path $ParentPath "lifers"
if (-not (Test-Path -LiteralPath $src)) {
  Write-Host "Skip: $src not found (already lifers or different layout)."
  exit 0
}
if (Test-Path -LiteralPath $dst) {
  Write-Error "Target exists: $dst — remove or rename it first."
}
Rename-Item -LiteralPath $src -NewName "lifers"
Write-Host "OK: $src -> $dst"
