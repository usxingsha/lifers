#Requires -Version 5.1
# openclaw/openclaw → rs/third_party/openclaw：优先 git 子模块，否则浅克隆（只读对照，不跑 npm 构建）
$ErrorActionPreference = "Stop"
$RsRoot = Split-Path -Parent $PSScriptRoot
$Dest = Join-Path $RsRoot "third_party\openclaw"
$Url = "https://github.com/openclaw/openclaw.git"
$Gitmodules = Join-Path $RsRoot ".gitmodules"
New-Item -ItemType Directory -Force -Path (Split-Path $Dest) | Out-Null
if (!(Get-Command git -ErrorAction SilentlyContinue)) {
  Write-Error "git not found in PATH"
  exit 1
}
if (Test-Path $Gitmodules) {
  Write-Host "Initializing submodule third_party/openclaw from .gitmodules"
  Push-Location $RsRoot
  try {
    & git submodule update --init --depth 1 third_party/openclaw
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
  }
  finally {
    Pop-Location
  }
  Write-Host "OK: submodule $Dest"
  exit 0
}
if (Test-Path (Join-Path $Dest ".git")) {
  Write-Host "Already present: $Dest (git pull optional)"
  exit 0
}
& git clone --depth 1 $Url $Dest
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Write-Host "OK: $Dest"
