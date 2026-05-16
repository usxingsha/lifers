#Requires -Version 5.1
<#
.SYNOPSIS
  从 Lifers **仓库根目录** 打 tar.gz（Windows→Kali 合并同步，不经由 git）；含 lifers、third_party（OpenClaw 检出源码、claw_code_rust）、tools、根配置等。

.DESCRIPTION
  排除：.git、子模块内 .git、node_modules、各 .venv、Rust target、打包产物目录、可选 weights、体积巨大的 data/ 与 shell/。
  OpenClaw 以工作区已检出的文件树打入包（子模块先 git submodule update）。

.PARAMETER IncludeWeights
  为 true 时包含 lifers/weights（可能极大，会覆盖 Kali 同名文件）。

.PARAMETER IncludeData
  为 true 时包含仓库根下 data/（通常含编辑器缓存，体积极大，默认排除）。

.EXAMPLE
  cd lifers\scripts
  .\package_rs_for_kali.ps1
  .\package_rs_for_kali.ps1 -IncludeWeights
#>
param(
  [switch] $IncludeWeights,
  [switch] $IncludeData
)

$ErrorActionPreference = "Stop"
$BrainRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$RepoRoot = (Resolve-Path (Join-Path $BrainRoot "..")).Path
$Dist = Join-Path $BrainRoot "dist"
New-Item -ItemType Directory -Force -Path $Dist | Out-Null
$Out = Join-Path $Dist "lifers_kali.tar.gz"
if (Test-Path $Out) { Remove-Item $Out -Force }

$oc = Join-Path $RepoRoot "third_party\openclaw\package.json"
if (-not (Test-Path -LiteralPath $oc)) {
  Write-Host "=== git submodule update (openclaw) ===" -ForegroundColor Cyan
  Push-Location $RepoRoot
  try {
    git submodule update --init --recursive --depth 1
    if ($LASTEXITCODE -ne 0) { throw "git submodule update failed" }
  } finally {
    Pop-Location
  }
}

$Leaf = Split-Path $BrainRoot -Leaf
if ($Leaf -ne "lifers") { throw "expected lifers as parent of scripts, got $Leaf" }

Push-Location $RepoRoot
try {
  $ex = @(
    "--exclude=.git",
    "--exclude=third_party/openclaw/.git",
    "--exclude=third_party/openclaw/node_modules",
    "--exclude=third_party/claw_code_rust/target",
    "--exclude=third_party/_refs",
    "--exclude=lifers/.venv",
    "--exclude=lifers/dist",
    "--exclude=__pycache__",
    "--exclude=lifers/config/secrets.env",
    "--exclude=**/node_modules",
    "--exclude=shell",
    "--exclude=.cursor"
  )
  if (-not $IncludeData) {
    $ex += "--exclude=data"
  }
  if (-not $IncludeWeights) {
    $ex += "--exclude=lifers/weights"
  }
  Write-Host "tar -C $RepoRoot (repo root, excludes: $($ex.Count) rules)" -ForegroundColor Cyan
  & tar -czf $Out @ex .
  if ($LASTEXITCODE -ne 0) { throw "tar exit $LASTEXITCODE" }
}
finally {
  Pop-Location
}

Write-Host "Wrote $Out"
if (-not $IncludeWeights) {
  Write-Host "Note: lifers/weights excluded (use -IncludeWeights to pack and overwrite Kali weights)."
}
if (-not $IncludeData) {
  Write-Host "Note: data/ excluded (use -IncludeData to include editor cache tree)."
}
Write-Host "Kali merge: mkdir -p ~/lifers && tar -xzf lifers_kali.tar.gz -C ~/lifers"
