#Requires -Version 5.1
<#
.SYNOPSIS
  将当前 Lifers 根目录（lifers 或 lifers_brain）打成 tar.gz，便于 scp 到 Kali 解压后跑 scripts/kali_train_weights.sh。

.EXAMPLE
  cd C:\...\rs\lifers_brain\scripts
  .\package_rs_for_kali.ps1
  scp ..\dist\lifers_kali.tar.gz user@kali:~/ 
#>
$ErrorActionPreference = "Stop"
$BrainRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Parent = Split-Path $BrainRoot -Parent
$Leaf = Split-Path $BrainRoot -Leaf
$Dist = Join-Path $BrainRoot "dist"
New-Item -ItemType Directory -Force -Path $Dist | Out-Null
$Out = Join-Path $Dist "lifers_kali.tar.gz"
if (Test-Path $Out) { Remove-Item $Out -Force }

Push-Location $Parent
try {
  # Windows 10+ bsdtar: exclude venv/cache/git; shrink tarball
  $ex1 = "--exclude=$Leaf/.venv"
  $ex2 = "--exclude=$Leaf/**/__pycache__"
  $ex3 = "--exclude=$Leaf/.git"
  $ex4 = "--exclude=$Leaf/dist/*.tar.gz"
  # 避免把 Windows 上的小权重覆盖 Kali 上已训练数小时的大 JSON
  $ex5 = "--exclude=$Leaf/weights"
  # 不把本机 API 密钥打进包（仍可用 OS 环境变量在 Kali 上配置）
  $ex6 = "--exclude=$Leaf/config/secrets.env"
  & tar -czf $Out $ex1 $ex2 $ex3 $ex4 $ex5 $ex6 $Leaf
  if ($LASTEXITCODE -ne 0) { throw "tar exit $LASTEXITCODE" }
}
finally {
  Pop-Location
}

Write-Host "Wrote $Out"
Write-Host "Kali 示例:"
Write-Host "  mkdir -p ~/lifers && tar -xzf lifers_kali.tar.gz -C ~/lifers"
Write-Host "  cd ~/lifers/$Leaf && bash scripts/kali_train_weights.sh"
