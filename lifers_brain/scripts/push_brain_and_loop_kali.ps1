#Requires -Version 5.1
<#
.SYNOPSIS
  打包 lifers_brain（不含 weights）、scp 到 Kali、解压并启动 Markov + escalate 长期循环（tmux lifers-stack）。

.EXAMPLE
  cd C:\...\rs\lifers_brain\scripts
  .\push_brain_and_loop_kali.ps1
  .\push_brain_and_loop_kali.ps1 -KaliHost "kali@192.168.234.152" -SkipBootstrap
#>
param(
  [string] $KaliHost = "kali@192.168.234.152",
  [string] $SshKey = "$env:USERPROFILE\.ssh\id_ed25519",
  [switch] $SkipPackage,
  [switch] $SkipBootstrap
)

$ErrorActionPreference = "Stop"
$BrainRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Dist = Join-Path $BrainRoot "dist"
$Tar = Join-Path $Dist "lifers_kali.tar.gz"
$ssh = "ssh"
$scp = "scp"
$sshOpts = @("-o", "BatchMode=yes", "-o", "ConnectTimeout=25")
if (Test-Path -LiteralPath $SshKey) {
  $sshOpts = @("-i", $SshKey) + $sshOpts
}

if (!$SkipPackage) {
  & powershell.exe -NoProfile -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot "package_rs_for_kali.ps1")
}
if (!(Test-Path -LiteralPath $Tar)) { throw "missing tarball: $Tar" }

$remoteTar = "/tmp/lifers_kali_push.tar.gz"
Write-Host "scp -> ${KaliHost}:$remoteTar"
& $scp @sshOpts $Tar "${KaliHost}:$remoteTar"
if ($LASTEXITCODE -ne 0) { throw "scp failed" }

$extractAndRun = @'
set -euo pipefail
DEST="$HOME/lifers"
mkdir -p "$DEST"
tar -xzf /tmp/lifers_kali_push.tar.gz -C "$DEST"
rm -f /tmp/lifers_kali_push.tar.gz
BR="$DEST/lifers_brain"
if [[ ! -f "$BR/scripts/remote_kali_bootstrap_train_loop.sh" ]]; then
  echo "extract failed: no $BR/scripts/remote_kali_bootstrap_train_loop.sh" >&2
  exit 1
fi
chmod +x "$BR/scripts/remote_kali_bootstrap_train_loop.sh"
'@

if ($SkipBootstrap) {
  $remote = $extractAndRun + @'

echo "SkipBootstrap: extracted only. Run on Kali:"
echo "  bash $BR/scripts/remote_kali_bootstrap_train_loop.sh"
'@
} else {
  $remote = $extractAndRun + @'

exec bash "$BR/scripts/remote_kali_bootstrap_train_loop.sh"
'@
}

$remoteUnix = ($remote -replace "`r`n", "`n").TrimEnd() + "`n"
$b64 = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($remoteUnix))
& $ssh @sshOpts $KaliHost "echo $b64 | base64 -d | bash -s --"
if ($LASTEXITCODE -ne 0) { throw "remote extract/bootstrap failed" }

Write-Host "OK. On Kali: tmux attach -t lifers-stack   |   tail -f ~/lifers_full_stack.log"
