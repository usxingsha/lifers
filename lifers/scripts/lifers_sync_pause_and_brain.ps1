#Requires -Version 5.1
<#
.SYNOPSIS
  暂停 Kali 训练后，将本仓库全量打成 tar 并 scp 合并到远端 ~/lifers（不自动再起 tmux 引导）。

.DESCRIPTION
  对 push_brain_and_loop_kali.ps1 传入 -SkipBootstrap。默认先 remote_pause_lifers_train.sh（与 push / sync_weights / UI 同步一致）。

.EXAMPLE
  cd lifers\scripts
  .\lifers_sync_pause_and_brain.ps1
  .\lifers_sync_pause_and_brain.ps1 -KaliHost "kali@10.0.0.5" -SkipPackage
#>
param(
  [string] $KaliHost = "kali@192.168.234.152",
  [string] $SshKey = "$env:USERPROFILE\.ssh\id_ed25519",
  [switch] $SkipPackage,
  [switch] $SkipPauseTrainFirst
)

$ErrorActionPreference = "Stop"
$push = Join-Path $PSScriptRoot "push_brain_and_loop_kali.ps1"
if (-not (Test-Path -LiteralPath $push)) { throw "missing $push" }

Write-Host ">> lifers_sync_pause_and_brain (pause + pack + scp, -SkipBootstrap)" -ForegroundColor Cyan
& powershell.exe -NoProfile -ExecutionPolicy Bypass -File $push `
  -KaliHost $KaliHost `
  -SshKey $SshKey `
  -SkipBootstrap `
  -SkipPackage:$SkipPackage `
  -SkipPauseTrainFirst:$SkipPauseTrainFirst
