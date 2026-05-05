#Requires -Version 5.1
<#
.SYNOPSIS
  从 Kali 上的 Lifers 智脑目录拉取训练权重等到本机 lifers_brain/weights（scp）。

.EXAMPLE
  cd lifers_brain\scripts
  .\sync_weights_from_kali.ps1
  .\sync_weights_from_kali.ps1 -KaliHost 192.168.234.152 -Force

.NOTES
  - 默认仅当远端 lifers_transformer.json 比本地新（mtime）或本地缺失时才拉取大文件； -Force 强制覆盖。
  - 大块权重可用 SFTP 压缩：scp 自带 -C。
  - 「完成一次同步一次」：训练段落结束或 pause 后在本机执行本脚本即可。
#>
param(
    [string] $KaliUser = "kali",
    [string] $KaliHost = "192.168.234.152",
    [string] $RemoteBrain = "/home/kali/lifers/lifers_brain",
    [switch] $Force,
    [switch] $IncludeTinyBackup,
    [switch] $IncludeCheckpoints
)

$ErrorActionPreference = "Stop"
$BrainRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Weights = Join-Path $BrainRoot "weights"
New-Item -ItemType Directory -Force -Path $Weights | Out-Null

$sshTarget = "${KaliUser}@${KaliHost}"
$rw = "$RemoteBrain/weights"

function Invoke-Ssh([string] $Cmd) {
    $out = & ssh -o BatchMode=yes -o ConnectTimeout=30 $sshTarget $Cmd 2>&1
    if ($LASTEXITCODE -ne 0) { throw "ssh failed: $out" }
    return $out
}

function RemoteStat([string] $Rel) {
    $p = "$rw/$Rel".Replace("\", "/")
    $line = Invoke-Ssh "stat -c '%Y %s' `"$p`" 2>/dev/null || echo MISSING"
    if ($line -match "MISSING") { return $null }
    $parts = ($line -split "\s+", 2)
    return @{ Mtime = [long]$parts[0]; Size = [long]$parts[1] }
}

function ShouldPullLargeTransformer {
    $remote = RemoteStat "lifers_transformer.json"
    if (-not $remote) { Write-Host "[skip] remote lifers_transformer.json missing"; return $false }
    $local = Join-Path $Weights "lifers_transformer.json"
    if (-not (Test-Path $local)) { Write-Host "[pull] local missing"; return $true }
    if ($Force) { Write-Host "[pull] -Force"; return $true }
    try {
        $lt = ([System.IO.File]::GetLastWriteTimeUtc($local) - [datetime]'1970-01-01').TotalSeconds
    } catch {
        $lt = 0
    }
    if ([long]$remote.Mtime -gt [long]$lt) {
        Write-Host "[pull] remote newer (remote_mtime=$($remote.Mtime) vs local~=$([int]$lt))"
        return $true
    }
    Write-Host "[skip] local lifers_transformer.json is same or newer than remote"
    return $false
}

function Scp-File([string] $Rel) {
    $src = "${sshTarget}:$rw/$Rel".Replace("\", "/")
    $dst = Join-Path $Weights (Split-Path $Rel -Leaf)
    Write-Host "scp -C $Rel ..."
    & scp -C -o BatchMode=yes -o ConnectTimeout=60 $src $dst
    if ($LASTEXITCODE -ne 0) { throw "scp failed for $Rel" }
}

Write-Host "=== Lifers weights sync from Kali ==="
Write-Host "remote $rw"

$small = @(
    ".train_control",
    "lifers_markov.json",
    ".lifers_train_state.json"
)
foreach ($f in $small) {
    $st = RemoteStat $f
    if (-not $st) {
        Write-Host "[skip] $f not on remote"
        continue
    }
    Scp-File $f
}

if (ShouldPullLargeTransformer) {
    Scp-File "lifers_transformer.json"
} elseif (-not $Force) {
    Write-Host "(use -Force to re-download lifers_transformer.json anyway)"
}

if ($IncludeTinyBackup) {
    $st = RemoteStat "tiny_transformer_v001.json"
    if ($st) { Scp-File "tiny_transformer_v001.json" }
}

if ($IncludeCheckpoints) {
    Write-Host "=== checkpoints (optional bulk) ==="
    Invoke-Ssh "test -d $rw/checkpoints && tar czf /tmp/lifers_ckpt.tgz -C $rw checkpoints || true" | Out-Null
    $has = Invoke-Ssh "test -f /tmp/lifers_ckpt.tgz && echo yes || echo no"
    if ($has -match "yes") {
        & scp -C "${sshTarget}:/tmp/lifers_ckpt.tgz" $Weights
        Push-Location $Weights
        try { tar -xzf lifers_ckpt.tgz 2>$null } finally { Pop-Location }
        Remove-Item (Join-Path $Weights "lifers_ckpt.tgz") -Force -ErrorAction SilentlyContinue
        Write-Host "checkpoints extracted under weights/checkpoints"
    } else {
        Write-Host "no checkpoints dir on remote"
    }
}

Write-Host "Done. Local: $Weights"
