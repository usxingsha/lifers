#Requires -Version 5.1
<#
.SYNOPSIS
  从 Kali 上的 Lifers 智脑目录拉取训练权重等到本机 lifers_brain/weights（scp）。

.EXAMPLE
  cd lifers_brain\scripts
  .\sync_weights_from_kali.ps1
  .\sync_weights_from_kali.ps1 -KaliHost 192.168.234.152 -Force
  .\sync_weights_from_kali.ps1 -Watch
  .\sync_weights_from_kali.ps1 -WatchIntervalSec 300
  .\sync_weights_from_kali.ps1 -SkipTrainPause   # 拉权重前不暂停远端训练（慎用）

.NOTES
  - 默认仅当远端 lifers_transformer.json 比本地新（mtime）或本地缺失时才拉取大文件； -Force 强制覆盖。
  - 大块权重可用 SFTP 压缩：scp 自带 -C。
  - 默认在拉取前 **暂停远端训练**（remote_pause_lifers_train.sh，与 push / UI 同步一致）；-SkipTrainPause 跳过。
  - Watch 模式下每次轮询前也会 pause（避免与写权重竞争）；短间隔可配 -SkipTrainPause。
#>
param(
    [string] $KaliUser = "kali",
    [string] $KaliHost = "192.168.234.152",
    [string] $RemoteBrain = "/home/kali/lifers/lifers_brain",
    [string] $SshKey = "$env:USERPROFILE\.ssh\id_ed25519",
    [switch] $Force,
    [switch] $IncludeTinyBackup,
    [switch] $IncludeCheckpoints,
    [switch] $Watch,
    [int] $WatchIntervalSec = 0,
    [switch] $SkipTrainPause
)

$ErrorActionPreference = "Stop"
$BrainRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Weights = Join-Path $BrainRoot "weights"

$pollSec = 0
if ($Watch) {
    $pollSec = if ($WatchIntervalSec -gt 0) { $WatchIntervalSec } else { 120 }
} elseif ($WatchIntervalSec -gt 0) {
    $pollSec = $WatchIntervalSec
}

$sshTarget = "${KaliUser}@${KaliHost}"
$rw = "$RemoteBrain/weights"

$sshOpts = @("-o", "BatchMode=yes", "-o", "ConnectTimeout=30")
if (Test-Path -LiteralPath $SshKey) {
    $sshOpts = @("-i", $SshKey) + $sshOpts
}

function Invoke-RemotePauseTrain {
    if ($SkipTrainPause) { return }
    Write-Host "=== pause remote training before weight pull (remote_pause_lifers_train.sh) ===" -ForegroundColor Cyan
    $pauseSrc = Join-Path $PSScriptRoot "remote_pause_lifers_train.sh"
    if (-not (Test-Path -LiteralPath $pauseSrc)) {
        Write-Warning "missing $pauseSrc — skip remote pause"
        return
    }
    $tmpPause = Join-Path $env:TEMP "lifers_remote_pause_before_weights.sh"
    Copy-Item $pauseSrc $tmpPause -Force
    $enc = New-Object System.Text.UTF8Encoding $false
    $txt = ([IO.File]::ReadAllText($tmpPause) -replace "`r`n", "`n")
    [IO.File]::WriteAllText($tmpPause, $txt, $enc)
    & scp @sshOpts $tmpPause "${sshTarget}:/tmp/lifers_remote_pause_before_weights.sh"
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "scp pause script failed $LASTEXITCODE (continuing)"
        return
    }
    & ssh @sshOpts $sshTarget "bash /tmp/lifers_remote_pause_before_weights.sh" 2>&1 | Write-Host
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "remote pause SSH exited $LASTEXITCODE (continuing pull)"
    }
}

function Invoke-Ssh([string] $Cmd) {
    $out = & ssh @sshOpts $sshTarget $Cmd 2>&1
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
    param([string] $WeightsDir)
    $remote = RemoteStat "lifers_transformer.json"
    if (-not $remote) { Write-Host "[skip] remote lifers_transformer.json missing"; return $false }
    $local = Join-Path $WeightsDir "lifers_transformer.json"
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

function Scp-File([string] $Rel, [string] $WeightsDir) {
    $src = "${sshTarget}:$rw/$Rel".Replace("\", "/")
    $dst = Join-Path $WeightsDir (Split-Path $Rel -Leaf)
    Write-Host "scp -C $Rel ..."
    & scp @sshOpts -C -o ConnectTimeout=60 $src $dst
    if ($LASTEXITCODE -ne 0) { throw "scp failed for $Rel" }
}

function Sync-LifersWeightsOnce {
    param(
        [string] $WeightsDir,
        [string] $RemoteWeightsPath
    )

    New-Item -ItemType Directory -Force -Path $WeightsDir | Out-Null

    Invoke-RemotePauseTrain

    Write-Host "=== Lifers weights sync from Kali ==="
    Write-Host "remote $RemoteWeightsPath"

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
        Scp-File $f $WeightsDir
    }

    if (ShouldPullLargeTransformer -WeightsDir $WeightsDir) {
        Scp-File "lifers_transformer.json" $WeightsDir
    } elseif (-not $Force) {
        Write-Host "(use -Force to re-download lifers_transformer.json anyway)"
    }

    if ($IncludeTinyBackup) {
        $st = RemoteStat "tiny_transformer_v001.json"
        if ($st) { Scp-File "tiny_transformer_v001.json" $WeightsDir }
    }

    if ($IncludeCheckpoints) {
        Write-Host "=== checkpoints (optional bulk) ==="
        Invoke-Ssh "test -d $RemoteWeightsPath/checkpoints && tar czf /tmp/lifers_ckpt.tgz -C $RemoteWeightsPath checkpoints || true" | Out-Null
        $has = Invoke-Ssh "test -f /tmp/lifers_ckpt.tgz && echo yes || echo no"
        if ($has -match "yes") {
            & scp @sshOpts -C "${sshTarget}:/tmp/lifers_ckpt.tgz" $WeightsDir
            Push-Location $WeightsDir
            try { tar -xzf lifers_ckpt.tgz 2>$null } finally { Pop-Location }
            Remove-Item (Join-Path $WeightsDir "lifers_ckpt.tgz") -Force -ErrorAction SilentlyContinue
            Write-Host "checkpoints extracted under weights/checkpoints"
        } else {
            Write-Host "no checkpoints dir on remote"
        }
    }

    Write-Host "Done. Local: $WeightsDir"
}

if ($pollSec -gt 0) {
    Write-Host "Watch mode: polling every ${pollSec}s (Ctrl+C to stop). First sync now..."
    while ($true) {
        try {
            Sync-LifersWeightsOnce -WeightsDir $Weights -RemoteWeightsPath $rw
        } catch {
            Write-Warning "sync failed: $_"
        }
        Start-Sleep -Seconds $pollSec
    }
} else {
    Sync-LifersWeightsOnce -WeightsDir $Weights -RemoteWeightsPath $rw
}
