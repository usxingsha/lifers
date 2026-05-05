#Requires -Version 5.1
# 通过 SSH 查看 Kali 上 lifers_brain 训练状态。
param(
  [string] $KaliHost = "kali@192.168.234.152",
  [string] $BrainPath = "/home/kali/lifers/lifers_brain"
)
$ErrorActionPreference = "Stop"
$ssh = "ssh"
$scp = "scp"
$opts = @("-o", "BatchMode=yes", "-o", "ConnectTimeout=20")
$sh = Join-Path $PSScriptRoot "remote_kali_train_status.sh"
if (!(Test-Path $sh)) { throw "missing $sh" }
$enc = New-Object System.Text.UTF8Encoding $false
$t = [IO.File]::ReadAllText($sh) -replace "`r`n", "`n"
[IO.File]::WriteAllText($sh, $t, $enc)
& $scp @opts $sh "${KaliHost}:/tmp/remote_kali_train_status.sh"
$bp = $BrainPath -replace "'", "'\''"
& $ssh @opts $KaliHost "bash -lc 'export LIFERS_BRAIN=''$bp''; bash /tmp/remote_kali_train_status.sh'"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
