#Requires -Version 5.1
<#
.SYNOPSIS
  删除并重装 Lifers UI 到所有已知 extensions 根目录；Python 修复 extensions.json / .obsolete；
  默认会先 pause Kali 训练（写 weights/.train_control）；加 -SkipTrainPause 则不打断训练。
  含本机 .vscode / .vscode-oss / .cursor、lifers_editor、rs\data；Kali 含 Flatpak 常见路径。
#>
param(
  [string] $KaliHost = "kali@192.168.234.152",
  [switch] $SkipCursor,
  [switch] $SkipKali,
  [switch] $SkipTrainPause,
  [switch] $ResumeKaliTrain
)

$ErrorActionPreference = "Stop"
$BrainRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$ExtSrc = Join-Path $BrainRoot "extensions\lifers-agents-ui"
if (!(Test-Path (Join-Path $ExtSrc "package.json"))) { throw "missing extension: $ExtSrc" }
$pkgPath = Join-Path $ExtSrc "package.json"
$utf8 = New-Object System.Text.UTF8Encoding $false
$pkgRaw = [IO.File]::ReadAllText($pkgPath, $utf8)
if ($pkgRaw -notmatch '"version"\s*:\s*"([^"]+)"') { throw "cannot parse version from $pkgPath" }
$LifersUiVer = $Matches[1]
$bundle = "lifers.lifers-agents-ui-$LifersUiVer"
Write-Host "Lifers UI version: $LifersUiVer ($bundle)"

$ssh = "ssh"
$scp = "scp"
$sshOpts = @("-o", "BatchMode=yes", "-o", "ConnectTimeout=20")

$repairPy = Join-Path $PSScriptRoot "repair_lifers_extensions_index.py"
$pyCmd = @(Get-Command python -ErrorAction SilentlyContinue; Get-Command python3 -ErrorAction SilentlyContinue) | Select-Object -First 1
if (-not $pyCmd) { throw "python or python3 required for repair_lifers_extensions_index.py" }

function Normalize-ShLf([string] $Path) {
  $enc = New-Object System.Text.UTF8Encoding $false
  $t = [IO.File]::ReadAllText($Path)
  $t = $t -replace "`r`n", "`n"
  [IO.File]::WriteAllText($Path, $t, $enc)
}

function Invoke-Ssh([string] $RemoteBash) {
  & $ssh @sshOpts $KaliHost $RemoteBash
  if ($LASTEXITCODE -ne 0) { throw "ssh failed: $LASTEXITCODE" }
}

function Purge-LifersUiDirs([string] $extParent) {
  if (!(Test-Path -LiteralPath $extParent)) { return }
  Get-ChildItem -LiteralPath $extParent -Directory -ErrorAction SilentlyContinue |
    Where-Object { $_.Name -like "lifers.lifers-agents-ui-*" } |
    ForEach-Object {
      Remove-Item -LiteralPath $_.FullName -Recurse -Force
      Write-Host "purged $($_.FullName)"
    }
}

function Install-LifersToExtensionRoot([string] $extParent) {
  New-Item -ItemType Directory -Force -Path $extParent | Out-Null
  Purge-LifersUiDirs $extParent
  $dest = Join-Path $extParent $bundle
  New-Item -ItemType Directory -Force -Path $dest | Out-Null
  & robocopy.exe $ExtSrc $dest /MIR /NFL /NDL /NJH /NJS /nc /ns /np
  if ($LASTEXITCODE -ge 8) { throw "robocopy failed ($LASTEXITCODE) $dest" }
  & $pyCmd.Source $repairPy $extParent $LifersUiVer
  if ($LASTEXITCODE -ne 0) { throw "repair index failed for $extParent" }
}

# ----- Kali -----
if (!$SkipKali) {
  Write-Host "== [1/4] Kali: tar install + repair all extension roots =="
  $pauseSh = Join-Path $PSScriptRoot "remote_pause_lifers_train.sh"
  $installSh = Join-Path $PSScriptRoot "remote_install_lifers_agents_ui_from_stdin.sh"
  Normalize-ShLf $pauseSh
  Normalize-ShLf $installSh
  & $scp @sshOpts $pauseSh "${KaliHost}:/tmp/lifers_remote_pause.sh"
  & $scp @sshOpts $installSh "${KaliHost}:/tmp/lifers_remote_install_ui.sh"
  & $scp @sshOpts $repairPy "${KaliHost}:/tmp/repair_lifers_extensions_index.py"
  if (!$SkipTrainPause) {
    Write-Host "(writing pause on Kali brain paths)"
    Invoke-Ssh "bash /tmp/lifers_remote_pause.sh"
  }

  $tmpTar = Join-Path $env:TEMP "lifers-agents-ui-kali.tgz"
  Push-Location (Join-Path $BrainRoot "extensions")
  & tar.exe -czf $tmpTar "lifers-agents-ui"
  if ($LASTEXITCODE -ne 0) { throw "tar pack failed" }
  Pop-Location

  & $scp @sshOpts $tmpTar "${KaliHost}:/tmp/lifers_agents_ui.tgz"
  Invoke-Ssh "LIFERS_UI_VER=$LifersUiVer bash /tmp/lifers_remote_install_ui.sh /tmp/lifers_agents_ui.tgz && rm -f /tmp/lifers_agents_ui.tgz"
  Remove-Item -LiteralPath $tmpTar -Force -ErrorAction SilentlyContinue

  $kaliRepair = 'export LIFERS_UI_VER=' + $LifersUiVer + '; for d in "$HOME/.vscode/extensions" "$HOME/.vscode-oss/extensions" "$HOME/.var/app/com.vscodium.VSCodium/data/vscode/extensions" "$HOME/.var/app/com.vscodium.VSCodium/data/extensions"; do if [ -d "$d" ] && [ -f /tmp/repair_lifers_extensions_index.py ]; then python3 /tmp/repair_lifers_extensions_index.py "$d" "$LIFERS_UI_VER"; fi; done'
  Invoke-Ssh $kaliRepair
  if ($ResumeKaliTrain) {
    Write-Host "== [1b/4] Kali: resume training (.train_control -> run) =="
    $resumeSh = Join-Path $PSScriptRoot "remote_resume_lifers_train.sh"
    Normalize-ShLf $resumeSh
    & $scp @sshOpts $resumeSh "${KaliHost}:/tmp/remote_resume_lifers_train.sh"
    Invoke-Ssh "bash /tmp/remote_resume_lifers_train.sh"
  }
}

# ----- Windows：本机 + Lifers Editor（不含 rs\data，下一步单独跑便携脚本）-----
Write-Host "== [2/4] Windows: purge + install + repair =="
$rsRoot = Split-Path $BrainRoot -Parent
$editorExtRoot = Join-Path $BrainRoot "tools\lifers_editor\extensions_dir"

$localParents = @(
  (Join-Path $env:USERPROFILE ".vscode\extensions"),
  (Join-Path $env:USERPROFILE ".vscode-oss\extensions"),
  (Join-Path $env:USERPROFILE ".cursor\extensions"),
  $editorExtRoot
)
foreach ($extParent in $localParents) {
  Write-Host "--- $extParent ---"
  Install-LifersToExtensionRoot $extParent
}

Write-Host "== [3/4] rs\data\extensions (portable VSCodium) =="
$portableRepair = Join-Path $rsRoot "scripts\repair_portable_lifers_extension.ps1"
if (Test-Path -LiteralPath $portableRepair) {
  & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $portableRepair -RsRoot $rsRoot
}

Write-Host "== [4/4] Optional: open Cursor =="
if (!$SkipCursor) {
  $cursor = "D:\ai\CURSOR\cursor\Cursor.exe"
  if (Test-Path $cursor) {
    Start-Process $cursor -ArgumentList @("-n", $BrainRoot)
  } else {
    Write-Warning "Cursor.exe not at $cursor"
  }
}

Write-Host "Done. In every VSCodium/Cursor: Developer: Reload Window."

if (!$SkipKali) {
  $openSh = Join-Path $PSScriptRoot "remote_open_lifers_brain_ide.sh"
  if (Test-Path $openSh) {
    Write-Host "== Optional: Kali codium =="
    Normalize-ShLf $openSh
    & $scp @sshOpts $openSh "${KaliHost}:/tmp/lifers_remote_open_ide.sh"
    Invoke-Ssh "bash /tmp/lifers_remote_open_ide.sh"
  }
}
