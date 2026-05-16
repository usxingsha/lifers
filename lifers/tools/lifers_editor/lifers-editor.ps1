#Requires -Version 5.1
# Lifers 专用壳：独立 user-data-dir + extensions-dir，默认只找 VSCodium（便携 shell 或系统安装），不依赖 Microsoft VS Code。
# 需要回退到 Code/Cursor 时：  powershell -File lifers-editor.ps1 -AllowProprietaryEditors
param(
  [switch]$AllowProprietaryEditors
)
$ErrorActionPreference = "Stop"
$EditorRoot = $PSScriptRoot
$BrainRoot = (Resolve-Path (Join-Path $EditorRoot "..\..")).Path
$PortableRoot = (Resolve-Path (Join-Path $BrainRoot "..")).Path
$ExtSrc = Join-Path $BrainRoot "extensions\lifers-agents-ui"
$pkgPath = Join-Path $ExtSrc "package.json"
$utf8 = New-Object System.Text.UTF8Encoding $false
$pkgRaw = [IO.File]::ReadAllText($pkgPath, $utf8)
if ($pkgRaw -notmatch '"version"\s*:\s*"([^"]+)"') { throw "cannot parse version from $pkgPath" }
$ver = $Matches[1]
$bundleName = "lifers.lifers-agents-ui-$ver"
$ExtDir = Join-Path $EditorRoot "extensions_dir"
$UserData = Join-Path $env:LOCALAPPDATA "LifersEditor\userdata"
New-Item -ItemType Directory -Force -Path $ExtDir, $UserData | Out-Null

$dest = Join-Path $ExtDir $bundleName
New-Item -ItemType Directory -Force -Path $dest | Out-Null
& robocopy.exe $ExtSrc $dest /MIR /NFL /NDL /NJH /NJS /nc /ns /np
if ($LASTEXITCODE -ge 8) { throw "robocopy bundle failed: $LASTEXITCODE" }

$py = @(Get-Command python -ErrorAction SilentlyContinue; Get-Command python3 -ErrorAction SilentlyContinue) | Select-Object -First 1
$repair = Join-Path $BrainRoot "scripts\repair_lifers_extensions_index.py"
if ($py -and (Test-Path -LiteralPath $repair)) {
  & $py.Source $repair $ExtDir $ver
  if ($LASTEXITCODE -ne 0) { throw "repair extensions.json failed" }
}

$userSettingsDir = Join-Path $UserData "User"
$settingsOut = Join-Path $userSettingsDir "settings.json"
$defaultsPath = Join-Path $PortableRoot "tools\vscodium_editor_defaults.json"
New-Item -ItemType Directory -Force -Path $userSettingsDir | Out-Null
if (-not (Test-Path -LiteralPath $settingsOut) -and (Test-Path -LiteralPath $defaultsPath)) {
  Copy-Item -LiteralPath $defaultsPath -Destination $settingsOut
}

$wsPrimary = Join-Path $PortableRoot "lifers.code-workspace"
$openTarget = $BrainRoot
if (Test-Path -LiteralPath $wsPrimary) { $openTarget = $wsPrimary }

$candidates = @(
  (Join-Path $PortableRoot "shell\VSCodium\app\VSCodium.exe"),
  (Join-Path $PortableRoot "shell\VSCodium\VSCodium.exe"),
  "${env:LocalAppData}\Programs\VSCodium\VSCodium.exe",
  "${env:ProgramFiles}\VSCodium\VSCodium.exe"
) | Where-Object { Test-Path $_ }
if ($AllowProprietaryEditors) {
  $extra = @(
    "${env:LocalAppData}\Programs\Microsoft VS Code\Code.exe",
    "${env:ProgramFiles}\Microsoft VS Code\Code.exe",
    "D:\ai\CURSOR\cursor\Cursor.exe"
  ) | Where-Object { Test-Path $_ }
  $candidates = @($candidates) + @($extra)
}

$candidates = @($candidates)
if ($candidates.Count -lt 1) {
  throw "No VSCodium found under shell\VSCodium or standard install paths. Place portable VSCodium under $PortableRoot\shell\VSCodium\, or install VSCodium. (Use -AllowProprietaryEditors for VS Code/Cursor.)"
}
$exe = $candidates[0]
Write-Host "Using: $exe"
Write-Host "Brain: $BrainRoot"
Write-Host "Open: $openTarget"
Write-Host "Extensions: $ExtDir"

$disable = @(
  "--disable-telemetry",
  "--disable-extension", "vscode.github-authentication",
  "--disable-extension", "vscode.microsoft-authentication"
)
if ($env:LIFERS_ALLOW_SSO -eq "1") {
  $disable = @("--disable-telemetry")
}

Start-Process -FilePath $exe -ArgumentList @(
  "--user-data-dir", $UserData,
  "--extensions-dir", $ExtDir
) + $disable + @("-n", $openTarget)
