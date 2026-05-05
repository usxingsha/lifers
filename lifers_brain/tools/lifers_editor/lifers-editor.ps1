#Requires -Version 5.1
# Lifers editor shell: isolated user-data-dir + extensions-dir; launches VSCodium / VS Code / Cursor.
$ErrorActionPreference = "Stop"
$EditorRoot = $PSScriptRoot
$BrainRoot = (Resolve-Path (Join-Path $EditorRoot "..\..")).Path
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

$candidates = @(
  @(
    "${env:LocalAppData}\Programs\VSCodium\VSCodium.exe",
    "${env:ProgramFiles}\VSCodium\VSCodium.exe",
    "D:\ai\CURSOR\cursor\Cursor.exe",
    "${env:LocalAppData}\Programs\Microsoft VS Code\Code.exe"
  ) | Where-Object { Test-Path $_ }
)

if ($candidates.Count -lt 1) {
  throw "No VSCodium, VS Code, or Cursor found. Install one, then re-run."
}
$exe = $candidates[0]
Write-Host "Using: $exe"
Write-Host "Brain: $BrainRoot"
Write-Host "Extensions: $ExtDir"
Start-Process -FilePath $exe -ArgumentList @(
  "--user-data-dir", $UserData,
  "--extensions-dir", $ExtDir,
  "-n",
  $BrainRoot
)
