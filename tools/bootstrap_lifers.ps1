#Requires -Version 5.1
# One-shot: materialize workspace, integrated_layout bootstrap (prune + conditional materialize),
#           Lifers Agents UI, marketplace recommendations, Cursor->VSCodium settings sync,
#           optional ..\lifers app junction, smoke test.
# Portable root folder may be named `lifers` (historically `rs`); paths below are relative to that root.
param(
    [switch]$SkipMarketplace,
    [switch]$PortableExtensionsOnly,
    [switch]$SkipIntegratedBootstrap,
    [switch]$SkipSyncVscodium,
    [switch]$SkipLinkLifers,
    [switch]$SkipSmoke
)

$ErrorActionPreference = 'Stop'
$toolsDir = $PSScriptRoot
$portableRoot = Split-Path -Parent $toolsDir
Set-Location -LiteralPath $portableRoot

Write-Host '== lifers materialize workspace (lifers.code-workspace) =='
$mat = Join-Path $toolsDir 'materialize_integrated_workspace.py'
if (-not (Test-Path -LiteralPath $mat)) { throw "Missing $mat" }
python $mat
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

if (-not $SkipIntegratedBootstrap) {
    Write-Host '== integrated_layout bootstrap (prune_paths + materialize if layout newer) =='
    $integ = Join-Path $toolsDir 'run_integrated_bootstrap.py'
    if (-not (Test-Path -LiteralPath $integ)) { throw "Missing $integ" }
    python $integ
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

Write-Host '== Lifers Agents UI (copy extension) =='
$agentsParams = @{}
if ($PortableExtensionsOnly) { $agentsParams['EditorTargets'] = 'Vscodium' }
& (Join-Path $toolsDir 'install_agents_extension.ps1') @agentsParams
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host '== Marketplace recommendations (extensions.json) =='
$params = @{}
if ($SkipMarketplace) { $params['SkipMarketplace'] = $true }
if ($PortableExtensionsOnly) { $params['PortableOnly'] = $true }
& (Join-Path $toolsDir 'install_recommended_extensions.ps1') @params

if (-not $SkipSyncVscodium) {
    Write-Host '== Sync Cursor user settings -> data/user-data (VSCodium portable) =='
    $sync = Join-Path $toolsDir 'sync_cursor_settings_to_vscodium.py'
    if (Test-Path -LiteralPath $sync) {
        python $sync
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    }
}

if (-not $SkipLinkLifers) {
    $siblingLifers = Join-Path $portableRoot '..\lifers'
    if (Test-Path -LiteralPath $siblingLifers) {
        $sib = (Resolve-Path -LiteralPath $siblingLifers).Path
        if ($sib -ieq (Resolve-Path -LiteralPath $portableRoot).Path) {
            Write-Host '== Skip link_lifers_app (sibling ..\lifers is the same as portable root) =='
        } else {
            Write-Host '== Optional lifers app junction (portable\lifers -> ..\lifers) =='
            $prevEap = $ErrorActionPreference
            $ErrorActionPreference = 'Continue'
            & (Join-Path $toolsDir 'link_lifers_app.ps1') 2>&1 | ForEach-Object { Write-Host $_ }
            $ErrorActionPreference = $prevEap
        }
    } else {
        Write-Host '== Skip link_lifers_app (no sibling ..\lifers) =='
    }
} else {
    Write-Host '== Skip link_lifers_app (-SkipLinkLifers) =='
}

if (-not $SkipSmoke) {
    Write-Host '== Agents UI smoke test =='
    & (Join-Path $toolsDir 'test_agents_ui_smoke.ps1')
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

Write-Host ''
Write-Host 'Bootstrap OK. Open lifers\lifers.code-workspace. Portable: run_lifers_vscodium.bat -> data\extensions. Reload window if the editor was open.'
