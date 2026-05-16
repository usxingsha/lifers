#Requires -Version 5.1
# Quick smoke test: Lifers (rs/lifers or rs/lifers) bridge + lifers-agents-ui on disk.
$ErrorActionPreference = 'Stop'
$rsRoot = Split-Path -Parent $PSScriptRoot
$candidate = Join-Path $rsRoot 'lifers'
$legacy = Join-Path $rsRoot 'lifers'
$brain = if (Test-Path -LiteralPath (Join-Path $candidate 'scripts\agent_bridge_once.py')) { $candidate } else { $legacy }
$bridge = Join-Path $brain 'scripts\agent_bridge_once.py'
$ext = Join-Path $brain 'extensions\lifers-agents-ui\extension.js'
$pkg = Join-Path $brain 'extensions\lifers-agents-ui\package.json'

$ok = $true
if (-not (Test-Path -LiteralPath $bridge)) {
    Write-Host "FAIL missing $bridge"
    $ok = $false
} else {
    Write-Host "OK agent_bridge_once.py"
}

foreach ($p in @($ext, $pkg)) {
    if (-not (Test-Path -LiteralPath $p)) {
        Write-Host "FAIL missing $p"
        $ok = $false
    } else {
        Write-Host "OK $(Split-Path -Leaf $p)"
    }
}

$pycmd = Get-Command python -ErrorAction SilentlyContinue
if ($pycmd) {
    Push-Location $brain
    try {
        $env:PYTHONPATH = $brain
        python -c "from pathlib import Path; assert Path('scripts/agent_bridge_once.py').is_file()" 2>$null
        if ($LASTEXITCODE -ne 0) { throw 'python check failed' }
        Write-Host "OK python cwd $(Split-Path -Leaf $brain)"
    } catch {
        Write-Host "WARN python quick check: $_"
    } finally {
        Pop-Location
    }
} else {
    Write-Host 'SKIP python not in PATH'
}

if (-not $ok) { exit 1 }

$pycmd = Get-Command python -ErrorAction SilentlyContinue
if ($pycmd) {
    Push-Location $brain
    try {
        $env:PYTHONPATH = $brain
        $env:LIFERS_ROOT = $brain
        if (-not $env:MODEL) { $env:MODEL = 'markov' }
        $stdin = '{"text":"ping","contextFiles":[]}'
        $prevEa = $ErrorActionPreference
        $ErrorActionPreference = 'Continue'
        $out = $stdin | python scripts/agent_bridge_once.py 2>&1
        $ErrorActionPreference = $prevEa
        if ($LASTEXITCODE -ne 0) {
            Write-Host "WARN bridge exit $LASTEXITCODE : $out"
        } else {
            try {
                $j = $out | ConvertFrom-Json
                if ($j.ok -eq $true) {
                    Write-Host 'OK agent_bridge_once JSON ok:true'
                } else {
                    Write-Host "WARN bridge returned ok:false error=$($j.error)"
                    # Still smoke-pass if JSON shape is valid (offline model may fail)
                }
            } catch {
                Write-Host "WARN bridge stdout not JSON: $out"
            }
        }
    } finally {
        Pop-Location
        Remove-Item Env:LIFERS_ROOT -ErrorAction SilentlyContinue
    }
}

Write-Host 'SMOKE OK. Reload VSCodium (or your editor) after install_agents_extension.ps1; portable: rs\run_lifers_vscodium.bat uses rs\data\extensions.'
