# bootstrap_lifers.ps1
# Auto-deploy: venv, extensions, directories, workspace validation
param(
    [switch]$SkipMarketplace,
    [switch]$SkipVenv,
    [switch]$Verbose
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ROOT   = Split-Path $PSScriptRoot -Parent
$BRAIN  = Join-Path $ROOT "lifers"
$VENV   = Join-Path $BRAIN ".venv"
$DATA   = Join-Path $ROOT "data"
$EXTDIR = Join-Path $DATA "extensions"
$UDDIR  = Join-Path $DATA "user-data"

Write-Host "`n=== Lifers Bootstrap ===" -ForegroundColor Cyan
Write-Host "Root: $ROOT"

# ── 1. Directory scaffold ─────────────────────────────────────────────────────
$dirs = @(
    (Join-Path $BRAIN "config\personas"),
    (Join-Path $BRAIN "core\npc"),
    (Join-Path $BRAIN "scripts"),
    (Join-Path $BRAIN "memory"),
    (Join-Path $BRAIN "weights"),
    (Join-Path $ROOT  "config"),
    (Join-Path $ROOT  "third_party"),
    (Join-Path $ROOT  "shell"),
    $EXTDIR,
    $UDDIR
)
foreach ($d in $dirs) {
    if (-not (Test-Path $d)) {
        New-Item -ItemType Directory -Path $d -Force | Out-Null
        Write-Host "  [+] Created $d" -ForegroundColor Green
    }
}

# ── 2. Python venv ────────────────────────────────────────────────────────────
if (-not $SkipVenv) {
    Write-Host "`n--- Python venv ---" -ForegroundColor Yellow
    $py = Get-Command python -ErrorAction SilentlyContinue
    if (-not $py) {
        Write-Warning "python not found in PATH — skipping venv creation"
    } else {
        if (-not (Test-Path (Join-Path $VENV "Scripts\python.exe"))) {
            Write-Host "  Creating venv at $VENV ..."
            & python -m venv $VENV
        } else {
            Write-Host "  Venv already exists." -ForegroundColor Green
        }
        $pip = Join-Path $VENV "Scripts\pip.exe"
        $req = Join-Path $BRAIN "requirements.txt"
        if (Test-Path $req) {
            Write-Host "  Installing requirements.txt ..."
            & $pip install -r $req --quiet
        } else {
            Write-Host "  No requirements.txt found — installing base deps ..."
            & $pip install requests uvicorn fastapi sqlite-utils --quiet
        }
    }
}

# ── 3. VSCodium / VSCode extension install ────────────────────────────────────
if (-not $SkipMarketplace) {
    Write-Host "`n--- Extension install ---" -ForegroundColor Yellow
    $exts = @(
        "ms-python.python",
        "ms-python.vscode-pylance",
        "streetsidesoftware.code-spell-checker"
    )
    $cliCandidates = @(
        (Join-Path $ROOT "shell\VSCodium\bin\codium.cmd"),
        (Join-Path $ROOT "shell\VSCode\bin\code.cmd"),
        "$env:LOCALAPPDATA\Programs\VSCodium\bin\codium.cmd",
        "$env:LOCALAPPDATA\Programs\Microsoft VS Code\bin\code.cmd"
    )
    $cli = $null
    foreach ($c in $cliCandidates) {
        if (Test-Path $c) { $cli = $c; break }
    }
    if ($cli) {
        foreach ($ext in $exts) {
            Write-Host "  Installing $ext ..."
            & $cli --extensions-dir $EXTDIR --install-extension $ext 2>&1 | Out-Null
        }
        Write-Host "  Extensions done." -ForegroundColor Green
    } else {
        Write-Warning "No VSCodium/VSCode CLI found — skipping extension install"
    }
}

# ── 4. Validate critical files ────────────────────────────────────────────────
Write-Host "`n--- Validating critical files ---" -ForegroundColor Yellow
$critical = @(
    (Join-Path $BRAIN "config\stack.json"),
    (Join-Path $BRAIN "config\tokenizer.json"),
    (Join-Path $BRAIN "scripts\run_agent.py"),
    (Join-Path $BRAIN "scripts\agent_bridge.py")
)
$missing = @()
foreach ($f in $critical) {
    if (Test-Path $f) {
        Write-Host "  [OK] $f" -ForegroundColor Green
    } else {
        Write-Host "  [!!] MISSING: $f" -ForegroundColor Red
        $missing += $f
    }
}

# ── 5. Summary ────────────────────────────────────────────────────────────────
Write-Host "`n=== Bootstrap complete ===" -ForegroundColor Cyan
if ($missing.Count -gt 0) {
    Write-Warning "$($missing.Count) critical file(s) missing — see above"
    exit 1
}
Write-Host "All checks passed. Run run_lifers_agent.bat to start." -ForegroundColor Green
