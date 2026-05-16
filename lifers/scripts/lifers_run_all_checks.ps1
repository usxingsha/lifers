#Requires -Version 5.1
$Brain = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $Brain
$env:PYTHONPATH = "."
python scripts\lifers_run_all_checks.py
exit $LASTEXITCODE
