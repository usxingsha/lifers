#Requires -Version 5.1
$Brain = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $Brain
$env:PYTHONPATH = "."
python scripts\run_lifers_gui_host.py @args
