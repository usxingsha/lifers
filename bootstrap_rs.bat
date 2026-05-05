@echo off
setlocal
REM Auto-deploy rs: workspace + Lifers extension + recommended VSIX-style installs via Cursor CLI
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\bootstrap_lifers.ps1" %*
endlocal
exit /b %ERRORLEVEL%
