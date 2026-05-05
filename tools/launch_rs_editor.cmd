@echo off
REM Portable mode: extensions live under %RS%\data\extensions (installed by install_agents_extension.ps1)
setlocal
set "RS=%~dp0.."
set "VSCODE_PORTABLE=%RS%\data"
set "EXE="
if exist "%RS%\shell\VSCodium\app\VSCodium.exe" set "EXE=%RS%\shell\VSCodium\app\VSCodium.exe"
if not defined EXE if exist "%RS%\shell\VSCodium\VSCodium.exe" set "EXE=%RS%\shell\VSCodium\VSCodium.exe"
if not defined EXE (
  echo No bundled VSCodium under rs\shell\VSCodium. Use your installed editor.
  exit /b 1
)
start "" "%EXE%" %*
