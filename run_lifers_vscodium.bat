@echo off
setlocal

REM Launcher for VSCodium/VSCode shell replacement.
REM Bundled shell under portable root: shell\VSCodium\ / shell\VSCode\
REM Then user-installed locations.

set "ROOT=%~dp0"
REM 1) Multi-root workspace  2) 便携根文件夹（含 lifers_brain） 3) 仅 brain
if exist "%ROOT%lifers.code-workspace" (
  set "WORKSPACE=%ROOT%lifers.code-workspace"
) else if exist "%ROOT%lifers_brain\scripts\agent_bridge_once.py" (
  set "WORKSPACE=%ROOT%"
) else (
  set "WORKSPACE=%ROOT%lifers_brain"
)
set "USER_DATA=%ROOT%data\user-data"
set "EXT_DIR=%ROOT%data\extensions"

REM Offline-first: no GitHub/Microsoft login extensions + no telemetry.
REM Set LIFERS_ALLOW_SSO=1 before launch if you need GitHub/Microsoft sign-in.
set "DISABLE_FLAGS=--disable-telemetry --disable-extension vscode.github-authentication --disable-extension vscode.microsoft-authentication"
if /I "%LIFERS_ALLOW_SSO%"=="1" (
  set "DISABLE_FLAGS=--disable-telemetry"
)

if not exist "%WORKSPACE%" (
  echo Missing workspace folder: "%WORKSPACE%"
  exit /b 1
)

REM Try VSCodium (bundled or user install)
set "CODIUM0=%ROOT%shell\VSCodium\VSCodium.exe"
set "CODIUM0A=%ROOT%shell\VSCodium\app\VSCodium.exe"
set "CODIUM1=%LOCALAPPDATA%\Programs\VSCodium\VSCodium.exe"
set "CODIUM2=%ProgramFiles%\VSCodium\VSCodium.exe"
if exist "%CODIUM0%" (
  start "" "%CODIUM0%" --user-data-dir "%USER_DATA%" --extensions-dir "%EXT_DIR%" %DISABLE_FLAGS% ^
    "%WORKSPACE%"
  exit /b 0
)
if exist "%CODIUM0A%" (
  start "" "%CODIUM0A%" --user-data-dir "%USER_DATA%" --extensions-dir "%EXT_DIR%" %DISABLE_FLAGS% ^
    "%WORKSPACE%"
  exit /b 0
)
if exist "%CODIUM1%" (
  start "" "%CODIUM1%" --user-data-dir "%USER_DATA%" --extensions-dir "%EXT_DIR%" %DISABLE_FLAGS% ^
    "%WORKSPACE%"
  exit /b 0
)
if exist "%CODIUM2%" (
  start "" "%CODIUM2%" --user-data-dir "%USER_DATA%" --extensions-dir "%EXT_DIR%" %DISABLE_FLAGS% ^
    "%WORKSPACE%"
  exit /b 0
)

REM Try VS Code
set "CODE0=%ROOT%shell\VSCode\Code.exe"
set "CODE1=%LOCALAPPDATA%\Programs\Microsoft VS Code\Code.exe"
set "CODE2=%ProgramFiles%\Microsoft VS Code\Code.exe"
if exist "%CODE0%" (
  start "" "%CODE0%" --user-data-dir "%USER_DATA%" --extensions-dir "%EXT_DIR%" %DISABLE_FLAGS% ^
    "%WORKSPACE%"
  exit /b 0
)
if exist "%CODE1%" (
  start "" "%CODE1%" --user-data-dir "%USER_DATA%" --extensions-dir "%EXT_DIR%" %DISABLE_FLAGS% ^
    "%WORKSPACE%"
  exit /b 0
)
if exist "%CODE2%" (
  start "" "%CODE2%" --user-data-dir "%USER_DATA%" --extensions-dir "%EXT_DIR%" %DISABLE_FLAGS% ^
    "%WORKSPACE%"
  exit /b 0
)

echo VSCodium/VSCode not found. Opening folder in Explorer instead.
start "" explorer "%WORKSPACE%"

endlocal

