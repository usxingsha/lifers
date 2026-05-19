@echo off
setlocal

REM Launcher for VSCodium/VSCode.
REM Prefers lifers.code-workspace (canonical), falls back to rs.code-workspace (legacy).
REM 若编辑器标题为 NO FOLDER OPENED：请先彻底退出 VSCodium，再用本脚本启动，并确认下方打印的 Workspace 为 lifers.code-workspace。

set "ROOT=%~dp0"

REM Workspace priority: lifers (canonical) > rs (legacy) > folder
if exist "%ROOT%lifers.code-workspace" (
  set "WORKSPACE=%ROOT%lifers.code-workspace"
) else if exist "%ROOT%rs.code-workspace" (
  set "WORKSPACE=%ROOT%rs.code-workspace"
) else if exist "%ROOT%lifers\scripts\agent_bridge.py" (
  set "WORKSPACE=%ROOT%lifers"
) else (
  set "WORKSPACE=%ROOT%"
)

REM 物化 lifers.code-workspace（多根 + 默认 lifers.bridgeTimeoutMs 等）。若文件曾被写成空 folders，会导致 Agents 仍走用户全局 900s 超时。
if exist "%ROOT%tools\materialize_integrated_workspace.py" (
  pushd "%ROOT%"
  python tools\materialize_integrated_workspace.py
  popd
)

set "USER_DATA=%ROOT%data\user-data"
set "EXT_DIR=%ROOT%data\extensions"

REM Offline-first: no GitHub/Microsoft login + no telemetry.
REM Set LIFERS_ALLOW_SSO=1 before launch if you need GitHub/Microsoft sign-in.
set "DISABLE_FLAGS=--disable-telemetry --disable-extension vscode.github-authentication --disable-extension vscode.microsoft-authentication"
if /I "%LIFERS_ALLOW_SSO%"=="1" (
  set "DISABLE_FLAGS=--disable-telemetry"
)

if not exist "%WORKSPACE%" (
  echo Missing workspace: "%WORKSPACE%"
  exit /b 1
)

echo Workspace: %WORKSPACE%
echo.
echo [Lifers] 推理链：clip 见 lifers/fix_inference_prompt + transformer_lm.generate_text；采样见 stack.json brain.local_lm_sampling；请保持本脚本打开的工作区（避免 NO FOLDER / stack 路径漂移）。
echo.

REM Try VSCodium (bundled then user-installed)
for %%E in (
  "%ROOT%shell\VSCodium\VSCodium.exe"
  "%ROOT%shell\VSCodium\app\VSCodium.exe"
  "%LOCALAPPDATA%\Programs\VSCodium\VSCodium.exe"
  "%ProgramFiles%\VSCodium\VSCodium.exe"
) do (
  if exist %%E (
    start "" %%E --user-data-dir "%USER_DATA%" --extensions-dir "%EXT_DIR%" %DISABLE_FLAGS% "%WORKSPACE%"
    exit /b 0
  )
)

REM Try VS Code (bundled then user-installed)
for %%E in (
  "%ROOT%shell\VSCode\Code.exe"
  "%LOCALAPPDATA%\Programs\Microsoft VS Code\Code.exe"
  "%ProgramFiles%\Microsoft VS Code\Code.exe"
) do (
  if exist %%E (
    start "" %%E --user-data-dir "%USER_DATA%" --extensions-dir "%EXT_DIR%" %DISABLE_FLAGS% "%WORKSPACE%"
    exit /b 0
  )
)

echo VSCodium/VSCode not found. Opening in Explorer instead.
start "" explorer "%WORKSPACE%"

endlocal
