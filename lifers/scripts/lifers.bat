@echo off
setlocal
set "LIFERS_ROOT=%~dp0.."
set "PYTHONPATH=%LIFERS_ROOT%"
"D:\biancheng\python.exe" "%LIFERS_ROOT%\scripts\lifers_chat.py" %*
endlocal
