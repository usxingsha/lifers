@echo off
REM Portable root bootstrap (same as bootstrap_rs.bat)
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\bootstrap_lifers.ps1" %*
