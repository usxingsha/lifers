@echo off
REM Portable root bootstrap (lifers.code-workspace + Agents UI + 推荐扩展)
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\bootstrap_lifers.ps1" %*
