@echo off
REM Lifers — 本地 AI 终端助手 (Windows → Kali 透明代理)
ssh -o ControlMaster=no -o StrictHostKeyChecking=no -t kali "lifers %*"
