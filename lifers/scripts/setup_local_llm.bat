@echo off
chcp 65001 >/dev/null
echo === Lifers 本地 LLM 配置助手 ===
echo.
echo 检测 Ollama 安装状态...
where ollama >/dev/null 2>&1
if %errorlevel% neq 0 (
    echo [未安装] Ollama 未找到
    echo 请从 https://ollama.com/download/windows 下载安装
    echo 安装后重新运行本脚本
    pause
    exit /b 1
)
echo [已安装] Ollama 已找到
echo.
echo 当前已拉取的模型:
ollama list
echo.
echo 推荐模型（可多选）:
echo   1. qwen2.5:7b        — 通义千问 7B，中文能力强
echo   2. deepseek-r1:7b    — DeepSeek R1 7B，推理能力强
echo   3. llama3.1:8b       — Meta Llama 3.1 8B
echo   4. gemma3:4b         — Google Gemma 3 4B
echo.
set /p MODEL="输入模型名 (默认 qwen2.5:7b): "
if "%MODEL%"=="" set MODEL=qwen2.5:7b
echo.
echo 正在拉取 %MODEL% ...
ollama pull %MODEL%
echo.
echo 完成！请在 VSCode 设置中启用:
echo   "lifers.remoteChat": true
echo   "lifers.chatApiUrl": "http://localhost:11434/v1/chat/completions"
echo   "lifers.chatModel": "%MODEL%"
echo.
pause
