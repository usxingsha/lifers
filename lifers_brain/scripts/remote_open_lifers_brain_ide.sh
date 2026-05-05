#!/usr/bin/env bash
# 在 Kali 桌面会话中尝试后台打开 VS Code / VSCodium（需 DISPLAY 与 GUI）。
set -eu
WS="${LIFERS_KALI_BRAIN_WS:-/home/kali/lifers/lifers_brain}"
export DISPLAY="${DISPLAY:-:0}"
for c in code codium; do
  if command -v "$c" >/dev/null 2>&1; then
    nohup "$c" "$WS" >/tmp/lifers-open-ide.log 2>&1 &
    echo "[lifers-sync] started $c $WS (see /tmp/lifers-open-ide.log)"
    exit 0
  fi
done
echo "[lifers-sync] no code/codium in PATH"
exit 0
