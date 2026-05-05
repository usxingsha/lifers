#!/usr/bin/env bash
# 写入 pause 到常见 brain 路径的 weights/.train_control（含 sudo 探测 /root 下目录）。
set -euo pipefail
for d in /home/kali/lifers/lifers_brain /home/kali/lifers_brain; do
  if [[ -d "$d" ]]; then
    mkdir -p "$d/weights"
    printf 'pause\n' >"$d/weights/.train_control"
    echo "[lifers-sync] pause -> $d/weights/.train_control"
  fi
done
if command -v sudo >/dev/null 2>&1; then
  for d in /root/lifers/lifers_brain /root/lifers_brain; do
    if sudo test -d "$d" 2>/dev/null; then
      sudo mkdir -p "$d/weights"
      printf 'pause\n' | sudo tee "$d/weights/.train_control" >/dev/null
      echo "[lifers-sync] pause (sudo) -> $d/weights/.train_control"
    fi
  done
fi
echo "[lifers-sync] train processes:"
pgrep -af train_lifers_escalate 2>/dev/null || echo "(none)"
