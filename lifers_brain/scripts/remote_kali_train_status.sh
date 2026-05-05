#!/usr/bin/env bash
# 由 kali_train_status.ps1 scp 后执行：BRAIN 环境变量指向 lifers_brain 根目录。
set -euo pipefail
BR="${LIFERS_BRAIN:-/home/kali/lifers/lifers_brain}"
echo "=== LIFERS_BRAIN=$BR ==="
echo "=== .train_control ==="
cat "$BR/weights/.train_control" 2>/dev/null || echo "(none)"
echo "=== processes ==="
pgrep -af train_lifers_escalate 2>/dev/null || echo "(none)"
echo "=== weights (sizes) ==="
ls -la "$BR/weights"/*.json 2>/dev/null | tail -12 || true
echo "=== live train status (weights/.train_status.json) ==="
if [[ -f "$BR/weights/.train_status.json" ]]; then
  python3 "$BR/scripts/show_train_progress.py" "$BR" 2>/dev/null || cat "$BR/weights/.train_status.json"
else
  echo "(none — run train_lifers_escalate with lifers_brain that has scripts/show_train_progress.py)"
fi
echo "=== install log tail ==="
tail -n 22 /home/kali/lifers_install.log 2>/dev/null || tail -n 22 "$(dirname "$BR")/lifers_install.log" 2>/dev/null || echo "(no log)"
