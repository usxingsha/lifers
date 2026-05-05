#!/usr/bin/env bash
# 控制 ramp 训练：run | pause | stop | status
# 默认写 weights/.train_control；可用 LIFERS_TRAIN_CONTROL 覆盖绝对路径。
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${LIFERS_BRAIN_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
ROOT="$(cd "$ROOT" && pwd)"
CTL="${LIFERS_TRAIN_CONTROL:-$ROOT/weights/.train_control}"
mkdir -p "$(dirname "$CTL")"
case "${1:-}" in
  run) printf 'run\n' >"$CTL" ;;
  pause) printf 'pause\n' >"$CTL" ;;
  stop) printf 'stop\n' >"$CTL" ;;
  status) [[ -f "$CTL" ]] && cat "$CTL" || echo "(missing $CTL)" ;;
  *) echo "usage: LIFERS_BRAIN_ROOT=... $0 run|pause|stop|status" >&2; exit 1 ;;
esac
