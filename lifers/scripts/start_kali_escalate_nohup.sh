#!/usr/bin/env bash
# 可选：nohup 前台跑 ramp 循环（与 tmux runner 二选一）。
# 不在此脚本默认 export LIFERS_ESCALATE_MAX_TIER，避免在内存充足的机器上误锁在 16 档。
# 若确需封顶，由调用方显式: export LIFERS_ESCALATE_MAX_TIER=16
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export LIFERS_ROOT="$ROOT"
export LIFERS_ESCALATE_UNLIMITED="${LIFERS_ESCALATE_UNLIMITED:-1}"
export LIFERS_RAMP_MAX_ITERS="${LIFERS_RAMP_MAX_ITERS:-999999}"
exec bash scripts/kali_train_escalate_loop.sh
