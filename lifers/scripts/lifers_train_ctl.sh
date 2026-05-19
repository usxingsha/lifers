#!/usr/bin/env bash
# 控制 ramp 训练：run | pause | stop | status
# 默认写 weights/.train_control；可用 LIFERS_TRAIN_CONTROL 覆盖绝对路径。
# 提示: 推荐使用 `lifers control` CLI 命令（统一界面，支持本地+远端+Kali）
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${LIFERS_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
ROOT="$(cd "$ROOT" && pwd)"
CTL="${LIFERS_TRAIN_CONTROL:-$ROOT/weights/.train_control}"
mkdir -p "$(dirname "$CTL")"
case "${1:-}" in
  run) printf 'run\n' >"$CTL" ;;
  pause) printf 'pause\n' >"$CTL" ;;
  stop) printf 'stop\n' >"$CTL" ;;
  status)
    echo "Control file: $CTL"
    [[ -f "$CTL" ]] && cat "$CTL" || echo "(missing $CTL)"
    # 同时检查 deep 训练
    DCTL="${ROOT}/weights/.deep_train_control"
    if [[ -f "$DCTL" ]]; then
      echo "Deep control: $(cat "$DCTL")  ($DCTL)"
    fi
    ;;
  pause_hold17|pause-hold-17)
    printf 'pause\n' >"$CTL"
    echo "[lifers-train-ctl] control=pause -> $CTL"
    echo "[lifers-train-ctl] 若使用 kali_train_escalate_loop：在 ~/.cache/lifers/run_lifers_escalate_loop.sh 或 tmux 启动脚本中"
    echo "  export LIFERS_ESCALATE_MAX_TIER=17 后再 run，可避免进入第 18 档覆盖权重；仅对话推理无需 run。"
    ;;
  *)
    echo "用法: LIFERS_ROOT=... $0 run|pause|stop|status|pause_hold17" >&2
    echo ""
    echo "推荐使用 CLI 统一命令: lifers control [pause|resume|stop|status] [local|kali|all]" >&2
    exit 1
    ;;
esac
