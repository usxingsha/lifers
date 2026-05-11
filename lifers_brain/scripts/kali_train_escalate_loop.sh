#!/usr/bin/env bash
# 有时间就跑：在 tmux/nohup 里挂此脚本；control=run 时跑 train_lifers_escalate，pause/stop 则空转等待。
# 建议与 LIFERS_ESCALATE_UNLIMITED=1、LIFERS_RAMP_MAX_ITERS=999999 同用。
#
# 可选环境变量：
#   LIFERS_BRAIN_ROOT
#   LIFERS_POST_CHECKPOINT_CMD  每满 LIFERS_CHECKPOINT_EVERY_B（默认 1）吉近似参后执行（如 scp/rsync）
#   LIFERS_PAUSE_ON_CHECKPOINT=1  每次跨过新的 B 档 checkpoint 后把 control 写成 pause，便于同步再手动 run
#   LIFERS_TRAIN_SUITE_DIR        训练 jsonl 目录；与 scripts/lifers_capability_queue.py env 配合轮换语料
set -eu
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${LIFERS_BRAIN_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
ROOT="$(cd "$ROOT" && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export LIFERS_ESCALATE_UNLIMITED="${LIFERS_ESCALATE_UNLIMITED:-1}"
# Optional: copy scripts/kali_escalate_env.example.sh -> scripts/kali_escalate_env.sh on low-RAM hosts.
if [[ -f "$ROOT/scripts/kali_escalate_env.sh" ]]; then
  # shellcheck disable=SC1091
  . "$ROOT/scripts/kali_escalate_env.sh"
fi
CTL="${LIFERS_TRAIN_CONTROL:-$ROOT/weights/.train_control}"
mkdir -p "$(dirname "$CTL")"
[[ -f "$CTL" ]] || printf 'run\n' >"$CTL"

_poll() {
  head -1 "$CTL" 2>/dev/null | tr '[:upper:]' '[:lower:]' | awk '{print $1}'
}

echo "[lifers-loop] BRAIN=$ROOT control=$CTL (lifers_train_ctl.sh run|pause|stop)"
# LIFERS_LOOP_IDLE_SEC：run/stop/pause 之间空转间隔；设为 0 或 0.2 拉满循环频率。
_IDLE="${LIFERS_LOOP_IDLE_SEC:-}"
if [[ -z "$_IDLE" ]]; then
  if [[ "${LIFERS_MAX_SPEED:-}" =~ ^(1|true|yes|max|on)$ ]]; then
    _IDLE="0"
  else
    _IDLE="1"
  fi
fi
while true; do
  mode="$(_poll)"
  case "$mode" in
    stop)
      sleep "${LIFERS_LOOP_IDLE_STOP_SEC:-$_IDLE}"
      ;;
    pause)
      sleep "${LIFERS_LOOP_IDLE_PAUSE_SEC:-$_IDLE}"
      ;;
    run)
      python3 scripts/train_lifers_escalate.py || true
      sleep "${LIFERS_LOOP_IDLE_RUN_SEC:-$_IDLE}"
      ;;
    *)
      sleep 2
      ;;
  esac
done
