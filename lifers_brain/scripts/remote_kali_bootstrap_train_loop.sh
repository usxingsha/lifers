#!/usr/bin/env bash
# 在 Kali 上：刷新 Markov 权重 + 在 tmux 里长期跑 train_lifers_escalate 循环（断 SSH 也不断）。
# 假定已通过 tar 或 rsync 更新本仓库到本机（如 ~/lifers/lifers_brain）；从 brain 根或 scripts 下执行均可。
#
# 用法：
#   cd ~/lifers/lifers_brain && bash scripts/remote_kali_bootstrap_train_loop.sh
#   LIFERS_BRAIN_ROOT=/path/to/lifers_brain bash scripts/remote_kali_bootstrap_train_loop.sh
#
# 环境变量（可选）：
#   LIFERS_TRAIN_TMUX_SESSION   默认 lifers-stack
#   LIFERS_TRAIN_LOG            默认 ~/lifers_full_stack.log
#   LIFERS_ESCALATE_UNLIMITED   默认 1
#   LIFERS_RAMP_MAX_ITERS       默认 999999
#   LIFERS_PAUSE_ON_CHECKPOINT  每满 B 档 checkpoint 后 pause，便于同步（默认不设）
#   LIFERS_POST_CHECKPOINT_CMD  每 checkpoint 后执行的命令（可选）
#   LIFERS_BOOTSTRAP_SKIP_MARKOV=1  跳过 train_weights.py（仅重启 escalate 循环时用）
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BR="${LIFERS_BRAIN_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
BR="$(cd "$BR" && pwd)"
cd "$BR"

# Tar from Windows 常带 CRLF；/bin/sh 会报 `pipefail\r` 无效，先统一成 Unix 换行
if command -v sed >/dev/null 2>&1; then
  find "$BR/scripts" -maxdepth 1 -type f -name '*.sh' -exec sed -i 's/\r$//' {} \; 2>/dev/null || true
fi

if [[ ! -f "$BR/scripts/kali_train_escalate_loop.sh" ]]; then
  echo "[lifers-bootstrap] missing brain: $BR" >&2
  exit 1
fi

export PYTHONPATH="$BR${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export LIFERS_BRAIN_ROOT="$BR"
export LIFERS_ESCALATE_UNLIMITED="${LIFERS_ESCALATE_UNLIMITED:-1}"
export LIFERS_RAMP_MAX_ITERS="${LIFERS_RAMP_MAX_ITERS:-999999}"

mkdir -p "$BR/weights"
chmod +x "$BR/scripts"/*.sh 2>/dev/null || true

if ! command -v python3 >/dev/null 2>&1; then
  echo "[lifers-bootstrap] 需要 python3：sudo apt-get update && sudo apt-get install -y python3" >&2
  exit 1
fi

echo "[lifers-bootstrap] BRAIN=$BR"
if [[ "${LIFERS_BOOTSTRAP_SKIP_MARKOV:-}" =~ ^(1|true|yes|on)$ ]]; then
  echo "[lifers-bootstrap] skip Markov (LIFERS_BOOTSTRAP_SKIP_MARKOV)"
else
  echo "[lifers-bootstrap] Markov (train_weights.py) …"
  python3 "$BR/scripts/train_weights.py"
fi

printf 'run\n' >"$BR/weights/.train_control"
echo "[lifers-bootstrap] control=run -> $BR/weights/.train_control"

SESSION="${LIFERS_TRAIN_TMUX_SESSION:-lifers-stack}"
LOG="${LIFERS_TRAIN_LOG:-$HOME/lifers_full_stack.log}"
mkdir -p "$(dirname "$LOG")" 2>/dev/null || true

CACHE="${XDG_CACHE_HOME:-$HOME/.cache}/lifers"
mkdir -p "$CACHE"
RUNNER="$CACHE/run_lifers_escalate_loop.sh"

{
  echo '#!/usr/bin/env bash'
  echo 'set -eu'
  printf 'cd %q\n' "$BR"
  echo 'export PYTHONUNBUFFERED=1'
  printf 'export PYTHONPATH=%q\n' "$BR"
  printf 'export LIFERS_BRAIN_ROOT=%q\n' "$BR"
  printf 'export LIFERS_ESCALATE_UNLIMITED=%q\n' "${LIFERS_ESCALATE_UNLIMITED}"
  printf 'export LIFERS_RAMP_MAX_ITERS=%q\n' "${LIFERS_RAMP_MAX_ITERS}"
  if [[ -n "${LIFERS_PAUSE_ON_CHECKPOINT:-}" ]]; then
    printf 'export LIFERS_PAUSE_ON_CHECKPOINT=%q\n' "${LIFERS_PAUSE_ON_CHECKPOINT}"
  fi
  if [[ -n "${LIFERS_POST_CHECKPOINT_CMD:-}" ]]; then
    echo "export LIFERS_POST_CHECKPOINT_CMD=$(printf %q "$LIFERS_POST_CHECKPOINT_CMD")"
  fi
  if [[ -n "${LIFERS_CHECKPOINT_EVERY_B:-}" ]]; then
    printf 'export LIFERS_CHECKPOINT_EVERY_B=%q\n' "${LIFERS_CHECKPOINT_EVERY_B}"
  fi
  if [[ -n "${LIFERS_MAX_SPEED:-}" ]]; then
    printf 'export LIFERS_MAX_SPEED=%q\n' "${LIFERS_MAX_SPEED}"
  fi
  if [[ -n "${LIFERS_LOOP_IDLE_SEC:-}" ]]; then
    printf 'export LIFERS_LOOP_IDLE_SEC=%q\n' "${LIFERS_LOOP_IDLE_SEC}"
  fi
  echo 'exec bash scripts/kali_train_escalate_loop.sh'
} >"$RUNNER"
chmod +x "$RUNNER"

_ensure_tmux() {
  if command -v tmux >/dev/null 2>&1; then
    return 0
  fi
  if command -v sudo >/dev/null 2>&1; then
    echo "[lifers-bootstrap] 安装 tmux …"
    sudo apt-get update -qq
    sudo apt-get install -y tmux
  fi
  command -v tmux >/dev/null 2>&1
}

if _ensure_tmux; then
  if tmux has-session -t "$SESSION" 2>/dev/null; then
    _alive=0
    if pgrep -f train_lifers_escalate.py >/dev/null 2>&1 || pgrep -f kali_train_escalate_loop.sh >/dev/null 2>&1; then
      _alive=1
    fi
    if [[ "$_alive" -eq 1 ]]; then
      echo "[lifers-bootstrap] tmux 会话已存在且训练进程在跑: $SESSION"
      echo "  接入: tmux attach -t $SESSION"
      echo "  日志: tail -f $LOG"
      echo "  状态: pgrep -af train_lifers_escalate"
      exit 0
    fi
    echo "[lifers-bootstrap] 警告: 会话 $SESSION 存在但未见 train 进程，正在重建 …"
    tmux kill-session -t "$SESSION" 2>/dev/null || true
  fi
  : >"$LOG"
  tmux new-session -d -s "$SESSION" bash -lc "bash $(printf %q "$RUNNER") 2>&1 | tee -a $(printf %q "$LOG")"
  echo "[lifers-bootstrap] 已启动 tmux 会话: $SESSION"
  echo "  接入: tmux attach -t $SESSION"
  echo "  日志: tail -f $LOG"
else
  echo "[lifers-bootstrap] 无 tmux，使用 nohup。"
  nohup bash -lc "exec bash $(printf %q "$RUNNER") 2>&1 | tee -a $(printf %q "$LOG")" >/dev/null &
  echo "[lifers-bootstrap] 后台 PID $!  日志: tail -f $LOG"
fi

echo "[lifers-bootstrap] 训练控制: scripts/lifers_train_ctl.sh run|pause|stop"
echo "[lifers-bootstrap] 能力队列: python3 scripts/lifers_capability_queue.py show"
