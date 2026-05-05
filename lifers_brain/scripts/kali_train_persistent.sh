#!/usr/bin/env bash
# 在 tmux 里跑 kali_train_weights（断 SSH 也不断）；无 tmux 时用 nohup。日志 tee 到文件。
#
# 环境变量：
#   LIFERS_BRAIN_ROOT     brain 根目录（默认：本脚本上级目录）
#   LIFERS_TRAIN_TMUX_SESSION  tmux 会话名，默认 lifers-train
#   LIFERS_TRAIN_LOG      日志路径，默认 ~/lifers/lifers_install.log
#
# 用法：
#   export LIFERS_BRAIN_ROOT=~/lifers/lifers_brain
#   bash scripts/kali_train_persistent.sh
#
# 或安装脚本末尾：--detach-train
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BRAIN="${LIFERS_BRAIN_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
BRAIN="$(cd "$BRAIN" && pwd)"
SESSION="${LIFERS_TRAIN_TMUX_SESSION:-lifers-train}"
LOG="${LIFERS_TRAIN_LOG:-$HOME/lifers/lifers_install.log}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

if [[ ! -f "$BRAIN/scripts/kali_train_weights.sh" ]]; then
  echo "[lifers-persist] 无效的 LIFERS_BRAIN_ROOT: $BRAIN" >&2
  exit 1
fi

_ensure_tmux() {
  if command -v tmux >/dev/null 2>&1; then
    return 0
  fi
  if command -v sudo >/dev/null 2>&1; then
    echo "[lifers-persist] 正在安装 tmux …"
    sudo apt-get update -qq
    sudo apt-get install -y tmux
  fi
  command -v tmux >/dev/null 2>&1
}

CACHE="${XDG_CACHE_HOME:-$HOME/.cache}/lifers"
mkdir -p "$CACHE" "$(dirname "$LOG")" 2>/dev/null || true
RUNNER="$CACHE/run_kali_train_weights.sh"

{
  echo '#!/usr/bin/env bash'
  echo 'set -euo pipefail'
  printf 'cd %q\n' "$BRAIN"
  echo 'export PYTHONUNBUFFERED=1'
  echo "export PYTHONPATH=\"$BRAIN\${PYTHONPATH:+:\$PYTHONPATH}\""
  echo 'exec bash scripts/kali_train_weights.sh'
} >"$RUNNER"
chmod +x "$RUNNER"

if _ensure_tmux; then
  if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "[lifers-persist] tmux 会话 '$SESSION' 已存在。"
    echo "  实时进度: tmux attach -t $SESSION"
    echo "  日志: tail -f $LOG"
    exit 0
  fi
  tmux new-session -d -s "$SESSION" bash -lc "exec bash $(printf %q "$RUNNER") 2>&1 | tee -a $(printf %q "$LOG")"
  echo "[lifers-persist] 已在 tmux 会话 '$SESSION' 中启动训练（含进度条）。"
  echo "  实时: tmux attach -t $SESSION"
  echo "  日志: tail -f $LOG"
else
  echo "[lifers-persist] 无 tmux，使用 nohup + tee。"
  nohup bash -lc "exec bash $(printf %q "$RUNNER") 2>&1 | tee -a $(printf %q "$LOG")" >/dev/null &
  echo "[lifers-persist] 后台已启动。日志: tail -f $LOG"
fi
