#!/usr/bin/env bash
# 在 Kali 上：安装 python3（如需）、可选从 tar 解压、以「尽量完整」预设跑 Markov + Transformer 权重。
#
# 用法（在已解压的 brain 根的上级目录或 brain 内执行均可）：
#   bash scripts/kali_install_full_train.sh
#   bash scripts/kali_install_full_train.sh ~/lifers_kali.tar.gz
#   LIFERS_KALI_TRAIN_MODE=extreme bash scripts/kali_install_full_train.sh
#   bash scripts/kali_install_full_train.sh --enable-boot ~/lifers_kali.tar.gz
#   bash scripts/kali_install_full_train.sh --detach-train
#       → 训练阶段在 tmux（或 nohup）里跑，断 SSH 也不断；日志默认 ~/lifers/lifers_install.log
#
# 环境变量：
#   LIFERS_KALI_DEST      解压父目录，默认 ~/lifers
#   LIFERS_KALI_TRAIN_MODE  传给 kali_train_weights.sh：full（默认本脚本）| extreme | fast
#   LIFERS_BRAIN_ROOT     若已固定 brain 绝对路径，可跳过自动探测
#   LIFERS_KALI_DETACH_TRAIN=1  与 --detach-train 相同（仅训练阶段后台）
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BRAIN_FROM_SCRIPT="$(cd "$SCRIPT_DIR/.." && pwd)"
DEST="${LIFERS_KALI_DEST:-$HOME/lifers}"
ENABLE_BOOT=0
DETACH_TRAIN=0
TAR=""
EXTRA_ARGS=()

for a in "$@"; do
  case "$a" in
    --enable-boot) ENABLE_BOOT=1 ;;
    --detach-train) DETACH_TRAIN=1 ;;
    -h|--help)
      sed -n '1,35p' "$0"
      exit 0
      ;;
    *.tar.gz|*.tgz)
      TAR="$(cd "$(dirname "$a")" && pwd)/$(basename "$a")"
      ;;
    *)
      EXTRA_ARGS+=("$a")
      ;;
  esac
done

if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  echo "未知参数: ${EXTRA_ARGS[*]}" >&2
  exit 1
fi

if [[ -n "$TAR" ]] && [[ ! -f "$TAR" ]]; then
  echo "找不到压缩包: $TAR" >&2
  exit 1
fi

need_sudo() {
  command -v sudo >/dev/null 2>&1
}

install_python3() {
  if command -v python3 >/dev/null 2>&1; then
    return 0
  fi
  echo "[kali-install] 正在安装 python3 …"
  if need_sudo; then
    sudo apt-get update -qq
    sudo apt-get install -y python3
  else
    echo "无 sudo：请手动 apt 安装 python3 后重试。" >&2
    exit 1
  fi
}

resolve_brain() {
  if [[ -n "${LIFERS_BRAIN_ROOT:-}" ]] && [[ -f "$LIFERS_BRAIN_ROOT/scripts/kali_train_weights.sh" ]]; then
    echo "$LIFERS_BRAIN_ROOT"
    return
  fi
  # 从本脚本所在仓库运行（已解压）
  if [[ -f "$BRAIN_FROM_SCRIPT/scripts/kali_train_weights.sh" ]]; then
    echo "$BRAIN_FROM_SCRIPT"
    return
  fi
  mkdir -p "$DEST"
  if [[ -n "$TAR" ]]; then
    echo "[kali-install] 解压 $TAR -> $DEST"
    tar -xzf "$TAR" -C "$DEST"
  fi
  local d
  for d in "$DEST"/*; do
    [[ -d "$d" ]] || continue
    if [[ -f "$d/scripts/kali_train_weights.sh" ]]; then
      echo "$(cd "$d" && pwd)"
      return
    fi
  done
  echo "[kali-install] 无法定位 brain 根。请先解压 tar 到 $DEST，或设置 LIFERS_BRAIN_ROOT，或在已解压目录下执行本脚本。" >&2
  exit 1
}

install_python3
BRAIN="$(resolve_brain)"
BRAIN="$(cd "$BRAIN" && pwd)"
echo "[kali-install] BRAIN=$BRAIN"

if [[ "${LIFERS_KALI_DETACH_TRAIN:-0}" == "1" ]] || [[ "${LIFERS_KALI_DETACH_TRAIN:-}" == "true" ]]; then
  DETACH_TRAIN=1
fi

export LIFERS_KALI_TRAIN_MODE="${LIFERS_KALI_TRAIN_MODE:-full}"
export LIFERS_TARGET_PARAM_B="${LIFERS_TARGET_PARAM_B:-20}"
cd "$BRAIN"
chmod +x scripts/kali_train_weights.sh scripts/kali_train_persistent.sh 2>/dev/null || true
if [[ "${DETACH_TRAIN:-0}" -eq 1 ]]; then
  export LIFERS_BRAIN_ROOT="$BRAIN"
  bash "$SCRIPT_DIR/kali_train_persistent.sh"
else
  bash scripts/kali_train_weights.sh
fi

if [[ "$ENABLE_BOOT" -eq 1 ]]; then
  UNIT_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
  ENV_FILE="${XDG_CONFIG_HOME:-$HOME/.config}/lifers-kali.env"
  mkdir -p "$UNIT_DIR"
  umask 077
  {
    echo "LIFERS_BRAIN_ROOT=$BRAIN"
    echo "LIFERS_KALI_TRAIN_MODE=${LIFERS_KALI_TRAIN_MODE:-full}"
  } >"$ENV_FILE"
  umask 022
  SERVICE_SRC="$SCRIPT_DIR/lifers-kali-train.service"
  if [[ ! -f "$SERVICE_SRC" ]]; then
    echo "[kali-install] 缺少 $SERVICE_SRC，跳过 systemd 安装。" >&2
    exit 0
  fi
  cp "$SERVICE_SRC" "$UNIT_DIR/lifers-kali-train.service"
  chmod 0644 "$UNIT_DIR/lifers-kali-train.service"
  systemctl --user daemon-reload
  systemctl --user enable lifers-kali-train.service
  echo "[kali-install] 已启用用户级开机任务: lifers-kali-train.service"
  echo "  若需未登录也跑: sudo loginctl enable-linger \"$USER\""
  echo "  立即试跑: systemctl --user start lifers-kali-train.service"
  echo "  日志: journalctl --user -u lifers-kali-train.service -e"
fi

echo "[kali-install] 全部完成。权重在: $BRAIN/weights/*.json"
