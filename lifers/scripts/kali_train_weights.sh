#!/usr/bin/env bash
# 在 Kali（或任意 Linux）上生成 Markov + Transformer 权重 JSON。
# 前置：已解压 lifers 包，且当前目录为 lifers 根（含 lifers/ 包目录与 scripts/）。
#
# 训练规模：LIFERS_KALI_TRAIN_MODE=fast|full|extreme（默认 fast，与历史行为一致）。
#   full    — 适合常规 Kali/虚拟机，较长但仍是「小模型」纯 Python 训练。
#   extreme — 步数更多，耗时显著增加；仍可通过 TT_STEPS 等环境变量覆盖。
# 一键安装并全量训练：见 scripts/kali_install_full_train.sh
# 断线仍跑 + 日志：kali_install_full_train.sh --detach-train 或 scripts/kali_train_persistent.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

if ! command -v python3 >/dev/null 2>&1; then
  echo "请安装: sudo apt-get update && sudo apt-get install -y python3" >&2
  exit 1
fi

# Check/install NumPy (required for backprop training)
if ! python3 -c "import numpy" 2>/dev/null; then
  echo "[kali] installing numpy..."
  pip3 install numpy 2>/dev/null || pip install numpy 2>/dev/null || sudo apt-get install -y python3-numpy || true
fi

_apply_train_mode() {
  local mode="${LIFERS_KALI_TRAIN_MODE:-full}"
  case "$mode" in
    extreme)
      export TT_STEPS="${TT_STEPS:-20000}"
      export TT_VOCAB="${TT_VOCAB:-256}"
      export TT_DMODEL="${TT_DMODEL:-128}"
      export TT_DFF="${TT_DFF:-384}"
      export TT_MAXSEQ="${TT_MAXSEQ:-64}"
      export TT_LR="${TT_LR:-3e-4}"
      ;;
    fast)
      export TT_STEPS="${TT_STEPS:-500}"
      export TT_VOCAB="${TT_VOCAB:-128}"
      export TT_DMODEL="${TT_DMODEL:-48}"
      export TT_DFF="${TT_DFF:-128}"
      export TT_MAXSEQ="${TT_MAXSEQ:-48}"
      export TT_LR="${TT_LR:-1e-3}"
      ;;
    full|*)
      export TT_STEPS="${TT_STEPS:-5000}"
      export TT_VOCAB="${TT_VOCAB:-200}"
      export TT_DMODEL="${TT_DMODEL:-96}"
      export TT_DFF="${TT_DFF:-256}"
      export TT_MAXSEQ="${TT_MAXSEQ:-56}"
      export TT_LR="${TT_LR:-3e-4}"
      ;;
  esac
}
_apply_train_mode
echo "[kali] LIFERS_KALI_TRAIN_MODE=${LIFERS_KALI_TRAIN_MODE:-full}  TT_STEPS=$TT_STEPS TT_VOCAB=$TT_VOCAB TT_DMODEL=$TT_DMODEL TT_DFF=$TT_DFF TT_MAXSEQ=$TT_MAXSEQ"

mkdir -p weights
export SANDBOX="${SANDBOX:-1}"

echo "[kali] train Markov -> weights/lifers_markov.json"
MODEL=markov python3 scripts/train_weights.py

echo "[kali] train Transformer -> weights/lifers_transformer.json"
export MODEL=transformer
mode="${LIFERS_KALI_TRAIN_MODE:-full}"
if [[ "${LIFERS_KALI_RAMP:-1}" != "0" ]] && [[ "$mode" != "fast" ]]; then
  export LIFERS_TARGET_PARAM_B="${LIFERS_TARGET_PARAM_B:-20}"
  echo "[kali] ramp train (LIFERS_TARGET_PARAM_B=${LIFERS_TARGET_PARAM_B} 名义 B；遇 OOM 停止). 单次训练: LIFERS_KALI_RAMP=0"
  echo "[kali] long-run: LIFERS_ESCALATE_UNLIMITED=1 + kali_train_escalate_loop.sh; control: lifers_train_ctl.sh"
  python3 scripts/train_lifers_escalate.py
else
  python3 scripts/train_transformer_weights.py
fi

ls -la weights/*.json
echo "[kali] 完成。将 weights/lifers_*.json 拷回 Windows 的 lifers/weights/ 即可。"
