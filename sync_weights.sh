#!/bin/bash
# 双向权重同步：Windows (主训练) ↔ Kali (备用+安全训练)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KALI_HOST="${LIFERS_KALI_HOST:-192.168.234.152}"
KALI_USER="${LIFERS_KALI_USER:-kali}"
KALI="${KALI_USER}@${KALI_HOST}"

LIFERS_ROOT="${LIFERS_ROOT:-$SCRIPT_DIR}"
WIN_WEIGHTS="${LIFERS_ROOT}/lifers/weights"
KALI_HOME="${LIFERS_KALI_HOME:-/home/${KALI_USER}/lifers}"
KALI_WEIGHTS="${KALI_HOME}/lifers/weights"
KALI_OUTER_WEIGHTS="${KALI_HOME}/weights"

echo "[sync] $(date '+%H:%M:%S') 开始同步..."

# 1. Deep Escalate: Windows → Kali (Windows 是主训练机)
echo "[sync] deep transformer → Kali..."
DEEP_JSON="$WIN_WEIGHTS/lifers_deep_transformer.json"
if [ -f "$DEEP_JSON" ]; then
    scp "$DEEP_JSON" "$KALI:$KALI_WEIGHTS/" 2>&1
    NPZ=$(python3 -c "import json; d=json.load(open('$DEEP_JSON')); print(d.get('_npz',''))" 2>/dev/null)
    if [ -n "$NPZ" ]; then
        echo "[sync] deep NPZ: $NPZ"
        scp "$WIN_WEIGHTS/$NPZ" "$KALI:$KALI_WEIGHTS/" 2>&1
    fi
    if [ -f "$WIN_WEIGHTS/lifers_deep_adam.npz" ]; then
        scp "$WIN_WEIGHTS/lifers_deep_adam.npz" "$KALI:$KALI_WEIGHTS/" 2>&1
    fi
fi

# 2. Perception: Windows → Kali
echo "[sync] perception classifier → Kali..."
scp "$WIN_WEIGHTS/lifers_perception_classifier.json" "$KALI:$KALI_WEIGHTS/" 2>&1

# 3. Safety: Kali → Windows (如果 Kali 在做安全训练)
echo "[sync] safety classifier ← Kali..."
scp "$KALI:$KALI_OUTER_WEIGHTS/lifers_safety_classifier.json" "$WIN_WEIGHTS/" 2>&1

echo "[sync] $(date '+%H:%M:%S') 完成"
