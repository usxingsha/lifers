#!/bin/bash
# 全支柱训练监控 + 双向权重同步
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KALI_HOST="${LIFERS_KALI_HOST:-192.168.234.152}"
KALI_USER="${LIFERS_KALI_USER:-kali}"
KALI="${KALI_USER}@${KALI_HOST}"

# 动态路径推导，不再硬编码用户名
LIFERS_ROOT="${LIFERS_ROOT:-$SCRIPT_DIR}"
WIN_WEIGHTS="${LIFERS_ROOT}/lifers/weights"
KALI_HOME="${LIFERS_KALI_HOME:-/home/${KALI_USER}/lifers}"
KALI_WEIGHTS="${KALI_HOME}/lifers/weights"
LOG="${LIFERS_ROOT}/training_monitor.log"

while true; do
    echo "" >> "$LOG"
    echo "=== $(date '+%Y-%m-%d %H:%M:%S') ===" >> "$LOG"
    SYNC_DONE=""

    # 1. Windows Deep Escalate 训练状态
    WIN_STATUS_FILE="$WIN_WEIGHTS/.train_status.json"
    if [ -f "$WIN_STATUS_FILE" ]; then
        WIN_STATUS=$(python -c "
import json
s=json.load(open('$WIN_STATUS_FILE'))
print(f\"D={s['architecture']['d_model']} L=? tier={s['ramp']['iter']}/{s['ramp']['max']} step={s['sgd']['step']}/{s['sgd']['total_steps']} ({s['overall_pct_approx']}%)\")
" 2>/dev/null)
        echo "[WIN DEEP] $WIN_STATUS" >> "$LOG"
    fi

    # 2. Kali 训练状态
    KALI_STATUS=$(ssh "$KALI" "cat ${KALI_WEIGHTS}/.kali_train_status.json 2>/dev/null" 2>/dev/null)
    if [ -n "$KALI_STATUS" ]; then
        KALI_PHASE=$(echo "$KALI_STATUS" | python -c "import json,sys; d=json.load(sys.stdin); print(d.get('phase','?'), d.get('current_pillar',''))" 2>/dev/null)
        echo "[KALI] $KALI_PHASE" >> "$LOG"
    fi

    # 3. 检查 Kali 进程
    KALI_PROCS=$(ssh "$KALI" "ps aux | grep python | grep -v grep | grep -v applet | grep -v blueman | wc -l" 2>/dev/null)
    echo "[KALI] 进程数: ${KALI_PROCS:-0}" >> "$LOG"

    # 4. 同步 Windows → Kali (Deep权重 + Perception等)
    DEEP_JSON="$WIN_WEIGHTS/lifers_deep_transformer.json"
    if [ -f "$DEEP_JSON" ]; then
        scp "$DEEP_JSON" "$KALI:$KALI_WEIGHTS/" 2>/dev/null && SYNC_DONE="deep"
        NPZ=$(python -c "import json; d=json.load(open('$DEEP_JSON')); print(d.get('_npz',''))" 2>/dev/null)
        if [ -n "$NPZ" ] && [ -f "$WIN_WEIGHTS/$NPZ" ]; then
            scp "$WIN_WEIGHTS/$NPZ" "$KALI:$KALI_WEIGHTS/" 2>/dev/null
        fi
    fi

    # 同步 Perception, Safety, Proactive, Social 权重到 Kali
    for WF in lifers_perception_classifier.json lifers_safety_classifier.json lifers_proactive_predictor.json lifers_social_classifier.json; do
        if [ -f "$WIN_WEIGHTS/$WF" ]; then
            scp "$WIN_WEIGHTS/$WF" "$KALI:$KALI_WEIGHTS/" 2>/dev/null
        fi
    done

    # 5. 同步 Kali → Windows (Kali训练的权重)
    for WF in lifers_safety_classifier.json lifers_perception_classifier.json lifers_proactive_predictor.json lifers_social_classifier.json; do
        KALI_FILE_TIME=$(ssh "$KALI" "stat --format='%Y' $KALI_WEIGHTS/$WF 2>/dev/null" 2>/dev/null)
        WIN_FILE_TIME=$(stat --format='%Y' "$WIN_WEIGHTS/$WF" 2>/dev/null)
        if [ -n "$KALI_FILE_TIME" ] && [ "$KALI_FILE_TIME" -gt "${WIN_FILE_TIME:-0}" ]; then
            scp "$KALI:$KALI_WEIGHTS/$WF" "$WIN_WEIGHTS/" 2>/dev/null && SYNC_DONE="$SYNC_DONE ${WF##lifers_}"
            echo "[SYNC] Kali→Win: $WF" >> "$LOG"
        fi
    done

    # 6. 控制台输出
    echo "  $(date '+%H:%M:%S') $WIN_STATUS | Kali: ${KALI_PROCS:-0}进程 $SYNC_DONE"

    sleep 300  # 5 分钟
done
