#!/usr/bin/env bash
# sync_weights_from_kali.sh — pull LLM training weights from Kali Linux to Windows
# Usage: ./scripts/sync_weights_from_kali.sh [kali_host] [--deep]
#   kali_host defaults to kali@192.168.234.152
#   --deep: also sync lifers_deep_transformer.json (large, default only tiny/markov)
# 提示: 推荐使用 `lifers sync` CLI 命令（统一界面）
set -euo pipefail

KALI="${1:-kali@192.168.234.152}"
KALI_WEIGHTS="/home/kali/lifers/lifers/weights"
WIN_WEIGHTS="$(cd "$(dirname "$0")/.." && pwd)/lifers/weights"
SYNC_DEEP=false
for a in "$@"; do [[ "$a" == "--deep" ]] && SYNC_DEEP=true; done

echo "=== Syncing weights from ${KALI}:${KALI_WEIGHTS} ==="

# Markov (small, always sync fresh copy)
echo "--- Markov ---"
scp "${KALI}:${KALI_WEIGHTS}/lifers_markov.json" "${WIN_WEIGHTS}/lifers_markov_from_kali.json" 2>/dev/null || true
if [ -f "${WIN_WEIGHTS}/lifers_markov_from_kali.json" ]; then
    KALI_MD5=$(ssh "${KALI}" "md5sum ${KALI_WEIGHTS}/lifers_markov.json" | cut -d' ' -f1)
    WIN_MD5=$(md5sum "${WIN_WEIGHTS}/lifers_markov_from_kali.json" 2>/dev/null | cut -d' ' -f1)
    if [ "$KALI_MD5" = "$WIN_MD5" ]; then
        mv "${WIN_WEIGHTS}/lifers_markov_from_kali.json" "${WIN_WEIGHTS}/lifers_markov.json"
        echo "  ✓ lifers_markov.json (unchanged)"
    else
        mv "${WIN_WEIGHTS}/lifers_markov.json" "${WIN_WEIGHTS}/lifers_markov.json.bak.$(date +%Y%m%d)" 2>/dev/null || true
        mv "${WIN_WEIGHTS}/lifers_markov_from_kali.json" "${WIN_WEIGHTS}/lifers_markov.json"
        echo "  ✓ lifers_markov.json UPDATED (${KALI_MD5})"
    fi
fi

# Tiny Transformer (check mtime before syncing)
echo "--- Transformer (tiny) ---"
KALI_MTIME=$(ssh "${KALI}" "stat -c '%Y' ${KALI_WEIGHTS}/lifers_transformer.json 2>/dev/null || echo 0")
WIN_MTIME=$(stat -c '%Y' "${WIN_WEIGHTS}/lifers_transformer.json" 2>/dev/null || echo 0)
KALI_SIZE=$(ssh "${KALI}" "stat -c%s ${KALI_WEIGHTS}/lifers_transformer.json 2>/dev/null || echo 0")
echo "  Kali: ${KALI_SIZE} bytes (mtime=${KALI_MTIME}) | Windows: mtime=${WIN_MTIME}"

if [ "${KALI_MTIME}" -gt "${WIN_MTIME}" ] || [ ! -f "${WIN_WEIGHTS}/lifers_transformer.json" ]; then
    echo "  Syncing ${KALI_SIZE} bytes..."
    scp "${KALI}:${KALI_WEIGHTS}/lifers_transformer.json" "${WIN_WEIGHTS}/lifers_transformer_from_kali.json"
    KS=$(stat -c%s "${WIN_WEIGHTS}/lifers_transformer_from_kali.json")
    if [ "$KS" = "$KALI_SIZE" ]; then
        mv "${WIN_WEIGHTS}/lifers_transformer.json" "${WIN_WEIGHTS}/lifers_transformer.json.bak.$(date +%Y%m%d)" 2>/dev/null || true
        mv "${WIN_WEIGHTS}/lifers_transformer_from_kali.json" "${WIN_WEIGHTS}/lifers_transformer.json"
        echo "  ✓ lifers_transformer.json UPDATED to ${KALI_SIZE} bytes"
    else
        echo "  ✗ size mismatch after scp (got ${KS}), keeping existing"
        rm -f "${WIN_WEIGHTS}/lifers_transformer_from_kali.json"
    fi
else
    echo "  ✓ lifers_transformer.json already up-to-date"
fi

# Deep Transformer (only with --deep, very large)
if $SYNC_DEEP; then
    echo "--- Transformer (deep) ---"
    KALI_MTIME=$(ssh "${KALI}" "stat -c '%Y' ${KALI_WEIGHTS}/lifers_deep_transformer.json 2>/dev/null || echo 0")
    WIN_MTIME=$(stat -c '%Y' "${WIN_WEIGHTS}/lifers_deep_transformer.json" 2>/dev/null || echo 0)
    if [ "${KALI_MTIME}" -gt "${WIN_MTIME}" ] || [ ! -f "${WIN_WEIGHTS}/lifers_deep_transformer.json" ]; then
        KALI_SIZE=$(ssh "${KALI}" "stat -c%s ${KALI_WEIGHTS}/lifers_deep_transformer.json 2>/dev/null || echo 0")
        echo "  Syncing ${KALI_SIZE} bytes..."
        scp "${KALI}:${KALI_WEIGHTS}/lifers_deep_transformer.json" "${WIN_WEIGHTS}/lifers_deep_transformer.json"
        echo "  ✓ lifers_deep_transformer.json UPDATED"
    else
        echo "  ✓ lifers_deep_transformer.json already up-to-date"
    fi
    # Also sync NPZ
    echo "--- Deep NPZ ---"
    KALI_MTIME=$(ssh "${KALI}" "stat -c '%Y' ${KALI_WEIGHTS}/lifers_deep_adam.npz 2>/dev/null || echo 0")
    WIN_MTIME=$(stat -c '%Y' "${WIN_WEIGHTS}/lifers_deep_adam.npz" 2>/dev/null || echo 0)
    if [ "${KALI_MTIME}" -gt "${WIN_MTIME}" ] || [ ! -f "${WIN_WEIGHTS}/lifers_deep_adam.npz" ]; then
        KALI_SIZE=$(ssh "${KALI}" "stat -c%s ${KALI_WEIGHTS}/lifers_deep_adam.npz 2>/dev/null || echo 0")
        echo "  Syncing ${KALI_SIZE} bytes..."
        scp -C "${KALI}:${KALI_WEIGHTS}/lifers_deep_adam.npz" "${WIN_WEIGHTS}/lifers_deep_adam.npz"
        echo "  ✓ lifers_deep_adam.npz UPDATED"
    else
        echo "  ✓ lifers_deep_adam.npz already up-to-date"
    fi
fi

# Also sync train state metadata
echo "--- Training State ---"
scp "${KALI}:${KALI_WEIGHTS}/.lifers_train_state.json" "${WIN_WEIGHTS}/.lifers_train_state.json" 2>/dev/null || true
scp "${KALI}:${KALI_WEIGHTS}/.train_control" "${WIN_WEIGHTS}/.train_control" 2>/dev/null || true
echo "  ✓ train state synced"

echo ""
echo "=== Sync complete ==="
echo "提示: 推荐使用 CLI 命令: lifers sync [--force] [--watch N]"
