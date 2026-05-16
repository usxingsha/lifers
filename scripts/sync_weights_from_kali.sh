#!/usr/bin/env bash
# sync_weights_from_kali.sh — pull LLM training weights from Kali Linux to Windows
# Usage: ./scripts/sync_weights_from_kali.sh [kali_host]
#   kali_host defaults to root@192.168.234.152
set -euo pipefail

KALI="${1:-root@192.168.234.152}"
KALI_WEIGHTS="/home/kali/lifers/lifers/weights"
WIN_WEIGHTS="$(cd "$(dirname "$0")/.." && pwd)/lifers/weights"

echo "=== Syncing weights from ${KALI}:${KALI_WEIGHTS} ==="

# Markov (small, always sync fresh copy)
echo "--- Markov ---"
scp "${KALI}:${KALI_WEIGHTS}/lifers_markov.json" "${WIN_WEIGHTS}/lifers_markov_from_kali.json"
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

# Transformer (large, check size before syncing)
echo "--- Transformer ---"
KALI_SIZE=$(ssh "${KALI}" "stat -c%s ${KALI_WEIGHTS}/lifers_transformer.json 2>/dev/null || stat -f%z ${KALI_WEIGHTS}/lifers_transformer.json")
WIN_SIZE=$(stat -c%s "${WIN_WEIGHTS}/lifers_transformer.json" 2>/dev/null || echo 0)
echo "  Kali: ${KALI_SIZE} bytes | Windows: ${WIN_SIZE} bytes"

if [ "$KALI_SIZE" != "$WIN_SIZE" ]; then
    echo "  Syncing ${KALI_SIZE} bytes..."
    scp "${KALI}:${KALI_WEIGHTS}/lifers_transformer.json" "${WIN_WEIGHTS}/lifers_transformer_from_kali.json"
    # Verify
    KS=$(stat -c%s "${WIN_WEIGHTS}/lifers_transformer_from_kali.json")
    if [ "$KS" = "$KALI_SIZE" ]; then
        mv "${WIN_WEIGHTS}/lifers_transformer.json" "${WIN_WEIGHTS}/lifers_transformer.json.bak.$(date +%Y%m%d)" 2>/dev/null || true
        mv "${WIN_WEIGHTS}/lifers_transformer_from_kali.json" "${WIN_WEIGHTS}/lifers_transformer.json"
        echo "  ✓ lifers_transformer.json UPDATED to ${KALI_SIZE} bytes"
    else
        echo "  ✗ size mismatch after scp (got ${KS}), keeping existing"
        rm -f "${WIN_WEIGHTS}/lifers_transformer_from_kali.json"
        exit 1
    fi
else
    echo "  ✓ lifers_transformer.json already up-to-date"
fi

# Also sync train state metadata
echo "--- Training State ---"
scp "${KALI}:${KALI_WEIGHTS}/.lifers_train_state.json" "${WIN_WEIGHTS}/.lifers_train_state.json" 2>/dev/null || true
scp "${KALI}:${KALI_WEIGHTS}/.train_control" "${WIN_WEIGHTS}/.train_control" 2>/dev/null || true
echo "  ✓ train state synced"

echo ""
echo "=== Sync complete ==="
