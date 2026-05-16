#!/usr/bin/env bash
# Sync trained weights from Kali back to Windows (or any target).
#
# Usage (on Kali, after training):
#   bash scripts/kali_sync_weights.sh
#   bash scripts/kali_sync_weights.sh user@windows-host:/path/to/lifers/weights/
#   bash scripts/kali_sync_weights.sh /mnt/c/Users/Lifeline/Desktop/curku/lifers/lifers/weights/
#
# Environment:
#   LIFERS_SYNC_TARGET  — scp target or local path (default: auto-detect Windows mount)
#   LIFERS_ROOT   — brain root (default: script parent dir)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${LIFERS_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
ROOT="$(cd "$ROOT" && pwd)"

TARGET="${LIFERS_SYNC_TARGET:-}"
if [[ -z "$TARGET" ]]; then
  # Auto-detect: try common Windows mount points under WSL
  if [[ -d "/mnt/c" ]]; then
    for candidate in \
      "/mnt/c/Users/$USER/Desktop/curku/lifers/lifers/weights" \
      "/mnt/c/Users/Lifeline/Desktop/curku/lifers/lifers/weights" \
    ; do
      if [[ -d "$candidate" ]]; then
        TARGET="$candidate"
        break
      fi
    done
  fi
  if [[ -z "$TARGET" ]]; then
    echo "[sync] No LIFERS_SYNC_TARGET set and no Windows mount found."
    echo "  Usage: LIFERS_SYNC_TARGET=user@host:/path bash $0"
    exit 1
  fi
fi

echo "[sync] ROOT=$ROOT"
echo "[sync] TARGET=$TARGET"

cd "$ROOT"
mkdir -p weights

for f in lifers_transformer.json lifers_markov.json; do
  src="$ROOT/weights/$f"
  if [[ ! -f "$src" ]]; then
    echo "[sync] skip $f (not found)"
    continue
  fi
  size=$(du -h "$src" | cut -f1)
  echo "[sync] $f ($size) -> $TARGET/"
  if [[ "$TARGET" =~ "@" ]] || [[ "$TARGET" =~ ":" ]] && [[ ! "$TARGET" =~ "^/" ]] && [[ ! "$TARGET" =~ "^[A-Z]:" ]]; then
    # scp/rsync target
    scp "$src" "$TARGET/" 2>/dev/null || rsync -av "$src" "$TARGET/" 2>/dev/null || {
      echo "[sync] scp/rsync failed; copying to local fallback: ~/lifers_weights/"
      mkdir -p ~/lifers_weights
      cp "$src" ~/lifers_weights/
    }
  else
    # Local path
    mkdir -p "$TARGET"
    cp "$src" "$TARGET/"
  fi
done

echo "[sync] Done."
ls -la "$ROOT"/weights/lifers_*.json 2>/dev/null || true
