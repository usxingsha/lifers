#!/usr/bin/env bash
# openclaw/openclaw → rs/third_party/openclaw：优先 git 子模块，否则浅克隆（与 Windows 脚本一致）
set -euo pipefail
RS_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
DEST="$RS_ROOT/third_party/openclaw"
URL="https://github.com/openclaw/openclaw.git"
mkdir -p "$(dirname "$DEST")"
command -v git >/dev/null || { echo "need git" >&2; exit 1; }
if [[ -f "$RS_ROOT/.gitmodules" ]] && grep -q "third_party/openclaw" "$RS_ROOT/.gitmodules" 2>/dev/null; then
  (cd "$RS_ROOT" && git submodule update --init --depth 1 third_party/openclaw)
  echo "OK: submodule $DEST"
  exit 0
fi
if [[ -e "$DEST/.git" ]]; then
  echo "Already present: $DEST"
  exit 0
fi
git clone --depth 1 "$URL" "$DEST"
echo "OK: $DEST"
