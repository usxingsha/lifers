#!/usr/bin/env bash
# Lifers 专用壳：独立 user-data-dir + extensions-dir，启动 codium/code/cursor 打开 lifers_brain。
set -euo pipefail
EDITOR_ROOT="$(cd "$(dirname "$0")" && pwd)"
BRAIN_ROOT="$(cd "$EDITOR_ROOT/../.." && pwd)"
EXT_SRC="$BRAIN_ROOT/extensions/lifers-agents-ui"
VER="$(python3 -c "import json;print(json.load(open('$EXT_SRC/package.json'))['version'])")"
BUNDLE="lifers.lifers-agents-ui-${VER}"
EXT_DIR="$EDITOR_ROOT/extensions_dir"
USER_DATA="${XDG_DATA_HOME:-$HOME/.local/share}/lifers-editor/user-data"
mkdir -p "$EXT_DIR" "$USER_DATA"
DST="$EXT_DIR/$BUNDLE"
rm -rf "$EXT_DIR"/lifers.lifers-agents-ui-* 2>/dev/null || true
rm -rf "$DST"
mkdir -p "$DST"
if command -v rsync >/dev/null 2>&1; then
  rsync -a "$EXT_SRC/" "$DST/"
else
  cp -a "$EXT_SRC"/. "$DST/"
fi

REPAIR="$BRAIN_ROOT/scripts/repair_lifers_extensions_index.py"
if [[ -f "$REPAIR" ]] && command -v python3 >/dev/null 2>&1; then
  python3 "$REPAIR" "$EXT_DIR" "$VER"
fi

for c in codium code cursor; do
  if command -v "$c" >/dev/null 2>&1; then
    echo "Using: $(command -v "$c")"
    exec "$c" --user-data-dir "$USER_DATA" --extensions-dir "$EXT_DIR" -n "$BRAIN_ROOT"
  fi
done
echo "未找到 codium / code / cursor。Kali 可尝试: sudo apt install -y vscodium" >&2
exit 1
