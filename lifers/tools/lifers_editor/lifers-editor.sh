#!/usr/bin/env bash
# Lifers 专用壳：独立 user-data-dir + extensions-dir；默认仅 VSCodium（codium / 便携 shell），与 tools/vscodium_editor_defaults.json 对齐。
# 允许 Code/Cursor： LIFERS_EDITOR_ALLOW_PROPRIETARY=1 ./lifers-editor.sh
set -euo pipefail
EDITOR_ROOT="$(cd "$(dirname "$0")" && pwd)"
BRAIN_ROOT="$(cd "$EDITOR_ROOT/../.." && pwd)"
PORTABLE_ROOT="$(cd "$BRAIN_ROOT/.." && pwd)"
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

USER_SETTINGS="$USER_DATA/User"
mkdir -p "$USER_SETTINGS"
DEFAULTS="$PORTABLE_ROOT/tools/vscodium_editor_defaults.json"
if [[ ! -f "$USER_SETTINGS/settings.json" && -f "$DEFAULTS" ]]; then
  cp "$DEFAULTS" "$USER_SETTINGS/settings.json"
fi

OPEN="$BRAIN_ROOT"
for ws in "$PORTABLE_ROOT/lifers.code-workspace"; do
  if [[ -f "$ws" ]]; then OPEN="$ws"; break; fi
done

DISABLE=(--disable-telemetry --disable-extension vscode.github-authentication --disable-extension vscode.microsoft-authentication)
if [[ "${LIFERS_ALLOW_SSO:-}" == "1" ]]; then DISABLE=(--disable-telemetry); fi

try_exec() {
  local bin="$1"
  if [[ -f "$bin" ]]; then
    echo "Using: $bin"
    exec "$bin" --user-data-dir "$USER_DATA" --extensions-dir "$EXT_DIR" "${DISABLE[@]}" -n "$OPEN"
  fi
}

for p in \
  "$PORTABLE_ROOT/shell/VSCodium/app/bin/codium" \
  "$PORTABLE_ROOT/shell/VSCodium/bin/codium" \
  "$PORTABLE_ROOT/shell/VSCodium/codium"; do
  try_exec "$p"
done
if command -v codium >/dev/null 2>&1; then
  echo "Using: $(command -v codium)"
  exec codium --user-data-dir "$USER_DATA" --extensions-dir "$EXT_DIR" "${DISABLE[@]}" -n "$OPEN"
fi
if command -v vscodium >/dev/null 2>&1; then
  echo "Using: $(command -v vscodium)"
  exec vscodium --user-data-dir "$USER_DATA" --extensions-dir "$EXT_DIR" "${DISABLE[@]}" -n "$OPEN"
fi

if [[ "${LIFERS_EDITOR_ALLOW_PROPRIETARY:-}" == "1" ]]; then
  for c in code cursor; do
    if command -v "$c" >/dev/null 2>&1; then
      echo "Using: $(command -v "$c")"
      exec "$c" --user-data-dir "$USER_DATA" --extensions-dir "$EXT_DIR" "${DISABLE[@]}" -n "$OPEN"
    fi
  done
fi

echo "未找到 VSCodium（便携 shell 或 PATH 中的 codium）。Kali: sudo apt install -y vscodium ；或设 LIFERS_EDITOR_ALLOW_PROPRIETARY=1 使用 code/cursor。" >&2
exit 1
