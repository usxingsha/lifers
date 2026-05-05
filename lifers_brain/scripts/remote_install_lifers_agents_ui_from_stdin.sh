#!/usr/bin/env bash
# 从 stdin 或第一个参数（.tar.gz）安装 lifers-agents-ui；目标版本由环境变量 LIFERS_UI_VER（如 0.5.1）决定。
# 会先删除 ~/.vscode/extensions 与 ~/.vscode-oss/extensions 下全部 lifers.lifers-agents-ui-*。
set -euo pipefail

VER="${LIFERS_UI_VER:-0.5.4}"
TARGET="lifers.lifers-agents-ui-${VER}"
VSC="${HOME}/.vscode/extensions"
OSS="${HOME}/.vscode-oss/extensions"
mkdir -p "$VSC" "$OSS"

shopt -s nullglob
for R in "$VSC" "$OSS"; do
  [[ -d "$R" ]] || continue
  for d in "$R"/lifers.lifers-agents-ui-*; do
    rm -rf "$d"
    echo "[lifers-sync] removed $d"
  done
done

rm -rf "$VSC/lifers-agents-ui-staging" 2>/dev/null || true
mkdir -p "$VSC/lifers-agents-ui-staging"
if [[ -n "${1:-}" ]]; then
  tar -xzf "$1" -C "$VSC/lifers-agents-ui-staging"
else
  tar -xzf - -C "$VSC/lifers-agents-ui-staging"
fi
if [[ ! -d "$VSC/lifers-agents-ui-staging/lifers-agents-ui" ]]; then
  echo "[lifers-sync] tar 缺少顶层 lifers-agents-ui/" >&2
  exit 1
fi
mv "$VSC/lifers-agents-ui-staging/lifers-agents-ui" "$VSC/$TARGET"
rmdir "$VSC/lifers-agents-ui-staging" 2>/dev/null || rm -rf "$VSC/lifers-agents-ui-staging"
cp -a "$VSC/$TARGET" "$OSS/$TARGET"
echo "[lifers-sync] OK $VSC/$TARGET + vscode-oss copy"
