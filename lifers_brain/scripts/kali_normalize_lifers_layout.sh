#!/usr/bin/env bash
# 在 Kali（kali 用户）上执行一次：把历史上落在 $HOME 根下的 Lifers 日志收拢到 $HOME/lifers/，与仓库根布局一致。
set -euo pipefail
ROOT="${LIFERS_ROOT:-$HOME/lifers}"
mkdir -p "$ROOT"
for name in lifers_full_stack.log lifers_install.log; do
  if [[ -f "$HOME/$name" ]] && [[ ! -e "$ROOT/$name" ]]; then
    mv -v "$HOME/$name" "$ROOT/$name"
  elif [[ -f "$HOME/$name" ]] && [[ -f "$ROOT/$name" ]]; then
    echo "[lifers-layout] append $HOME/$name -> $ROOT/$name"
    cat "$HOME/$name" >>"$ROOT/$name"
    rm -v "$HOME/$name"
  fi
done
echo "[lifers-layout] done. ROOT=$ROOT"
