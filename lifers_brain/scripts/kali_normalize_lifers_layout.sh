#!/usr/bin/env bash
# 在 Kali（kali 用户）执行：把历史上落在 $HOME 根下的 Lifers 相关物收拢到 $HOME/lifers/（与仓库根一致）。
# claw-code：若在 ~/claw-code（常为 root 拥有），用 rsync 合并进 ~/lifers/third_party/claw-code（排除 rust/target、.git 以省体积与权限问题）；原目录需 root 时由你本机执行文内 sudo 两行删掉。
set -euo pipefail
ROOT="${LIFERS_ROOT:-$HOME/lifers}"
mkdir -p "$ROOT/third_party"

# 日志：~/lifers_full_stack.log 等 → ~/lifers/
for name in lifers_full_stack.log lifers_install.log; do
  if [[ -f "$HOME/$name" ]]; then
    if [[ ! -f "$ROOT/$name" ]]; then
      mv -v "$HOME/$name" "$ROOT/$name"
    else
      echo "[lifers-layout] append $HOME/$name -> $ROOT/$name"
      cat "$HOME/$name" >>"$ROOT/$name"
      rm -v "$HOME/$name"
    fi
  fi
done

CLAW_SRC="$HOME/claw-code"
CLAW_DST="$ROOT/third_party/claw-code"
if [[ -d "$CLAW_SRC" ]]; then
  mkdir -p "$CLAW_DST"
  echo "[lifers-layout] rsync $CLAW_SRC -> $CLAW_DST (exclude rust/target .git)"
  rsync -a --delete --exclude=rust/target --exclude=.git "$CLAW_SRC/" "$CLAW_DST/" || true
  owner="$(stat -c %U "$CLAW_SRC" 2>/dev/null || echo unknown)"
  if [[ "$owner" == "root" ]]; then
    echo "[lifers-layout] 源目录为 root 所有，无法用普通权限删除。若确认已合并，可在本机执行："
    echo "  sudo rm -rf $CLAW_SRC"
    echo "  sudo chown -R \"$(id -un):$(id -gn)\" $CLAW_DST"
  else
    rm -rf "$CLAW_SRC"
  fi
fi

echo "[lifers-layout] done. ROOT=$ROOT"
if [[ -f "$CLAW_DST/rust/Cargo.toml" ]]; then
  echo "[lifers-layout] claw rust at: $CLAW_DST/rust"
fi
