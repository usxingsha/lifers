#!/usr/bin/env bash
# Kali / Debian：安装 fcitx5 中文输入法（拼音）及字体；需 sudo。
# 图形会话重启或重新登录后，在托盘配置输入法；详见 playbook §（终端中文）。
set -euo pipefail
if ! sudo -n true 2>/dev/null; then
  echo "需要 sudo。请在 Kali 本机打开终端（可输入密码）执行: bash $0" >&2
  echo "或: sudo bash $0" >&2
  exit 1
fi
export DEBIAN_FRONTEND=noninteractive
sudo apt-get update -qq
sudo apt-get install -y \
  fcitx5 \
  fcitx5-chinese-addons \
  fcitx5-frontend-gtk3 \
  fcitx5-frontend-gtk4 \
  fcitx5-frontend-qt5 \
  dbus-x11 \
  fonts-noto-cjk \
  im-config

PROFILE="$HOME/.profile"
MARK_BEGIN="# BEGIN lifers-fcitx5"
MARK_END="# END lifers-fcitx5"
block="$(cat <<EOF
$MARK_BEGIN
export GTK_IM_MODULE=fcitx
export QT_IM_MODULE=fcitx
export XMODIFIERS=@im=fcitx
export INPUT_METHOD=fcitx
$MARK_END
EOF
)"
if [[ -f "$PROFILE" ]] && grep -q "$MARK_BEGIN" "$PROFILE"; then
  echo "fcitx env block already in $PROFILE"
else
  printf '\n%s\n' "$block" >>"$PROFILE"
  echo "Appended fcitx env to $PROFILE"
fi

echo "OK. Log out and log back in (or reboot). Run: im-config -n fcitx5"
echo "Then: fcitx5-configtool — add Pinyin / Cloud Pinyin if needed."
