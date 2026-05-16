#!/bin/bash
# Lifers — 一键安装脚本 (Linux / Kali)
set -e

echo "=== Lifers Install ==="

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$SCRIPT_DIR"

# Install Python package
cd "$ROOT"
pip install --break-system-packages -e . 2>/dev/null || pip install -e . 2>/dev/null || {
    echo "pip install failed, creating symlink instead..."
    ln -sf "$SCRIPT_DIR/lifers/scripts/cli.py" /usr/local/bin/lifers
    chmod +x /usr/local/bin/lifers
}

# Create config directory
mkdir -p ~/.lifers

# Set default config if not exists
if [ ! -f ~/.lifers/config.json ]; then
    cat > ~/.lifers/config.json << 'EOF'
{
  "temperature": 0.7,
  "max_tokens": 80
}
EOF
    echo "  ~/.lifers/config.json created"
fi

# Detect shell and add alias if needed
SHELL_NAME=$(basename "$SHELL")
if [ "$SHELL_NAME" = "bash" ]; then
    PROFILE="$HOME/.bashrc"
elif [ "$SHELL_NAME" = "zsh" ]; then
    PROFILE="$HOME/.zshrc"
else
    PROFILE=""
fi

echo ""
echo "  lifers v$(lifers --version 2>/dev/null | grep -oP 'v[\d.]+' || echo '?')"
echo "  Usage: lifers"
echo "         lifers 'question'"
echo "         lifers stats"
echo ""
echo "  Done."
