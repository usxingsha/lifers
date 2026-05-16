#!/usr/bin/env bash
# auto_sync_to_kali.sh — 自动同步变更到 Kali 训练机
# 用法: bash scripts/auto_sync_to_kali.sh
# 后台监控: nohup bash scripts/auto_sync_to_kali.sh --watch > /dev/null 2>&1 &
set -euo pipefail

KALI_HOST="${KALI_HOST:-kali@192.168.234.152}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
KALI_ROOT="${KALI_ROOT:-/home/kali/lifers/lifers}"
SYNC_INTERVAL="${SYNC_INTERVAL:-60}"

# 需要同步的关键文件（相对于项目根目录）
# 注意：绝不包含 logs/ weights/*.npz weights/*.json weights/.train_*
# 日志和训练权重 Kali 独立维护，同步覆盖会导致进程写入幽灵文件
SYNC_FILES=(
    "scripts/cli.py"
    "scripts/lifers_chat.py"
    "scripts/auto_expand_corpus.py"
    "scripts/train_watchdog.py"
    "scripts/edge_inference.py"
    "scripts/train_deep_escalate.py"
    "scripts/lifers"
    "deep_transformer.py"
    "deep_transformer_train.py"
    "__main__.py"
    ".gitignore"
    "Dockerfile"
    "docker-compose.yml"
    "requirements.txt"
    ".env.example"
    "weights/training_corpus.txt"
)

do_sync() {
    local synced=0
    for rel in "${SYNC_FILES[@]}"; do
        local_file="${ROOT}/${rel}"
        remote_dir="$(dirname "${KALI_ROOT}/${rel}")"
        if [[ -f "$local_file" ]]; then
            # 检查文件是否比远程新
            local_mtime=$(stat -c %Y "$local_file" 2>/dev/null || stat -f %m "$local_file" 2>/dev/null || echo 0)
            remote_mtime=$(ssh "$KALI_HOST" "stat -c %Y ${KALI_ROOT}/${rel} 2>/dev/null || echo 0" 2>/dev/null || echo 0)
            if [[ "$local_mtime" -gt "$remote_mtime" ]]; then
                echo "[sync] $(date +%H:%M:%S) ${rel}"
                scp -q "$local_file" "${KALI_HOST}:${remote_dir}/" 2>/dev/null && synced=$((synced + 1))
            fi
        fi
    done
    if [[ $synced -gt 0 ]]; then
        echo "[sync] $(date +%H:%M:%S) 同步了 ${synced} 个文件"
    fi
}

if [[ "${1:-}" == "--watch" ]]; then
    echo "[auto-sync] 启动监控模式，间隔 ${SYNC_INTERVAL}s，目标 ${KALI_HOST}"
    while true; do
        do_sync
        sleep "$SYNC_INTERVAL"
    done
else
    do_sync
fi
