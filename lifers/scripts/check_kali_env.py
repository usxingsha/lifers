#!/usr/bin/env python3
"""检查 Kali 上的 lifers 环境并启动训练"""
import paramiko
import sys
import os

# 统一配置 (通过环境变量覆盖)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config.deploy_config import KALI_HOST, KALI_USER, KALI_PASS, KALI_PORT, KALI_LIFERS, KALI_WEIGHTS, KALI_SCRIPTS, get_ssh_client

def run_cmd(client, cmd, timeout=60):
    stdin, stdout, stderr = client.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode("utf-8", errors="replace").strip()
    err = stderr.read().decode("utf-8", errors="replace").strip()
    return out, err

client = get_ssh_client()
print("[SSH] Connected")

# 1. Check lifers structure on Kali
print("\n=== Kali lifers 目录结构 ===")
out, _ = run_cmd(client, f"ls -la {KALI_LIFERS}/ 2>/dev/null | head -30")
print(out)
out, _ = run_cmd(client, f"ls -la {KALI_SCRIPTS}/ 2>/dev/null")
print(out)

# 2. Check Python
print("\n=== Python 环境 ===")
out, _ = run_cmd(client, "which python3; python3 --version; python3 -c 'import lifers; print(lifers.__file__)' 2>&1")
print(out)

# 3. Check weights
print("\n=== 权重文件 ===")
out, _ = run_cmd(client, f"ls -lh {KALI_WEIGHTS}/ 2>/dev/null")
print(out)

# 4. Check training script
print("\n=== 训练脚本 ===")
out, _ = run_cmd(client, f"ls -lh {KALI_SCRIPTS}/train_*.py 2>/dev/null")
print(out)

# 5. Check training status
print("\n=== 训练状态 ===")
out, _ = run_cmd(client, f"cat {KALI_WEIGHTS}/.kali_train_status.json 2>/dev/null || echo '无训练状态'")
print(out)

client.close()
