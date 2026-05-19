#!/usr/bin/env python3
"""同步 Lifers 代码和语料库到 Kali 服务器"""
import paramiko
import os
import sys

# 统一配置 (通过环境变量覆盖)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config.deploy_config import KALI_HOST, KALI_USER, KALI_PASS, KALI_PORT, KALI_LIFERS, LOCAL_ROOT, get_ssh_client

LIFERS_ROOT = str(LOCAL_ROOT / "lifers")
KALI_LIFERS_ROOT = KALI_LIFERS  # 兼容旧变量名

# 需要同步的文件
FILES_TO_SYNC = [
    (os.path.join(LIFERS_ROOT, "weights", "training_corpus.txt"),
     os.path.join(KALI_LIFERS_ROOT, "weights", "training_corpus.txt")),
    (os.path.join(LIFERS_ROOT, "bridge_turn.py"),
     os.path.join(KALI_LIFERS_ROOT, "bridge_turn.py")),
    (os.path.join(LIFERS_ROOT, "deep_transformer.py"),
     os.path.join(KALI_LIFERS_ROOT, "deep_transformer.py")),
    (os.path.join(LIFERS_ROOT, "local_brain.py"),
     os.path.join(KALI_LIFERS_ROOT, "local_brain.py")),
    (os.path.join(LIFERS_ROOT, "agent.py"),
     os.path.join(KALI_LIFERS_ROOT, "agent.py")),
]

def ssh_connect():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    print(f"[SSH] 连接 {KALI_USER}@{KALI_HOST}:{KALI_PORT} ...")
    client.connect(KALI_HOST, port=KALI_PORT, username=KALI_USER, password=KALI_PASS, timeout=15)
    print("[SSH] 连接成功")
    return client

def run_cmd(client, cmd, timeout=30):
    stdin, stdout, stderr = client.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode("utf-8", errors="replace").strip()
    err = stderr.read().decode("utf-8", errors="replace").strip()
    return out, err

def check_kali(client):
    print("\n=== Kali 系统状态 ===")
    out, _ = run_cmd(client, "uname -a; echo '---'; free -h | head -2; echo '---'; uptime; echo '---'; df -h / | tail -1")
    print(out)

def check_kali_training(client):
    print("\n=== Kali 训练状态 ===")
    out, _ = run_cmd(client, f"ls -lh {KALI_LIFERS_ROOT}/weights/*.json 2>/dev/null; echo '---'; cat {KALI_LIFERS_ROOT}/weights/.kali_train_status.json 2>/dev/null || echo '无monitor状态'")
    print(out)
    out, _ = run_cmd(client, "ps aux | grep -E '(train|python.*lifers)' | grep -v grep || echo '无训练进程'")
    print(out)

def sync_files(client):
    print("\n=== 同步文件 ===")
    sftp = client.open_sftp()
    for local_path, remote_path in FILES_TO_SYNC:
        if not os.path.exists(local_path):
            print(f"  [跳过] 本地文件不存在: {local_path}")
            continue
        local_size = os.path.getsize(local_path)
        try:
            remote_stat = sftp.stat(remote_path)
            remote_size = remote_stat.st_size
            if local_size == remote_size:
                print(f"  [一致] {os.path.basename(local_path)} ({local_size/1024:.1f}KB)")
                continue
        except FileNotFoundError:
            pass

        # 确保远程目录存在
        remote_dir = os.path.dirname(remote_path)
        try:
            sftp.stat(remote_dir)
        except FileNotFoundError:
            run_cmd(client, f"mkdir -p {remote_dir}")

        print(f"  [上传] {os.path.basename(local_path)} ({local_size/1024:.1f}KB) ...")
        sftp.put(local_path, remote_path)
        print(f"  [完成] {os.path.basename(local_path)}")
    sftp.close()

def check_kali_services(client):
    print("\n=== Kali 服务状态 ===")
    out, _ = run_cmd(client, "netstat -tlnp 2>/dev/null | grep -E '(55555|18765)' || echo '无Lifers端口监听'")
    print(out)

def main():
    print("╔══════════════════════════════════════╗")
    print("║   Lifers → Kali 同步工具            ║")
    print("╚══════════════════════════════════════╝")

    try:
        client = ssh_connect()
    except Exception as e:
        print(f"[错误] SSH连接失败: {e}")
        sys.exit(1)

    try:
        check_kali(client)
        check_kali_training(client)
        check_kali_services(client)
        sync_files(client)

        print("\n=== 同步完成 ===")
        print(f"提示: 如需监控训练:")
        print(f"  ssh {KALI_USER}@{KALI_HOST}")
        print(f"  cat {KALI_LIFERS_ROOT}/weights/.kali_train_status.json")
    finally:
        client.close()

if __name__ == "__main__":
    main()
