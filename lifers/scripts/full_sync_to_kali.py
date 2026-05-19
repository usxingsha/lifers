#!/usr/bin/env python3
"""全面同步 lifers 到 Kali — 代码、权重、训练脚本，然后安装包并启动训练"""
import paramiko
import os
import sys
import glob

# 统一配置 (通过环境变量覆盖)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config.deploy_config import (KALI_HOST, KALI_USER, KALI_PASS, KALI_PORT,
                                   LOCAL_ROOT, LIFERS_SRC, KALI_LIFERS,
                                   get_ssh_client, get_sftp_client)

LOCAL_ROOT = str(LOCAL_ROOT)
LIFERS_SRC = str(LIFERS_SRC)
KALI_ROOT = os.path.dirname(KALI_LIFERS)  # /home/kali/lifers

def run_cmd(client, cmd, timeout=120):
    stdin, stdout, stderr = client.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode("utf-8", errors="replace").strip()
    err = stderr.read().decode("utf-8", errors="replace").strip()
    if err:
        print(f"  [stderr] {err[:300]}")
    return out

def ensure_remote_dir(client, remote_path):
    """Ensure remote directory exists using exec_command (more reliable than sftp.mkdir)"""
    dirpath = os.path.dirname(remote_path).replace("\\", "/")
    if dirpath and dirpath != "/":
        run_cmd(client, f"mkdir -p {dirpath}")

def sync_file(client, sftp, local_path, remote_path):
    """Sync one file — upload if different size or not exists"""
    if not os.path.exists(local_path):
        return "skip_missing"
    local_size = os.path.getsize(local_path)
    try:
        remote_stat = sftp.stat(remote_path)
        if remote_stat.st_size == local_size:
            return "skip_same"
    except FileNotFoundError:
        pass

    ensure_remote_dir(client, remote_path)
    fname = os.path.basename(local_path)
    print(f"    Uploading {fname} ({local_size/1024:.1f}KB)...", end=" ", flush=True)
    sftp.put(local_path, remote_path)
    print("done")
    return "uploaded"

def main():
    client = get_ssh_client()
    sftp = client.open_sftp()

    try:
        # ========== 1. Sync core lifers package .py files ==========
        print("\n[1/5] Syncing lifers core package...")
        py_files = glob.glob(os.path.join(LIFERS_SRC, "*.py"))
        for local_path in sorted(py_files):
            fname = os.path.basename(local_path)
            remote_path = KALI_LIFERS + "/" + fname
            result = sync_file(client, sftp, local_path, remote_path)
            if result == "uploaded":
                pass

        # ========== 2. Sync scripts ==========
        print("\n[2/5] Syncing scripts...")
        script_files = glob.glob(os.path.join(LIFERS_SRC, "scripts", "*.py"))
        script_files += glob.glob(os.path.join(LIFERS_SRC, "scripts", "*.sh"))
        for local_path in sorted(script_files):
            fname = os.path.basename(local_path)
            remote_path = KALI_LIFERS + "/scripts/" + fname
            sync_file(client, sftp, local_path, remote_path)

        # ========== 3. Sync key weight files ==========
        print("\n[3/5] Syncing weight files...")
        weight_dir = os.path.join(LIFERS_SRC, "weights")
        key_weights = [
            "lifers_deep_transformer.json",
            "lifers_deep_adam.npz",
            "training_corpus.txt",
        ]
        for fname in key_weights:
            local_path = os.path.join(weight_dir, fname)
            remote_path = KALI_LIFERS + "/weights/" + fname
            sync_file(client, sftp, local_path, remote_path)

        # ========== 4. Sync config files ==========
        print("\n[4/5] Syncing config files...")
        config_dir = os.path.join(LOCAL_ROOT, "config")
        for cf in glob.glob(os.path.join(config_dir, "*.py")):
            fname = os.path.basename(cf)
            remote_path = KALI_LIFERS + "/config/" + fname
            sync_file(client, sftp, cf, remote_path)
        for cf in glob.glob(os.path.join(config_dir, "*.json")):
            fname = os.path.basename(cf)
            remote_path = KALI_ROOT + "/config/" + fname
            sync_file(client, sftp, cf, remote_path)

        sftp.close()

        # ========== 5. Install lifers package on Kali ==========
        print("\n[5/5] Installing lifers package on Kali...")

        out = run_cmd(client, f"cd {KALI_LIFERS} && pip3 install -e . --quiet --break-system-packages 2>&1")
        print(f"  pip install: {out[:500]}")

        out = run_cmd(client, "python3 -c 'import lifers; print(\"OK:\", lifers.__file__)' 2>&1")
        print(f"  Import check: {out}")

        # ========== Final status ==========
        print("\n=== Kali 最终状态 ===")
        out = run_cmd(client, f"du -sh {KALI_ROOT}/")
        print(f"  lifers size: {out}")
        out = run_cmd(client, f"find {KALI_LIFERS} -type f | wc -l")
        print(f"  File count: {out}")
        out = run_cmd(client, "free -h | head -2")
        print(f"  Memory:\n{out}")

        print("\n[完成] 同步完成！")

    finally:
        client.close()

if __name__ == "__main__":
    main()
