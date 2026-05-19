#!/usr/bin/env python3
"""
Deployment configuration — reads from environment variables with safe defaults.
Override via: export LIFERS_KALI_HOST=..., or create a .env file.
"""
import os
from pathlib import Path

# Kali SSH connection
KALI_HOST = os.environ.get("LIFERS_KALI_HOST", "192.168.234.152")
KALI_USER = os.environ.get("LIFERS_KALI_USER", "kali")
KALI_PASS = os.environ.get("LIFERS_KALI_PASS", "")
KALI_PORT = int(os.environ.get("LIFERS_KALI_PORT", "22"))

# Local project root (auto-detected, no hardcoding)
LOCAL_ROOT = Path(__file__).resolve().parent.parent
LIFERS_SRC = LOCAL_ROOT / "lifers"

# Remote Kali paths
KALI_HOME = os.environ.get("LIFERS_KALI_HOME", f"/home/{KALI_USER}/lifers")
KALI_LIFERS = os.environ.get("LIFERS_KALI_LIFERS", f"{KALI_HOME}/lifers")
KALI_WEIGHTS = os.environ.get("LIFERS_KALI_WEIGHTS", f"{KALI_LIFERS}/weights")
KALI_SCRIPTS = os.environ.get("LIFERS_KALI_SCRIPTS", f"{KALI_LIFERS}/scripts")


def get_ssh_client():
    """Create a connected paramiko SSH client"""
    import paramiko
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(KALI_HOST, port=KALI_PORT, username=KALI_USER,
                   password=KALI_PASS, timeout=15)
    return client


def get_sftp_client():
    """Create a connected paramiko SFTP client"""
    ssh = get_ssh_client()
    return ssh.open_sftp(), ssh
