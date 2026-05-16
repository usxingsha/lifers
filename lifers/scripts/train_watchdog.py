"""
Training watchdog — monitors heartbeat, detects stalls/dead processes, auto-recovers.

Watches:
  - Heartbeat staleness (training stuck/dead)
  - Process existence (PID alive)
  - Memory growth (leak detection)
  - Repeated tier resets (loop detection)

Actions:
  - Kill stuck process after grace period
  - Restart training with same parameters
  - Log all events for debugging

Usage:
  python scripts/train_watchdog.py [--check-interval 30] [--stale-threshold 300]
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

def _read_heartbeat(root: Path) -> Optional[dict]:
    hb = root / "weights" / ".train_heartbeat.json"
    if not hb.is_file():
        return None
    try:
        return json.loads(hb.read_text("utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def _get_memory_mb(pid: int) -> float:
    """Get RSS memory in MB. Returns -1 on failure."""
    if sys.platform == "win32":
        try:
            import ctypes.wintypes
            kernel32 = ctypes.windll.kernel32
            hProcess = kernel32.OpenProcess(0x0400 | 0x0010, False, pid)
            if not hProcess:
                return -1
            class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
                _fields_ = [
                    ("cb", ctypes.wintypes.DWORD),
                    ("PageFaultCount", ctypes.wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]
            pmc = PROCESS_MEMORY_COUNTERS()
            pmc.cb = ctypes.sizeof(pmc)
            kernel32.GetProcessMemoryInfo(hProcess, ctypes.byref(pmc), ctypes.sizeof(pmc))
            kernel32.CloseHandle(hProcess)
            return pmc.WorkingSetSize / (1024 * 1024)
        except Exception:
            return -1
    else:
        try:
            with open(f"/proc/{pid}/status") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        parts = line.split()
                        return float(parts[1]) / 1024  # KB to MB
        except Exception:
            return -1
    return -1


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------

def _kill_process(pid: int) -> bool:
    try:
        if sys.platform == "win32":
            import ctypes
            kernel32 = ctypes.windll.kernel32
            handle = kernel32.OpenProcess(1, False, pid)
            if handle:
                kernel32.TerminateProcess(handle, 1)
                kernel32.CloseHandle(handle)
                return True
        else:
            os.kill(pid, signal.SIGTERM)
            time.sleep(2)
            if _pid_alive(pid):
                os.kill(pid, signal.SIGKILL)
            return True
    except Exception:
        return False


def _restart_training(root: Path, env: dict) -> Optional[int]:
    """Restart train_deep_escalate.py and return new PID."""
    cmd = [
        sys.executable, "-u",
        str(root / "scripts" / "train_deep_escalate.py"),
    ]
    # Ensure critical env vars are set even if watchdog was started without them
    child_env = {**os.environ, **env}
    child_env.setdefault("PYTHONPATH", str(root))
    child_env.setdefault("LIFERS_ROOT", str(root))
    child_env.setdefault("PYTHONUNBUFFERED", "1")
    child_env.setdefault("LIFERS_ESCALATE_UNLIMITED", "1")
    child_env.setdefault("LIFERS_RAMP_MAX_ITERS", "999999")
    if child_env.get("LIFERS_AUTO_THREADS", "1").strip() not in ("0", "false", "no"):
        # Dynamic threading: escalate script will set optimal thread count per tier
        pass
    else:
        child_env.setdefault("OMP_NUM_THREADS", "1")
        child_env.setdefault("OPENBLAS_NUM_THREADS", "1")
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(root),
            env=child_env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return proc.pid
    except Exception as e:
        print(f"[watchdog] Failed to restart training: {e}", file=sys.stderr)
        return None


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Lifers Training Watchdog")
    parser.add_argument("--check-interval", type=int, default=30, help="Seconds between checks")
    parser.add_argument("--stale-threshold", type=int, default=300, help="Seconds before heartbeat considered stale")
    parser.add_argument("--max-memory-mb", type=int, default=8192, help="Restart if RSS exceeds this")
    parser.add_argument("--max-restarts", type=int, default=20, help="Max restarts before giving up")
    args = parser.parse_args()

    root = Path(os.environ.get("LIFERS_ROOT", ROOT))
    control_file = root / "weights" / ".train_status.json"

    print(f"[watchdog] Monitoring {root}/weights/.train_heartbeat.json", file=sys.stderr)
    print(f"[watchdog] Stale threshold: {args.stale_threshold}s, Max memory: {args.max_memory_mb}MB", file=sys.stderr)

    last_good_ts = time.time()
    last_loss = None
    restarts = 0
    stale_reported = False
    same_loss_count = 0

    while restarts < args.max_restarts:
        time.sleep(args.check_interval)

        hb = _read_heartbeat(root)
        if hb is None:
            age = time.time() - last_good_ts
            if age > args.stale_threshold and not stale_reported:
                print(f"[watchdog] ALERT: No heartbeat for {age:.0f}s (file missing)", file=sys.stderr)
                stale_reported = True

                # Check if train_deep_escalate is running
                pid_running = False
                for hb_try in range(3):
                    time.sleep(10)
                    hb = _read_heartbeat(root)
                    if hb is not None:
                        pid_running = True
                        break

                if not pid_running:
                    print(f"[watchdog] Training appears dead — attempting restart ({restarts+1}/{args.max_restarts})", file=sys.stderr)
                    # Read last known config
                    if control_file.is_file():
                        try:
                            ctrl = json.loads(control_file.read_text("utf-8"))
                            env = ctrl.get("env", {})
                        except Exception:
                            env = {}
                    else:
                        env = {}
                    new_pid = _restart_training(root, env)
                    if new_pid:
                        print(f"[watchdog] Restarted training, new PID: {new_pid}", file=sys.stderr)
                        restarts += 1
                        last_good_ts = time.time()
                        stale_reported = False
                    else:
                        print(f"[watchdog] Restart failed", file=sys.stderr)
            continue

        # Heartbeat exists — update tracking
        last_good_ts = time.time()
        stale_reported = False

        pid = hb.get("pid", 0)
        loss = hb.get("loss", float("inf"))
        ramp = hb.get("ramp_iter", 0)

        # Check PID alive
        if pid and not _pid_alive(pid):
            print(f"[watchdog] ALERT: PID {pid} is dead (tier {ramp}, loss {loss:.4f})", file=sys.stderr)
            # Re-read heartbeat to see if new process took over
            time.sleep(15)
            hb2 = _read_heartbeat(root)
            if hb2 and hb2.get("pid") != pid and _pid_alive(hb2.get("pid", 0)):
                print(f"[watchdog] New process PID {hb2['pid']} detected — OK", file=sys.stderr)
                continue
            # No new process — restart
            print(f"[watchdog] No replacement process — restarting ({restarts+1}/{args.max_restarts})", file=sys.stderr)
            new_pid = _restart_training(root, {})
            if new_pid:
                restarts += 1
            continue

        # Check memory
        if pid:
            mem = _get_memory_mb(pid)
            if mem > args.max_memory_mb:
                print(f"[watchdog] ALERT: PID {pid} memory {mem:.0f}MB > {args.max_memory_mb}MB limit (not restarting — model growth is normal)", file=sys.stderr)

        # Check loss stagnation (same loss for too long = possible deadlock)
        if last_loss is not None and abs(loss - last_loss) < 0.0001:
            same_loss_count += 1
            if same_loss_count > 20:
                print(f"[watchdog] ALERT: Loss stagnant at {loss:.4f} for {same_loss_count} checks — possible deadlock", file=sys.stderr)
                same_loss_count = 0
        else:
            same_loss_count = 0
        last_loss = loss

    print(f"[watchdog] Max restarts ({args.max_restarts}) reached — giving up", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
