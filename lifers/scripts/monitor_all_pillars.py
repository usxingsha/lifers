"""
全支柱训练实时监控系统 — 监控 Windows + Kali 双边训练状态
输出: weights/.monitor_status.json (前端轮询) + terminal日志
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, Optional

ROOT = Path(__file__).resolve().parent.parent
WEIGHTS = ROOT / "weights"
MONITOR_FILE = WEIGHTS / ".monitor_status.json"

# 从统一配置读取 Kali 连接信息（支持环境变量覆盖）
import importlib.util
_config_path = ROOT.parent / "config" / "deploy_config.py"
if _config_path.exists():
    spec = importlib.util.spec_from_file_location("deploy_config", _config_path)
    _cfg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(_cfg)
    KALI_HOST = f"{_cfg.KALI_USER}@{_cfg.KALI_HOST}"
    KALI_WEIGHTS = _cfg.KALI_WEIGHTS
else:
    KALI_HOST = os.environ.get("LIFERS_KALI_HOST", "kali@192.168.234.152")
    KALI_WEIGHTS = os.environ.get("LIFERS_KALI_WEIGHTS", "/home/kali/lifers/lifers/weights")

PILLAR_WEIGHT_FILES = {
    "deep_escalate": "lifers_deep_transformer.json",
    "perception": "lifers_perception_classifier.json",
    "safety": "lifers_safety_classifier.json",
    "proactive": "lifers_proactive_predictor.json",
    "social": "lifers_social_classifier.json",
    "kg": "lifers_kg_embeddings.json",
    "rl": "lifers_rl_policy.json",
    "voice": "lifers_voice_acoustic.json",
    "robot_hal": "lifers_robot_hal_policy.json",
    "simulation": "lifers_simulation_evaluator.json",
    "swarm": "lifers_swarm_policy.json",
    "telemetry": "lifers_telemetry_detector.json",
    "dashboard": "lifers_dashboard_config.json",
}

DATA_FILES = {
    "perception": "perception_samples.jsonl",
    "proactive": "proactive_samples.jsonl",
    "safety_safe": "safety_safe.jsonl",
    "safety_unsafe": "safety_unsafe.jsonl",
    "social": "social_samples.jsonl",
    "corpus": "training_corpus.txt",
}


def read_train_status() -> Optional[Dict]:
    path = WEIGHTS / ".train_status.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def check_weight_file(name: str) -> Dict:
    wf = WEIGHTS / name
    if not wf.is_file():
        return {"status": "missing"}
    stat = wf.stat()
    age_s = time.time() - stat.st_mtime
    return {
        "status": "present",
        "size_kb": round(stat.st_size / 1024, 1),
        "age_min": round(age_s / 60, 1),
        "updated": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
    }


def check_data_file(name: str, data_dir: Path) -> Dict:
    path = data_dir / name
    if not path.is_file():
        return {"status": "missing", "lines": 0, "size_mb": 0}
    size_mb = path.stat().st_size / (1024 * 1024)
    try:
        with open(path, encoding="utf-8") as f:
            lines = sum(1 for _ in f)
    except Exception:
        lines = -1
    return {"status": "present", "lines": lines, "size_mb": round(size_mb, 1)}


def check_kali_pillar(pillar: str) -> Optional[Dict]:
    wf_name = PILLAR_WEIGHT_FILES.get(pillar)
    if not wf_name:
        return None
    try:
        result = subprocess.run(
            ["ssh", KALI_HOST, f"stat --format='%s %Y' {KALI_WEIGHTS}/{wf_name} 2>/dev/null || echo missing"],
            capture_output=True, text=True, timeout=10
        )
        out = result.stdout.strip()
        if out == "missing" or not out:
            return {"status": "missing"}
        parts = out.split()
        size = int(parts[0])
        mtime = int(parts[1])
        age_s = time.time() - mtime
        return {
            "status": "present",
            "size_kb": round(size / 1024, 1),
            "age_min": round(age_s / 60, 1),
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


def check_kali_process() -> Dict:
    try:
        result = subprocess.run(
            ["ssh", KALI_HOST, "ps aux | grep python | grep -v grep | grep -v applet | grep -v blueman"],
            capture_output=True, text=True, timeout=10
        )
        lines = [l for l in result.stdout.strip().split("\n") if l.strip()]
        processes = []
        for l in lines:
            parts = l.split()
            if len(parts) >= 11:
                processes.append({"pid": parts[1], "cpu": parts[2], "mem": parts[3], "time": parts[9], "cmd": " ".join(parts[10:])[:80]})
        return {"count": len(processes), "processes": processes}
    except Exception as e:
        return {"count": 0, "error": str(e)}


def check_windows_process() -> Dict:
    try:
        import psutil
        processes = []
        for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_info', 'cmdline']):
            try:
                info = proc.info
                if info['name'] and 'python' in info['name'].lower():
                    cmd = ' '.join(info.get('cmdline', []) or [])[:100]
                    mem_mb = info['memory_info'].rss / (1024 * 1024) if info.get('memory_info') else 0
                    processes.append({
                        "pid": info['pid'],
                        "cpu": info.get('cpu_percent', 0),
                        "mem_mb": round(mem_mb, 0),
                        "cmd": cmd,
                    })
            except Exception:
                pass
        return {"count": len(processes), "processes": processes}
    except ImportError:
        return {"count": -1, "error": "psutil not available"}
    except Exception as e:
        return {"count": -1, "error": str(e)}


def collect_status() -> Dict:
    data_dir = ROOT.parent / "data"

    status = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "local_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "deep_escalate": {},
        "pillars": {},
        "data": {},
        "kali": {"pillars": {}, "processes": {}},
        "windows": {"processes": {}},
    }

    # Deep escalate training status
    train_status = read_train_status()
    if train_status:
        status["deep_escalate"] = {
            "active": True,
            "pid": train_status.get("pid"),
            "phase": train_status.get("phase"),
            "d_model": train_status.get("architecture", {}).get("d_model"),
            "vocab": train_status.get("architecture", {}).get("max_vocab"),
            "ramp_iter": train_status.get("ramp", {}).get("iter"),
            "ramp_max": train_status.get("ramp", {}).get("max"),
            "sgd_step": train_status.get("sgd", {}).get("step"),
            "sgd_total": train_status.get("sgd", {}).get("total_steps"),
            "overall_pct": train_status.get("overall_pct_approx"),
            "message": train_status.get("message"),
        }

    # All pillar weights
    for pillar, wf_name in PILLAR_WEIGHT_FILES.items():
        status["pillars"][pillar] = check_weight_file(wf_name)

    # Data files
    for name, fname in DATA_FILES.items():
        status["data"][name] = check_data_file(fname, data_dir)

    # Check corpus in weights too
    corpus_w = WEIGHTS / "training_corpus.txt"
    if corpus_w.is_file():
        status["data"]["corpus"]["size_mb"] = round(corpus_w.stat().st_size / (1024 * 1024), 1)

    # Kali status
    for pillar in ["deep_escalate", "perception", "safety", "proactive", "social"]:
        result = check_kali_pillar(pillar)
        if result:
            status["kali"]["pillars"][pillar] = result
    status["kali"]["processes"] = check_kali_process()

    # Windows processes
    status["windows"]["processes"] = check_windows_process()

    return status


def print_status(status: Dict):
    s = status
    print(f"\n{'='*70}")
    print(f"  Lifers 全支柱监控  {s['local_time']}")
    print(f"{'='*70}")

    # Deep Escalate
    de = s.get("deep_escalate", {})
    if de.get("active"):
        print(f"\n  [Deep Escalate] D={de.get('d_model')} V={de.get('vocab')} "
              f"Tier {de.get('ramp_iter')}/{de.get('ramp_max')} "
              f"SGD {de.get('sgd_step')}/{de.get('sgd_total')} "
              f"({de.get('overall_pct', 0):.1f}%) PID={de.get('pid')}")
    else:
        print(f"\n  [Deep Escalate] 未运行")

    # Pillars
    print(f"\n  {'支柱':<18} {'状态':<10} {'大小':>8} {'更新时间':>12}")
    print(f"  {'-'*50}")
    for pillar, info in s.get("pillars", {}).items():
        status_str = info.get("status", "?")
        size = f"{info.get('size_kb', 0):.1f}KB"
        age = f"{info.get('age_min', 0):.0f}min前"
        print(f"  {pillar:<18} {status_str:<10} {size:>8} {age:>12}")

    # Data
    print(f"\n  {'数据文件':<20} {'状态':<10} {'行数':>10} {'大小':>10}")
    print(f"  {'-'*52}")
    for name, info in s.get("data", {}).items():
        status_str = info.get("status", "?")
        lines = f"{info.get('lines', 0):,}"
        size = f"{info.get('size_mb', 0):.1f}MB"
        print(f"  {name:<20} {status_str:<10} {lines:>10} {size:>10}")

    # Kali
    kp = s.get("kali", {}).get("processes", {})
    print(f"\n  [Kali] 进程: {kp.get('count', 0)}, 权重支柱: {len(s.get('kali', {}).get('pillars', {}))}")

    # Windows
    wp = s.get("windows", {}).get("processes", {})
    print(f"  [Windows] 进程: {wp.get('count', 0)}")

    print(f"\n{'='*70}")


def run_daemon(interval: int = 60):
    print(f"[Lifers Monitor] 监控守护启动，间隔 {interval}s", flush=True)
    while True:
        try:
            status = collect_status()
            # Write to file for Web UI
            MONITOR_FILE.parent.mkdir(parents=True, exist_ok=True)
            MONITOR_FILE.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
            print_status(status)
        except Exception as e:
            print(f"[Lifers Monitor] 错误: {e}", flush=True)
        time.sleep(interval)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Lifers 全支柱监控系统")
    parser.add_argument("--once", action="store_true", help="单次采集")
    parser.add_argument("--interval", type=int, default=60, help="采集间隔秒")
    parser.add_argument("--json", action="store_true", help="JSON输出")
    args = parser.parse_args()

    if args.once:
        status = collect_status()
        if args.json:
            print(json.dumps(status, ensure_ascii=False, indent=2))
        else:
            print_status(status)
    else:
        run_daemon(args.interval)
