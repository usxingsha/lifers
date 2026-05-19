"""
静默/生产模式 — 后台服务日志统一管理
环境变量 LIFERS_SILENT=1 启用静默模式（stderr→文件，stdout仅JSON）
"""
from __future__ import annotations

import os
import sys
import logging
from pathlib import Path
from datetime import datetime

LOG_DIR = Path(os.environ.get("LIFERS_LOG_DIR", Path(__file__).resolve().parent.parent / "logs"))
LOG_DIR.mkdir(parents=True, exist_ok=True)


def is_silent() -> bool:
    return os.environ.get("LIFERS_SILENT", "0") == "1"


def setup_silent(name: str = "lifers"):
    if not is_silent():
        return

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fh = logging.FileHandler(LOG_DIR / f"{name}.log", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(fh)
    logger.propagate = False

    # Redirect stderr to log file
    log_path = LOG_DIR / f"{name}_stderr.log"
    log_fp = open(str(log_path), "a", encoding="utf-8")
    sys.stderr = log_fp

    # Keep stdout for JSON responses
    # Don't touch stdout

    return logger


def silent_print(msg: str, level: str = "info"):
    """静默模式下写入日志文件，否则输出到stderr"""
    if is_silent():
        log_path = LOG_DIR / "lifers_output.log"
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}\n")
    else:
        print(msg, file=sys.stderr)


def service_banner(service: str, **meta) -> str:
    """统一的JSON启动banner，兼容管道消费"""
    banner = {
        "lifers": service,
        "started": datetime.now().isoformat(),
        "pid": os.getpid(),
        **meta,
    }
    return json.dumps(banner, ensure_ascii=False)


def _json_dumps(obj):
    import json as _json
    return _json.dumps(obj, ensure_ascii=False)


def service_done(service: str):
    print(service_banner(service, status="stopped"), flush=True)
