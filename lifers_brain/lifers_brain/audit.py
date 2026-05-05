from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional


def _root_dir() -> Path:
    root = os.environ.get("LIFERS_ROOT", "").strip()
    return Path(root) if root else Path.cwd()


def audit_log(event: Dict[str, Any]) -> None:
    """
    Append-only audit log. One JSON per line.
    Keep it dependency-free and robust.
    """
    root = _root_dir()
    from .stack_env import load_stack

    rel = str((load_stack(root).get("brain") or {}).get("audit_log", "logs/audit.jsonl"))
    p = Path(rel)
    path = p.resolve() if p.is_absolute() else (root / rel).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    event = dict(event)
    event.setdefault("ts_ms", int(time.time() * 1000))
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")

