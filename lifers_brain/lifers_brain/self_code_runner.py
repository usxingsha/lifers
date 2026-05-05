"""
自改代码队列：在桥接每轮前消费 state/self_code_queue/*.json。

每条 JSON：{"rel_path":"lifers_brain/...","new_text":"..."}（或 content 键）。
受 stack.brain.self_code 约束；SANDBOX=1 时跳过写入。
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List

from lifers_brain.safe_file_backup import commit_journal, safe_replace_file_text
from lifers_brain.stack_env import load_stack


def _norm_rel(rel: str) -> str:
    return rel.replace("\\", "/").strip().lstrip("/")


def _allowed(rel_norm: str, prefixes: Any) -> bool:
    if prefixes is None:
        return True
    if not isinstance(prefixes, list) or len(prefixes) == 0:
        return True
    if any(str(p).strip() in ("*", "**", "all") for p in prefixes):
        return True
    for p in prefixes:
        ps = str(p).replace("\\", "/").strip().rstrip("/") + "/"
        if rel_norm == ps.rstrip("/") or rel_norm.startswith(ps):
            return True
    return False


def _write_under_root(root: Path, rel_norm: str, new_text: str, max_bytes: int) -> Dict[str, Any]:
    raw = new_text if isinstance(new_text, str) else str(new_text)
    b = raw.encode("utf-8")
    if len(b) > int(max_bytes):
        return {"ok": False, "error": f"payload exceeds max_file_bytes={max_bytes}"}
    if ".." in Path(rel_norm).parts:
        return {"ok": False, "error": "path must not contain .."}

    target = (root / rel_norm).resolve()
    root_r = root.resolve()
    if not str(target).startswith(str(root_r)):
        return {"ok": False, "error": "path escapes LIFERS_ROOT"}

    sandbox = os.environ.get("SANDBOX", "0") == "1"
    if sandbox:
        return {"ok": False, "error": "SANDBOX=1 blocks self_code queue writes"}

    rec = safe_replace_file_text(target, raw, encoding="utf-8")
    if not rec.get("ok"):
        return {"ok": False, "error": rec.get("error", "write failed")}
    if rec.get("journal_dir"):
        commit_journal(str(rec["journal_dir"]))
    return {"ok": True, "path": str(target), "journal_dir": rec.get("journal_dir")}


def process_self_code_queue(root: Path) -> Dict[str, Any]:
    if os.environ.get("LIFERS_SELF_CODE_QUEUE", "1").strip().lower() in ("0", "false", "no", "off"):
        return {"skipped": True}

    stack = load_stack(root)
    sc = (stack.get("brain") or {}).get("self_code") or {}
    if not sc.get("enabled", False):
        return {"skipped": True, "reason": "self_code.enabled=false"}

    if not sc.get("auto_consume_queue", True):
        return {"skipped": True, "reason": "auto_consume_queue=false"}

    qrel = str(sc.get("queue_dir", "state/self_code_queue")).replace("\\", "/").strip().lstrip("/")
    qdir = (root / qrel).resolve()
    done_dir = (root / str(sc.get("done_dir", "state/self_code_done")).replace("\\", "/").strip().lstrip("/")).resolve()
    err_dir = (root / str(sc.get("error_dir", "state/self_code_error")).replace("\\", "/").strip().lstrip("/")).resolve()
    max_bytes = int(sc.get("max_file_bytes", 800_000) or 800_000)
    prefixes = sc.get("allow_rel_prefixes")

    if not qdir.is_dir():
        qdir.mkdir(parents=True, exist_ok=True)
    done_dir.mkdir(parents=True, exist_ok=True)
    err_dir.mkdir(parents=True, exist_ok=True)

    files: List[Path] = sorted(qdir.glob("*.json"))
    results: List[Dict[str, Any]] = []
    for fp in files:
        entry: Dict[str, Any] = {"file": fp.name}
        try:
            data = json.loads(fp.read_text(encoding="utf-8"))
            rel = _norm_rel(str(data.get("rel_path") or data.get("path") or ""))
            body = data.get("new_text")
            if body is None:
                body = data.get("content")
            if not rel or body is None:
                raise ValueError("missing rel_path or new_text/content")
            if not _allowed(rel, prefixes):
                raise ValueError(f"rel_path not allowed by allow_rel_prefixes: {rel}")
            wr = _write_under_root(root, rel, str(body), max_bytes)
            if not wr.get("ok"):
                raise RuntimeError(wr.get("error", "write failed"))
            entry.update(wr)
            shutil.move(str(fp), str(done_dir / fp.name))
        except Exception as e:  # noqa: BLE001
            entry["error"] = str(e)
            try:
                shutil.move(str(fp), str(err_dir / fp.name))
            except Exception:
                pass
        results.append(entry)

    return {"processed": len(files), "results": results}
