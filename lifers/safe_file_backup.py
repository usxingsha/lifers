"""
写前备份 + 失败自动回滚（文件库）。

- 每次写入在 LIFERS_ROOT/state/lifers_journal/<id>/ 下保留 content.bak 与 meta.json。
- 写入过程中抛错则立即从备份恢复或删除误建的新文件。
- 工具 verify 失败时由既有 rollback 路径调用 restore_from_hint。

环境：LIFERS_FILE_JOURNAL=0 时退化为「目标旁 .bak 临时文件」（与旧行为相近）。
"""

from __future__ import annotations

import json
import os
import shutil
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional


def _now_ms() -> int:
    return int(time.time() * 1000)


def _journal_root() -> Path:
    lr = os.environ.get("LIFERS_ROOT", "").strip()
    base = Path(lr).resolve() if lr else Path.cwd().resolve()
    root = base / "state" / "lifers_journal"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _use_central_journal() -> bool:
    return os.environ.get("LIFERS_FILE_JOURNAL", "1").strip().lower() not in ("0", "false", "no", "off")


def _restore_from_journal_dir(jdir: Path) -> None:
    meta_p = jdir / "meta.json"
    if not meta_p.is_file():
        return
    try:
        meta = json.loads(meta_p.read_text(encoding="utf-8"))
    except Exception:
        return
    restore_to = Path(str(meta.get("restore_to", "")))
    was_new = bool(meta.get("was_new"))
    if was_new:
        if restore_to.exists():
            try:
                restore_to.unlink()
            except OSError:
                pass
        return
    bf = jdir / "content.bak"
    if bf.is_file() and restore_to:
        restore_to.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(bf, restore_to)


def restore_from_hint(hint: Dict[str, Any]) -> Dict[str, Any]:
    """
    供 Tool.rollback 使用。支持：
    - journal_dir + meta.json（推荐）
    - 旧字段 backup_path + restore_to（无 was_new 时按「有备份=覆盖恢复」）
    """
    jd = hint.get("journal_dir")
    if jd:
        jdir = Path(str(jd))
        if jdir.is_dir():
            _restore_from_journal_dir(jdir)
            try:
                shutil.rmtree(jdir, ignore_errors=True)
            except Exception:
                pass
            return {"ok": True, "mode": "journal"}

    restore_to = hint.get("restore_to")
    backup_path = hint.get("backup_path")
    was_new = hint.get("was_new")
    if not restore_to:
        return {"ok": True, "note": "nothing to restore"}

    t = Path(str(restore_to))
    if was_new is True or (was_new is None and not backup_path):
        if t.exists():
            try:
                t.unlink()
            except OSError as e:
                return {"ok": False, "error": str(e)}
        return {"ok": True, "mode": "delete_new"}

    if not backup_path:
        return {"ok": True, "note": "no backup path"}
    bp = Path(str(backup_path))
    if not bp.is_file():
        return {"ok": False, "error": f"backup missing: {backup_path}"}
    t.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(bp, t)
    try:
        bp.unlink(missing_ok=True)  # type: ignore[arg-type]
    except Exception:
        pass
    return {"ok": True, "mode": "copy_backup"}


def safe_replace_file_text(
    target: Path,
    text: str,
    *,
    encoding: str = "utf-8",
) -> Dict[str, Any]:
    """
    写前备份并写入；任一步失败则尽力恢复并返回 ok=False。

    成功返回:
      ok, path, journal_dir?, backup_path?, restore_to, was_new
    """
    target = target.resolve()
    raw = text.encode(encoding)
    jdir: Optional[Path] = None
    side_backup: Optional[str] = None

    if not _use_central_journal():
        # 旁路 .bak：仍保证「写失败则恢复」
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            was_new = not target.is_file()
            if not was_new:
                side_backup = str(target) + f".bak.{_now_ms()}"
                shutil.copy2(target, side_backup)
            target.write_bytes(raw)
            return {
                "ok": True,
                "path": str(target),
                "journal_dir": None,
                "backup_path": side_backup,
                "restore_to": str(target),
                "was_new": was_new,
            }
        except Exception as e:
            if side_backup:
                try:
                    bp = Path(side_backup)
                    if bp.is_file():
                        shutil.copy2(bp, target)
                        bp.unlink(missing_ok=True)  # type: ignore[arg-type]
                except Exception:
                    pass
            elif not side_backup:
                try:
                    if target.exists() and target.is_file():
                        target.unlink()
                except OSError:
                    pass
            return {"ok": False, "error": str(e), "path": str(target)}

    jroot = _journal_root()
    journal_id = f"{_now_ms()}_{uuid.uuid4().hex[:10]}"
    jdir = jroot / journal_id
    try:
        jdir.mkdir(parents=True, exist_ok=True)
        was_new = not target.is_file()
        meta = {
            "restore_to": str(target),
            "was_new": was_new,
            "encoding": encoding,
            "created_ms": _now_ms(),
        }
        (jdir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        if not was_new:
            shutil.copy2(target, jdir / "content.bak")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
        return {
            "ok": True,
            "path": str(target),
            "journal_dir": str(jdir),
            "backup_path": None if was_new else str(jdir / "content.bak"),
            "restore_to": str(target),
            "was_new": was_new,
        }
    except Exception as e:
        if jdir and jdir.is_dir():
            _restore_from_journal_dir(jdir)
            try:
                shutil.rmtree(jdir, ignore_errors=True)
            except Exception:
                pass
        return {"ok": False, "error": str(e), "path": str(target)}


def rollback_hint_from_prior(prior: Any) -> Dict[str, Any]:
    """合并 execute 阶段写入 data / rollback_hint，供 Tool.rollback 使用。"""
    if prior is None:
        return {}
    h = dict(getattr(prior, "rollback_hint", None) or {})
    d = getattr(prior, "data", None) or {}
    for k in ("journal_dir", "backup_path", "restore_to", "was_new", "path"):
        if (k not in h or h.get(k) in (None, "")) and k in d:
            h[k] = d[k]
    if not h.get("restore_to") and d.get("path"):
        h["restore_to"] = d.get("path")
    return h


def commit_journal(journal_dir: Optional[str]) -> None:
    """写入与 verify 均成功后可选调用，删除日志目录释放空间。"""
    if not journal_dir:
        return
    p = Path(str(journal_dir))
    if p.is_dir():
        try:
            shutil.rmtree(p, ignore_errors=True)
        except Exception:
            pass
