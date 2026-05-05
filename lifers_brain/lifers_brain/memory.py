from __future__ import annotations

import json
import sqlite3
import time
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

MemoryType = Literal[
    "fact",
    "preference",
    "commitment",
    "skill",
    "episode",
    "tool_result",
    "reflection",
    "instinct",
    "taskflow",
]


@dataclass
class MemoryItem:
    type: MemoryType
    content: Any
    importance: float = 0.5
    source: str = "system"
    ts_ms: int = 0


class Scratchpad:
    """Temporary per-run notes; not persisted unless promoted."""

    def __init__(self) -> None:
        self._items: List[MemoryItem] = []

    def add(self, item: MemoryItem) -> None:
        if item.ts_ms == 0:
            item.ts_ms = int(time.time() * 1000)
        self._items.append(item)

    def items(self) -> List[MemoryItem]:
        return list(self._items)

    def clear(self) -> None:
        self._items.clear()


class SessionMemory:
    """Short-term: rolling window + summary."""

    def __init__(self, max_turns: int = 8) -> None:
        self.max_turns = max_turns
        self.turns: List[Tuple[str, str]] = []  # (role, text)
        self.summary: str = ""

    def add_turn(self, role: str, text: str) -> None:
        self.turns.append((role, text))
        if len(self.turns) > self.max_turns:
            # Move older turns into summary (very simple).
            old = self.turns[:-self.max_turns]
            self.turns = self.turns[-self.max_turns :]
            snippet = "\n".join([f"{r}: {t}" for r, t in old])[:800]
            if self.summary:
                self.summary = (self.summary + "\n" + snippet)[-2000:]
            else:
                self.summary = snippet[-2000:]

    def context_text(self) -> str:
        turns = "\n".join([f"{r}: {t}" for r, t in self.turns])
        if self.summary:
            return f"SESSION_SUMMARY:\n{self.summary}\n\nRECENT_TURNS:\n{turns}"
        return f"RECENT_TURNS:\n{turns}"

    def sleep_compact(self) -> str:
        """
        本能·睡眠：把短期轮次卷入摘要，释放窗口（类比离线巩固）。
        """
        chunks: List[str] = []
        if self.summary.strip():
            chunks.append(self.summary.strip())
        if self.turns:
            chunks.append("\n".join([f"{r}: {t}" for r, t in self.turns]))
        blob = "\n\n".join(chunks).strip()
        self.turns.clear()
        tail = blob[-2400:] if len(blob) > 2400 else blob
        self.summary = tail
        return blob


class LongTermMemory:
    """
    Minimal dependency long-term memory:
    - SQLite full-text-ish via LIKE + type filters
    - stores JSON content
    """

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init()

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(str(self.db_path))
        con.execute("PRAGMA journal_mode=WAL;")
        return con

    def _init(self) -> None:
        with self._connect() as con:
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS memories (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  type TEXT NOT NULL,
                  content_hash TEXT,
                  content_json TEXT NOT NULL,
                  content_text TEXT NOT NULL,
                  importance REAL NOT NULL,
                  source TEXT NOT NULL,
                  ts_ms INTEGER NOT NULL
                );
                """
            )
            # Migration: add content_hash column if missing (older DBs).
            cols = [r[1] for r in con.execute("PRAGMA table_info(memories)").fetchall()]
            if "content_hash" not in cols:
                con.execute("ALTER TABLE memories ADD COLUMN content_hash TEXT;")
            con.execute("CREATE INDEX IF NOT EXISTS idx_mem_type ON memories(type);")
            con.execute("CREATE INDEX IF NOT EXISTS idx_mem_ts ON memories(ts_ms);")
            con.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_mem_hash ON memories(content_hash);")

    def _hash(self, t: str, content_json: str) -> str:
        h = hashlib.sha256((t + "\n" + content_json).encode("utf-8")).hexdigest()
        return h

    def add(self, item: MemoryItem) -> int:
        ts = item.ts_ms or int(time.time() * 1000)
        content_json = json.dumps(item.content, ensure_ascii=False)
        content_text = content_json if isinstance(item.content, (dict, list)) else str(item.content)
        ch = self._hash(item.type, content_json)
        with self._connect() as con:
            # Dedupe: if content_hash exists, don't insert again.
            existing = con.execute("SELECT id FROM memories WHERE content_hash = ?", (ch,)).fetchone()
            if existing:
                return int(existing[0])
            cur = con.execute(
                "INSERT INTO memories(type,content_hash,content_json,content_text,importance,source,ts_ms) VALUES(?,?,?,?,?,?,?)",
                (item.type, ch, content_json, content_text, float(item.importance), item.source, int(ts)),
            )
            return int(cur.lastrowid)

    def search(self, query: str, types: Optional[List[MemoryType]] = None, k: int = 6) -> List[Dict[str, Any]]:
        q = query.strip()
        if not q:
            return []
        with self._connect() as con:
            where = "content_text LIKE ?"
            params: List[Any] = [f"%{q}%"]
            if types:
                where += " AND type IN (" + ",".join(["?"] * len(types)) + ")"
                params.extend(types)
            rows = con.execute(
                f"SELECT id,type,content_json,importance,source,ts_ms FROM memories WHERE {where} ORDER BY importance DESC, ts_ms DESC LIMIT ?",
                (*params, int(k)),
            ).fetchall()
        out = []
        for rid, t, cj, imp, src, ts in rows:
            out.append(
                {
                    "id": rid,
                    "type": t,
                    "content": json.loads(cj),
                    "importance": imp,
                    "source": src,
                    "ts_ms": ts,
                }
            )
        return out

    def count_all(self) -> int:
        with self._connect() as con:
            row = con.execute("SELECT COUNT(*) FROM memories").fetchone()
            return int(row[0]) if row else 0

    def prune(self, min_importance: float = 0.15, older_than_days: int = 30, limit: int = 500) -> Dict[str, Any]:
        """
        Simple human-like forgetting:
        delete low-importance items older than N days.
        """
        cutoff = int(time.time() * 1000) - int(older_than_days) * 24 * 3600 * 1000
        with self._connect() as con:
            rows = con.execute(
                "SELECT id FROM memories WHERE importance < ? AND ts_ms < ? ORDER BY ts_ms ASC LIMIT ?",
                (float(min_importance), int(cutoff), int(limit)),
            ).fetchall()
            ids = [int(r[0]) for r in rows]
            if ids:
                con.execute(
                    f"DELETE FROM memories WHERE id IN ({','.join(['?']*len(ids))})",
                    tuple(ids),
                )
        return {"deleted": len(ids), "ids": ids, "cutoff_ts_ms": cutoff}

    def prune_type_older_than(self, mem_type: str, older_than_days: int = 14, limit: int = 500) -> Dict[str, Any]:
        """按类型与时间删除（如 taskflow 学习痕迹），不影响 preference 等其它类型。"""
        cutoff = int(time.time() * 1000) - int(older_than_days) * 24 * 3600 * 1000
        with self._connect() as con:
            rows = con.execute(
                "SELECT id FROM memories WHERE type = ? AND ts_ms < ? ORDER BY ts_ms ASC LIMIT ?",
                (str(mem_type), int(cutoff), int(limit)),
            ).fetchall()
            ids = [int(r[0]) for r in rows]
            if ids:
                con.execute(
                    f"DELETE FROM memories WHERE id IN ({','.join(['?'] * len(ids))})",
                    tuple(ids),
                )
        return {"deleted": len(ids), "type": mem_type, "cutoff_ts_ms": cutoff}

