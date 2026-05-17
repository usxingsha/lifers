from __future__ import annotations

import json
import re
import sqlite3
import sys
import time
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

import numpy as np

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

    def __init__(self, max_turns: int = 8, max_turn_chars: int = 6000) -> None:
        self.max_turns = max_turns
        self.max_turn_chars = max_turn_chars
        self.turns: List[Tuple[str, str]] = []  # (role, text)
        self.summary: str = ""

    def add_turn(self, role: str, text: str) -> None:
        if len(text) > self.max_turn_chars:
            text = text[: self.max_turn_chars].rstrip() + "…"
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

    def context_text(self, max_chars: int = 0) -> str:
        turns = "\n".join([f"{r}: {t}" for r, t in self.turns])
        prefix = ""
        if self.summary:
            prefix = f"SESSION_SUMMARY:\n{self.summary}\n\nRECENT_TURNS:\n{turns}"
        else:
            prefix = f"RECENT_TURNS:\n{turns}"
        if max_chars > 0 and len(prefix) > max_chars:
            prefix = "…" + prefix[-max_chars + 1 :]
        return prefix

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


# ── FTS5 shared infrastructure ──────────────────────────────────────────────

_CJK_RE = re.compile(r"[一-鿿㐀-䶿豈-﫿]")
_FTS_INIT_CACHE: set[tuple[str, int]] = set()
_FTS_INIT_VER = 4  # bump when FTS triggers / schema must be reapplied in-process


def _ensure_fts(db_path: str) -> None:
    """Idempotent FTS5 virtual table + trigger creation (safe to call from multiple sites)."""
    cache_key = (db_path, _FTS_INIT_VER)
    if cache_key in _FTS_INIT_CACHE:
        return
    try:
        with sqlite3.connect(db_path) as con:
            con.execute("PRAGMA journal_mode=WAL;")
            con.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
                    content_text,
                    tokenize='unicode61'
                );
            """
            )
            count = con.execute("SELECT count(*) FROM memories_fts").fetchone()[0]
            if count == 0:
                con.execute(
                    """
                    INSERT INTO memories_fts(rowid, content_text)
                    SELECT id, content_text FROM memories
                    WHERE content_text IS NOT NULL AND content_text != ''
                """
                )
            gap = con.execute(
                """
                SELECT COUNT(*) FROM memories m
                WHERE m.content_text IS NOT NULL AND m.content_text != ''
                  AND NOT EXISTS (SELECT 1 FROM memories_fts f WHERE f.rowid = m.id)
                """
            ).fetchone()[0]
            if int(gap or 0) > 0:
                con.execute("DELETE FROM memories_fts")
                con.execute(
                    """
                    INSERT INTO memories_fts(rowid, content_text)
                    SELECT id, content_text FROM memories
                    WHERE content_text IS NOT NULL AND content_text != ''
                """
                )
            con.execute(
                """
                CREATE TRIGGER IF NOT EXISTS memories_fts_ai AFTER INSERT ON memories BEGIN
                    INSERT INTO memories_fts(rowid, content_text) VALUES (new.id, new.content_text);
                END;
            """
            )
            con.execute("DROP TRIGGER IF EXISTS memories_fts_ad")
            con.execute(
                """
                CREATE TRIGGER memories_fts_ad AFTER DELETE ON memories BEGIN
                    DELETE FROM memories_fts WHERE rowid = old.id;
                END;
            """
            )
            con.execute("DROP TRIGGER IF EXISTS memories_fts_au")
            con.execute(
                """
                CREATE TRIGGER memories_fts_au AFTER UPDATE ON memories BEGIN
                    DELETE FROM memories_fts WHERE rowid = old.id;
                    INSERT INTO memories_fts(rowid, content_text) VALUES (new.id, new.content_text);
                END;
            """
            )
    except sqlite3.OperationalError:
        pass  # FTS5 not available or table locked — fallback will still work
    _FTS_INIT_CACHE.add(cache_key)


def _fts_query(query: str) -> str:
    """Convert user query to FTS5-safe string with CJK character separation."""
    cleaned = re.sub(r'["*^()ORNOT+\-]', " ", query)
    # Insert space after each CJK character so unicode61 indexes them individually
    spaced = _CJK_RE.sub(lambda m: m.group(0) + " ", cleaned)
    return spaced.strip()


def fts5_search(
    db_path: str,
    query: str,
    k: int = 6,
    types: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """FTS5 search shared by LongTermMemory and tools (e.g. KbSearchTool).

    Falls back to ``LIKE %q%`` when FTS returns nothing or the syntax fails.
    """
    q = query.strip()
    if not q:
        return []
    _ensure_fts(db_path)

    fts_q = _fts_query(q)
    out: List[Dict[str, Any]] = []

    try:
        with sqlite3.connect(db_path) as con:
            sql = """
                SELECT m.id, m.type, m.content_json, m.importance, m.source, m.ts_ms
                FROM memories m
                INNER JOIN memories_fts fts ON m.id = fts.rowid
                WHERE memories_fts MATCH ?
            """
            params: List[Any] = [fts_q]
            if types:
                sql += " AND m.type IN (" + ",".join(["?"] * len(types)) + ")"
                params.extend(types)
            sql += " ORDER BY m.importance DESC, m.ts_ms DESC LIMIT ?"
            params.append(int(k))
            rows = con.execute(sql, tuple(params)).fetchall()
    except sqlite3.OperationalError:
        rows = []

    if not rows:
        # Fallback to LIKE
        try:
            with sqlite3.connect(db_path) as con:
                where = "content_text LIKE ?"
                like_params: List[Any] = [f"%{q}%"]
                if types:
                    where += " AND type IN (" + ",".join(["?"] * len(types)) + ")"
                    like_params.extend(types)
                rows = con.execute(
                    f"SELECT id,type,content_json,importance,source,ts_ms FROM memories WHERE {where} ORDER BY importance DESC, ts_ms DESC LIMIT ?",
                    (*like_params, int(k)),
                ).fetchall()
        except sqlite3.OperationalError:
            rows = []

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


def _check_disk_space(db_path: Path, min_mb: int = 50) -> bool:
    """Return True if there is enough disk space for DB operations.
    Falls back to True when the check is not supported (platform limitation)."""
    try:
        import shutil
        _, _, free = shutil.disk_usage(str(db_path.parent))
        return (free // (1024 * 1024)) >= min_mb
    except (OSError, AttributeError):
        return True  # can't check, assume ok


def _wal_pragmas(con: "sqlite3.Connection") -> None:
    """Apply WAL size limits to prevent unbounded journal growth on edge devices."""
    con.execute("PRAGMA journal_size_limit=16777216;")       # 16 MB max WAL
    con.execute("PRAGMA wal_autocheckpoint=500;")             # checkpoint every 500 pages (~4 MB)


class LongTermMemory:
    """
    Long-term memory with optional vector search:
    - SQLite with FTS5 full-text search (fallback to LIKE for short/weird queries)
    - Optional vector DB backends: FAISS / Chroma / LanceDB
    - Hybrid search: reciprocal rank fusion of FTS5 + vector
    - stores JSON content
    - WAL size limited to 16 MB for edge deployment
    """

    def __init__(
        self,
        db_path: Path,
        vector_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init()
        _ensure_fts(str(self.db_path))
        # Periodic WAL checkpoint on init to keep journal small
        try:
            with sqlite3.connect(str(self.db_path)) as con:
                con.execute("PRAGMA wal_checkpoint(PASSIVE);")
        except sqlite3.OperationalError:
            pass
        # Vector DB integration (lazy)
        self._vector_store = None
        self._embedding_provider = None
        self._hybrid_search = None
        self._vector_config = vector_config or {}
        if self._vector_config.get("enabled"):
            self._init_vector()

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(str(self.db_path))
        con.execute("PRAGMA journal_mode=WAL;")
        _wal_pragmas(con)
        return con

    # ── Vector DB ─────────────────────────────────────────────────────────

    def _init_vector(self) -> None:
        cfg = self._vector_config
        backend = cfg.get("backend", "faiss")
        dim = cfg.get("dim", 256)
        try:
            if backend == "faiss":
                from lifers.vector_db.faiss_store import FAISSStore
                store_path = cfg.get("faiss_path", str(self.db_path.parent / "vector.faiss"))
                self._vector_store = FAISSStore(dim=dim, store_path=store_path)
                # Probe that faiss is actually importable
                self._vector_store._lazy_import()
            elif backend == "chroma":
                from lifers.vector_db.chroma_store import ChromaStore
                persist = cfg.get("chroma_path", str(self.db_path.parent / "chroma"))
                self._vector_store = ChromaStore(
                    collection_name=cfg.get("collection", "lifers_memory"),
                    persist_path=persist,
                )
                self._vector_store._lazy_connect()
            elif backend == "lancedb":
                from lifers.vector_db.lancedb_store import LanceDBStore
                self._vector_store = LanceDBStore(
                    uri=cfg.get("lancedb_uri", str(self.db_path.parent / "lancedb")),
                    table_name=cfg.get("collection", "lifers_memory"),
                )
                self._vector_store._lazy_connect()
            if self._vector_store is not None:
                from lifers.vector_db.embeddings import create_embedding_provider
                embed_kind = cfg.get("embed_kind", "tfidf")
                tfidf_path = cfg.get("tfidf_path")
                try:
                    self._embedding_provider = create_embedding_provider(
                        kind=embed_kind,
                        dim=dim,
                        model_name=cfg.get("embed_model", "all-MiniLM-L6-v2"),
                        tfidf_path=tfidf_path,
                    )
                except ImportError:
                    # sentence-transformers not installed, fallback to TF-IDF
                    self._embedding_provider = create_embedding_provider(
                        kind="tfidf", dim=dim,
                    )
                # Fit TF-IDF on existing memories if needed
                if embed_kind in ("tfidf", "tfidf_load") and tfidf_path:
                    try:
                        existing = self._all_documents()
                        if existing and hasattr(self._embedding_provider, "fit_from_documents"):
                            self._embedding_provider.fit_from_documents(existing)
                            # Save fitted embedder
                            self._embedding_provider.save(tfidf_path)
                    except Exception:
                        pass
                from lifers.vector_db.hybrid import HybridSearch
                self._hybrid_search = HybridSearch(
                    vector_store=self._vector_store,
                    embedding_provider=self._embedding_provider,
                    fts5_search_fn=lambda q, k, types: fts5_search(
                        str(self.db_path), q, k=k, types=types
                    ),
                )
                # Backfill existing memories into vector store
                self._backfill_vector()
        except (ImportError, ModuleNotFoundError) as e:
            sys.stderr.write(f"LIFERS_PROGRESS vector_init backend_not_available backend={backend} error={e}\n")
            sys.stderr.flush()
            self._vector_store = None
            self._embedding_provider = None
            self._hybrid_search = None

    def _all_documents(self) -> List[dict]:
        try:
            with self._connect() as con:
                rows = con.execute(
                    "SELECT id, type, content_json, importance, source, ts_ms FROM memories"
                ).fetchall()
            return [
                {
                    "id": r[0], "type": r[1], "content": json.loads(r[2]),
                    "importance": r[3], "source": r[4], "ts_ms": r[5],
                }
                for r in rows
            ]
        except Exception:
            return []

    def _backfill_vector(self) -> None:
        if self._vector_store is None or self._embedding_provider is None:
            return
        vs_count = self._vector_store.count()
        sql_count = self.count_all()
        if vs_count >= sql_count:
            return
        try:
            from lifers.vector_db.embeddings import _doc_content
            missing = []
            with self._connect() as con:
                rows = con.execute("SELECT id, content_json FROM memories").fetchall()
            for rid, cj in rows:
                doc = {"id": rid, "content": json.loads(cj)}
                content_text = _doc_content(doc)
                if content_text.strip():
                    missing.append((rid, content_text))
            if missing:
                ids, texts = zip(*missing)
                vecs = self._embedding_provider.embed(list(texts))
                self._vector_store.add(list(ids), vecs)
        except Exception:
            pass

    def vector_search(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        if self._hybrid_search is None:
            return []
        results = self._hybrid_search.search(query, k=k, vector_weight=1.0, fts_weight=0.0)
        return [{"id": r.id, "score": r.score, "metadata": r.metadata} for r in results]

    def hybrid_search(
        self,
        query: str,
        k: int = 10,
        vector_weight: float = 0.6,
        fts_weight: float = 0.4,
        types: Optional[List[MemoryType]] = None,
    ) -> List[Dict[str, Any]]:
        if self._hybrid_search is None:
            return self.search(query, types=types, k=k)
        results = self._hybrid_search.search(
            query, k=k, vector_weight=vector_weight, fts_weight=fts_weight, types=types,
        )
        enriched: List[Dict[str, Any]] = []
        for r in results:
            item = {"id": r.id, "score": r.score, "metadata": r.metadata}
            # Fetch full content from SQLite
            try:
                with self._connect() as con:
                    row = con.execute(
                        "SELECT type, content_json, importance, source, ts_ms FROM memories WHERE id=?",
                        (r.id,),
                    ).fetchone()
                if row:
                    item["type"] = row[0]
                    item["content"] = json.loads(row[1])
                    item["importance"] = row[2]
                    item["source"] = row[3]
                    item["ts_ms"] = row[4]
            except Exception:
                pass
            enriched.append(item)
        return enriched

    def get_vector_status(self) -> Dict[str, Any]:
        return {
            "enabled": self._vector_config.get("enabled", False),
            "backend": self._vector_config.get("backend", "none"),
            "vector_count": self._vector_store.count() if self._vector_store else 0,
            "sql_count": self.count_all(),
            "embed_dim": self._embedding_provider.dim if self._embedding_provider else 0,
        }

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
            row = con.execute(
                "SELECT sql FROM sqlite_master WHERE type='index' AND name=?",
                ("idx_mem_hash",),
            ).fetchone()
            if row and row[0] and "UNIQUE" in row[0].upper():
                con.execute("DROP INDEX IF EXISTS idx_mem_hash")
            con.execute("CREATE INDEX IF NOT EXISTS idx_mem_hash ON memories(content_hash);")

    def _hash(self, t: str, content_json: str) -> str:
        h = hashlib.sha256((t + "\n" + content_json).encode("utf-8")).hexdigest()
        return h

    def add(self, item: MemoryItem) -> int:
        if not _check_disk_space(self.db_path):
            sys.stderr.write(f"LIFERS_PROGRESS memory disk_space_low db_path={self.db_path}\n")
            sys.stderr.flush()
            return -1
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
            row_id = int(cur.lastrowid)
        # Also add to vector store
        if self._vector_store is not None and self._embedding_provider is not None:
            try:
                from lifers.vector_db.embeddings import _doc_content
                text = _doc_content({"content": item.content})
                if text.strip():
                    vec = self._embedding_provider.embed([text])
                    self._vector_store.add([row_id], vec, [{"type": item.type, "source": item.source}])
            except (ImportError, ModuleNotFoundError):
                self._vector_store = None
            except Exception:
                pass
        return row_id

    def search(self, query: str, types: Optional[List[MemoryType]] = None, k: int = 6) -> List[Dict[str, Any]]:
        q = query.strip()
        if not q:
            return []
        return fts5_search(
            str(self.db_path),
            q,
            k=k,
            types=list(types) if types else None,
        )

    def count_all(self) -> int:
        with self._connect() as con:
            row = con.execute("SELECT COUNT(*) FROM memories").fetchone()
            return int(row[0]) if row else 0

    def checkpoint(self) -> None:
        """Manually trigger WAL checkpoint to shrink journal (best-effort)."""
        try:
            with self._connect() as con:
                con.execute("PRAGMA wal_checkpoint(PASSIVE);")
        except sqlite3.OperationalError:
            pass

    def vacuum(self) -> Dict[str, Any]:
        """Recover disk space from deleted rows.  Returns page stats."""
        try:
            before = (self.db_path.stat().st_size) if self.db_path.is_file() else 0
            with self._connect() as con:
                con.execute("VACUUM;")
            after = (self.db_path.stat().st_size) if self.db_path.is_file() else 0
            return {"recovered_bytes": max(0, before - after), "before_bytes": before, "after_bytes": after}
        except (OSError, sqlite3.OperationalError) as e:
            return {"error": str(e)}

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
        if ids:
            if self._vector_store is not None:
                try:
                    self._vector_store.delete(ids)
                except Exception:
                    pass
            try:
                with self._connect() as c:
                    c.execute("PRAGMA wal_checkpoint(TRUNCATE);")
            except sqlite3.OperationalError:
                pass
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
        if ids:
            if self._vector_store is not None:
                try:
                    self._vector_store.delete(ids)
                except Exception:
                    pass
            try:
                with self._connect() as c:
                    c.execute("PRAGMA wal_checkpoint(TRUNCATE);")
            except sqlite3.OperationalError:
                pass
        return {"deleted": len(ids), "type": mem_type, "cutoff_ts_ms": cutoff}

