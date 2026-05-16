"""Tests for LongTermMemory, Scratchpad, SessionMemory, fts5_search."""

from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path

from lifers.memory import (
    LongTermMemory,
    MemoryItem,
    Scratchpad,
    SessionMemory,
    _fts_query,
    fts5_search,
)


# ── Scratchpad ──────────────────────────────────────────────────────────────


class TestScratchpad:
    def test_add_and_items(self) -> None:
        sp = Scratchpad()
        assert sp.items() == []
        sp.add(MemoryItem(type="fact", content={"key": "val"}, importance=0.8))
        items = sp.items()
        assert len(items) == 1
        assert items[0].type == "fact"

    def test_clear(self) -> None:
        sp = Scratchpad()
        sp.add(MemoryItem(type="preference", content="abc"))
        sp.clear()
        assert sp.items() == []

    def test_auto_timestamp(self) -> None:
        sp = Scratchpad()
        sp.add(MemoryItem(type="fact", content="x"))
        assert sp.items()[0].ts_ms > 0

    def test_items_is_copy(self) -> None:
        sp = Scratchpad()
        sp.add(MemoryItem(type="fact", content="x"))
        items = sp.items()
        items.clear()
        assert len(sp.items()) == 1


# ── SessionMemory ───────────────────────────────────────────────────────────


class TestSessionMemory:
    def test_empty_context(self) -> None:
        sm = SessionMemory()
        assert sm.context_text() == "RECENT_TURNS:\n"

    def test_add_turn(self) -> None:
        sm = SessionMemory(max_turns=4)
        sm.add_turn("user", "hello")
        sm.add_turn("assistant", "hi")
        ctx = sm.context_text()
        assert "user: hello" in ctx
        assert "assistant: hi" in ctx

    def test_turn_truncation(self) -> None:
        sm = SessionMemory(max_turns=2, max_turn_chars=10)
        sm.add_turn("user", "a" * 100)
        ctx = sm.context_text()
        assert len(ctx.split("user:")[1].strip()) <= 11  # "a…" = truncated

    def test_turn_overflow_to_summary(self) -> None:
        sm = SessionMemory(max_turns=2)
        sm.add_turn("user", "m1")
        sm.add_turn("assistant", "m2")
        sm.add_turn("user", "m3")  # pushes m1 into summary
        ctx = sm.context_text()
        assert "SESSION_SUMMARY" in ctx
        assert "m1" in ctx
        assert "m3" in ctx

    def test_sleep_compact(self) -> None:
        sm = SessionMemory(max_turns=4)
        sm.add_turn("user", "hello")
        sm.add_turn("assistant", "world")
        blob = sm.sleep_compact()
        assert "user: hello" in blob
        assert sm.turns == []  # cleared
        assert sm.summary != ""

    def test_context_text_max_chars(self) -> None:
        sm = SessionMemory(max_turns=4)
        sm.add_turn("user", "long message here")
        ctx = sm.context_text(max_chars=10)
        assert len(ctx) <= 11  # "…" prefix adds 1 char


# ── FTS helpers ─────────────────────────────────────────────────────────────


class TestFtsQuery:
    def test_removes_special_chars(self) -> None:
        result = _fts_query('hello "world" (test)')
        assert '"' not in result
        assert "(" not in result

    def test_cjk_spacing(self) -> None:
        result = _fts_query("你好世界")
        assert "你好世界" not in result  # should have spaces between chars
        assert "你" in result

    def test_empty_after_cleaning(self) -> None:
        result = _fts_query('"*()-+')
        assert result == "" or result.strip() == ""


# ── LongTermMemory ──────────────────────────────────────────────────────────


class TestLongTermMemory:
    def test_init_creates_db(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            db = Path(td) / "test.db"
            mem = LongTermMemory(db)
            assert db.is_file()
            assert mem.count_all() == 0

    def test_add_and_count(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            mem = LongTermMemory(Path(td) / "test.db")
            id1 = mem.add(MemoryItem(type="fact", content={"x": 1}, importance=0.9))
            id2 = mem.add(MemoryItem(type="fact", content={"x": 2}, importance=0.5))
            assert id1 > 0
            assert id2 > id1
            assert mem.count_all() == 2

    def test_add_dedup(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            mem = LongTermMemory(Path(td) / "test.db")
            id1 = mem.add(MemoryItem(type="fact", content={"x": 1}, importance=0.9))
            id2 = mem.add(MemoryItem(type="fact", content={"x": 1}, importance=0.9))
            assert id1 == id2  # same content_hash → returns existing id
            assert mem.count_all() == 1

    def test_search_by_content(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            mem = LongTermMemory(Path(td) / "test.db")
            mem.add(MemoryItem(type="fact", content="hello world", importance=0.9))
            mem.add(MemoryItem(type="fact", content="goodbye world", importance=0.5))
            results = mem.search("hello")
            assert len(results) >= 1
            assert results[0]["content"] == "hello world"

    def test_search_empty_query(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            mem = LongTermMemory(Path(td) / "test.db")
            assert mem.search("") == []

    def test_search_by_type_filter(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            mem = LongTermMemory(Path(td) / "test.db")
            mem.add(MemoryItem(type="fact", content="apple", importance=0.8))
            mem.add(MemoryItem(type="preference", content="apple", importance=0.6))
            results = mem.search("apple", types=["fact"])
            assert all(r["type"] == "fact" for r in results)

    def test_prune_removes_low_importance_old_items(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            mem = LongTermMemory(Path(td) / "test.db")
            old_ts = int(time.time() * 1000) - 100 * 24 * 3600 * 1000  # 100 days ago
            mem.add(MemoryItem(
                type="fact", content="old and unimportant",
                importance=0.05, ts_ms=old_ts,
            ))
            mem.add(MemoryItem(
                type="fact", content="important and old",
                importance=0.9, ts_ms=old_ts,
            ))
            result = mem.prune(min_importance=0.15, older_than_days=30)
            assert result["deleted"] == 1  # only the unimportant one
            assert mem.count_all() == 1

    def test_prune_nothing_to_delete(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            mem = LongTermMemory(Path(td) / "test.db")
            mem.add(MemoryItem(
                type="fact", content="important",
                importance=0.9,
            ))
            result = mem.prune(min_importance=0.15, older_than_days=0)
            assert result["deleted"] == 0

    def test_prune_type_older_than(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            mem = LongTermMemory(Path(td) / "test.db")
            old_ts = int(time.time() * 1000) - 30 * 24 * 3600 * 1000  # 30 days ago
            mem.add(MemoryItem(
                type="taskflow", content="old task", importance=0.3, ts_ms=old_ts,
            ))
            mem.add(MemoryItem(
                type="preference", content="keep pref", importance=0.3, ts_ms=old_ts,
            ))
            result = mem.prune_type_older_than("taskflow", older_than_days=7)
            assert result["deleted"] == 1
            assert result["type"] == "taskflow"

    def test_prune_type_older_than_nothing(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            mem = LongTermMemory(Path(td) / "test.db")
            mem.add(MemoryItem(type="fact", content="new item", importance=0.3))
            result = mem.prune_type_older_than("fact", older_than_days=365)
            assert result["deleted"] == 0

    def test_checkpoint_succeeds(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            mem = LongTermMemory(Path(td) / "test.db")
            mem.add(MemoryItem(type="fact", content="x", importance=0.5))
            mem.checkpoint()

    def test_vacuum_succeeds(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            mem = LongTermMemory(Path(td) / "test.db")
            mem.add(MemoryItem(type="fact", content="x", importance=0.5))
            result = mem.vacuum()
            assert "error" not in result
            assert "recovered_bytes" in result

    def test_fts5_search_function(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            db_path = Path(td) / "test.db"
            mem = LongTermMemory(db_path)
            mem.add(MemoryItem(type="fact", content="unique search term", importance=0.9))
            results = fts5_search(str(db_path), "unique", k=5)
            assert len(results) >= 1
            assert "unique search term" in results[0]["content"]

    def test_fts5_search_empty_query(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            assert fts5_search(str(Path(td) / "test.db"), "") == []

    def test_count_all_on_empty(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            mem = LongTermMemory(Path(td) / "empty.db")
            assert mem.count_all() == 0
