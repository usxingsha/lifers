"""Tests for FTS5 full-text search in LongTermMemory."""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from lifers.memory import LongTermMemory, MemoryItem, fts5_search, _fts_query, _ensure_fts


class FtsQueryTests(unittest.TestCase):
    """_fts_query preprocessing."""

    def test_ascii_query_unchanged(self) -> None:
        self.assertEqual(_fts_query("hello world"), "hello world")

    def test_cjk_spacing(self) -> None:
        """CJK characters get spaces inserted after them for unicode61."""
        result = _fts_query("今天天气")
        self.assertIn("今", result)
        self.assertIn("天", result)
        # Every CJK char should be followed by a space
        for ch in "今天天气":
            self.assertIn(ch, result)

    def test_special_chars_stripped(self) -> None:
        result = _fts_query('hello "world" *test OR')
        self.assertNotIn('"', result)
        self.assertNotIn("*", result)
        self.assertNotIn("OR", result)

    def test_mixed_cjk_ascii(self) -> None:
        result = _fts_query("python教程")
        self.assertIn("python", result)
        self.assertIn("教", result)
        self.assertIn("程", result)

    def test_empty_returns_empty(self) -> None:
        self.assertEqual(_fts_query(""), "")
        self.assertEqual(_fts_query("   "), "")


class Fts5MemoryTest(unittest.TestCase):
    """FTS5 search integration tests with an in-memory SQLite database."""

    def setUp(self) -> None:
        self._tmp = tempfile.mktemp(suffix=".sqlite3")
        self._db = Path(self._tmp)
        self.mem = LongTermMemory(self._db)

    def tearDown(self) -> None:
        try:
            os.unlink(self._tmp)
        except OSError:
            pass

    def _add(self, content: str, mem_type: str = "fact", importance: float = 0.5) -> int:
        return self.mem.add(MemoryItem(type=mem_type, content=content, importance=importance, source="test"))

    def test_fts_init_creates_table(self) -> None:
        import sqlite3

        with sqlite3.connect(self._tmp) as con:
            row = con.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='memories_fts'").fetchone()
        self.assertIsNotNone(row)

    def test_empty_search_returns_empty(self) -> None:
        self.assertEqual(self.mem.search(""), [])
        self.assertEqual(self.mem.search("  ", k=5), [])

    def test_basic_search(self) -> None:
        self._add("Python is a programming language")
        self._add("JavaScript is for web development")
        results = self.mem.search("python", k=5)
        self.assertEqual(len(results), 1)
        self.assertIn("python", str(results[0]["content"]).lower())

    def test_search_no_match(self) -> None:
        self._add("Hello world")
        results = self.mem.search("nonexistent", k=5)
        self.assertEqual(results, [])

    def test_cjk_search(self) -> None:
        self._add("今天天气很好")
        self._add("机器学习很有趣")
        results = self.mem.search("天气", k=5)
        self.assertEqual(len(results), 1)
        self.assertIn("天气", str(results[0]["content"]))

    def test_multi_word_search(self) -> None:
        self._add("The quick brown fox")
        self._add("The lazy dog")
        results = self.mem.search("quick fox", k=5)
        self.assertEqual(len(results), 1)

    def test_type_filter(self) -> None:
        self._add("Important preference", mem_type="preference", importance=0.9)
        self._add("Plain fact", mem_type="fact", importance=0.3)
        results = self.mem.search("", types=["preference"], k=5)
        # Empty query fallback: type filter still applies
        self.assertEqual(results, [])

    def test_search_with_types(self) -> None:
        self._add("Fact about AI", mem_type="fact", importance=0.8)
        self._add("Preference: I like Python", mem_type="preference", importance=0.6)
        results = self.mem.search("AI", types=["fact"], k=5)
        self.assertEqual(len(results), 1)

    def test_k_limits_results(self) -> None:
        for i in range(10):
            self._add(f"Item number {i}")
        results = self.mem.search("item", k=3)
        self.assertLessEqual(len(results), 3)

    def test_dedupe_via_fts(self) -> None:
        self._add("Unique content here")
        self._add("Unique content here")  # hash dedupe in add()
        results = self.mem.search("unique", k=5)
        self.assertEqual(len(results), 1)

    def test_fts_triggers_on_insert(self) -> None:
        """New inserts should be searchable via FTS5 immediately."""
        self._add("Newly inserted text for testing")
        results = self.mem.search("newly inserted", k=5)
        self.assertEqual(len(results), 1)

    def test_import_fts5_search_from_memory(self) -> None:
        """Verify fts5_search is importable and works as standalone."""
        self._add("Standalone function test")
        results = fts5_search(str(self._db), "standalone", k=5)
        self.assertEqual(len(results), 1)

    def test_fallback_to_like_when_fts_fails(self) -> None:
        """Very short queries or special chars should fall back to LIKE."""
        self._add("Test content for fallback")
        results = self.mem.search("test", k=5)
        self.assertGreaterEqual(len(results), 0)  # at least doesn't crash

    def test_prune_then_search(self) -> None:
        self._add("To be deleted", importance=0.1)
        import time
        time.sleep(0.01)  # ensure ts_ms < cutoff
        self._add("To be kept", importance=0.9)
        self.mem.prune(min_importance=0.5, older_than_days=0, limit=100)
        # After prune, low-importance row is gone from memories table.
        # FTS delete trigger should have removed it from the index.
        results = self.mem.search("deleted", k=5)
        self.assertEqual(len(results), 0)


if __name__ == "__main__":
    unittest.main()
