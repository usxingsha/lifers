"""Smoke tests for clean module imports after refactoring."""
from __future__ import annotations

import unittest


class ModuleImportTests(unittest.TestCase):
    """Verify all public exports resolve from their canonical locations."""

    def test_import_planner(self) -> None:
        from lifers.planner import Planner

        self.assertIsNotNone(Planner)

    def test_import_local_brain(self) -> None:
        from lifers.local_brain import AgentConfig, LocalBrain

        self.assertIsNotNone(AgentConfig)
        self.assertIsNotNone(LocalBrain)

    def test_import_fts_search(self) -> None:
        from lifers.memory import fts5_search, _ensure_fts, _fts_query

        self.assertIsNotNone(fts5_search)
        self.assertIsNotNone(_ensure_fts)
        self.assertIsNotNone(_fts_query)

    def test_import_from_init(self) -> None:
        import lifers

        self.assertTrue(hasattr(lifers, "AgentConfig"))
        self.assertTrue(hasattr(lifers, "LocalBrain"))
        self.assertTrue(hasattr(lifers, "Planner"))
        self.assertTrue(hasattr(lifers, "LifersAgent"))

    def test_backward_compat_imports(self) -> None:
        """Old import paths from agent.py must still work."""
        from lifers.agent import AgentConfig, LifersAgent
        from lifers.agent import LocalBrain  # re-exported for compat
        from lifers.agent import Planner  # re-exported for compat

        self.assertIsNotNone(AgentConfig)
        self.assertIsNotNone(LifersAgent)
        self.assertIsNotNone(LocalBrain)
        self.assertIsNotNone(Planner)

    def test_backward_compat_bridge_imports(self) -> None:
        from lifers.bridge_turn import lifers_turn, iter_stream_simple_chars

        self.assertIsNotNone(lifers_turn)
        self.assertIsNotNone(iter_stream_simple_chars)

    def test_all_exports(self) -> None:
        import lifers

        expected = [
            "AgentConfig",
            "LocalBrain",
            "Planner",
            "LifersAgent",
            "ToolCall",
            "ToolResult",
            "ToolRegistry",
            "build_default_registry",
            "LongTermMemory",
            "MemoryItem",
            "Scratchpad",
            "SessionMemory",
            "run_lifers_turn",
            "iter_stream_simple_chars",
            "lifers_turn",
        ]
        for name in expected:
            self.assertTrue(hasattr(lifers, name), f"Missing export: {name}")


if __name__ == "__main__":
    unittest.main()
