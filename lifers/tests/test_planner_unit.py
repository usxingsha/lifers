"""Unit tests for the Planner module (extracted from agent.py)."""
from __future__ import annotations

import os
import unittest
from pathlib import Path
from typing import List

from lifers.planner import Planner
from lifers.tools import ToolCall


class PlannerUnitTests(unittest.TestCase):
    def setUp(self) -> None:
        self._prev_root = os.environ.get("LIFERS_ROOT")
        root = Path(__file__).resolve().parents[1]
        os.environ["LIFERS_ROOT"] = str(root)

    def tearDown(self) -> None:
        if self._prev_root is None:
            os.environ.pop("LIFERS_ROOT", None)
        else:
            os.environ["LIFERS_ROOT"] = self._prev_root

    def test_empty_input_returns_empty(self) -> None:
        self.assertEqual(Planner().plan(""), [])
        self.assertEqual(Planner().plan("   "), [])

    def test_web_search_prefix(self) -> None:
        calls: List[ToolCall] = Planner().plan("search python tutorial")
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].name, "web_search")
        self.assertIn("python", calls[0].args.get("query", ""))

    def test_url_trigger(self) -> None:
        calls = Planner().plan("check https://example.com/page")
        names = [c.name for c in calls]
        self.assertIn("web_fetch", names)
        self.assertIn("extract_evidence", names)

    def test_kb_search_prefix(self) -> None:
        calls = Planner().plan("kb_search lifers agent")
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].name, "kb_search")
        self.assertEqual(calls[0].args.get("k"), 6)

    def test_workflow_prefix(self) -> None:
        calls = Planner().plan("流程 测试工作流")
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0].name, "kb_search")
        self.assertEqual(calls[1].name, "web_search")

    def test_workflow_english_prefix(self) -> None:
        calls = Planner().plan("workflow test workflow")
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0].name, "kb_search")
        self.assertEqual(calls[1].name, "web_search")

    def test_sim_run_prefix(self) -> None:
        calls = Planner().plan("sim_run abc123")
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].name, "sim_run")
        self.assertEqual(calls[0].args.get("task_id"), "abc123")

    def test_cmd_prefix(self) -> None:
        calls = Planner().plan("cmd ls -la")
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].name, "cmd_run")
        self.assertEqual(calls[0].args.get("cmd"), "ls -la")

    def test_weather_instinct_no_search(self) -> None:
        """Short weather queries return empty from plan() — handled by plan_real_world_instinct."""
        calls = Planner().plan("天气怎么样")
        self.assertEqual(calls, [])

    def test_real_world_instinct_weather(self) -> None:
        calls = Planner().plan_real_world_instinct("北京天气怎么样")
        names = [c.name for c in calls]
        self.assertIn("real_world", names)
        weather = [c for c in calls if c.name == "real_world" and c.args.get("action") == "weather"]
        self.assertEqual(len(weather), 1)

    def test_real_world_instinct_map(self) -> None:
        calls = Planner().plan_real_world_instinct("地图 北京故宫")
        names = [c.name for c in calls]
        self.assertIn("real_world", names)
        map_calls = [c for c in calls if c.name == "real_world" and c.args.get("action") == "map"]
        self.assertEqual(len(map_calls), 1)

    def test_real_world_instinct_clock(self) -> None:
        calls = Planner().plan_real_world_instinct("现在几点")
        names = [c.name for c in calls]
        self.assertIn("real_world", names)
        clock = [c for c in calls if c.name == "real_world" and c.args.get("action") == "clock"]
        self.assertEqual(len(clock), 1)

    def test_real_world_dedup(self) -> None:
        """Same action should not appear twice."""
        calls = Planner().plan_real_world_instinct("现在几点 今天星期几")
        actions = [c.args.get("action") for c in calls if c.name == "real_world"]
        self.assertEqual(len([a for a in actions if a == "clock"]), 1)

    def test_plan_returns_empty_for_meta_weather(self) -> None:
        """plan() should not duplicate real_world_instinct tools."""
        calls = Planner().plan("今天天气怎么样")
        self.assertEqual(calls, [])

    def test_kb_prune_prefix(self) -> None:
        calls = Planner().plan("kb_prune 0.1 60")
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].name, "kb_prune")
        self.assertAlmostEqual(calls[0].args.get("min_importance"), 0.1)

    def test_kb_compact_prefix(self) -> None:
        calls = Planner().plan("kb_compact https://example.com")
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].name, "kb_compact")
        self.assertEqual(calls[0].args.get("url"), "https://example.com")


if __name__ == "__main__":
    unittest.main()
