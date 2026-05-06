"""对话推理分发器：路由类型与中文说明。"""
from __future__ import annotations

import io
import sys
import unittest
from contextlib import redirect_stderr

from lifers_brain.taskflow.dialogue_router import infer_dialogue_route
from lifers_brain.taskflow.kinds import TaskKind


class DialogueRouterTests(unittest.TestCase):
    def test_daily_chat_quick(self) -> None:
        # 避免含「天气/几点」等触发 real_world 本能的口语（Planner 侧启发式会变）。
        r = infer_dialogue_route("好的，我们继续。", False, emit=False)
        self.assertEqual(r.kind, TaskKind.CHAT_QUICK)
        self.assertEqual(r.reason, "daily_chat_quick")
        self.assertIn("日常", r.notes_zh)

    def test_explicit_search(self) -> None:
        r = infer_dialogue_route("search  rust async", False, emit=False)
        self.assertEqual(r.kind, TaskKind.WEB_SEARCH)
        self.assertEqual(r.reason, "web_or_cn_query")

    def test_context_prefix_full_pipeline(self) -> None:
        r = infer_dialogue_route("仅尾句", True, emit=False)
        self.assertEqual(r.kind, TaskKind.FULL_PIPELINE)
        self.assertEqual(r.reason, "context_prefix")

    def test_emit_progress_line(self) -> None:
        buf = io.StringIO()
        with redirect_stderr(buf):
            infer_dialogue_route("你好", False, emit=True)
        err = buf.getvalue()
        self.assertIn("LIFERS_PROGRESS dialogue_route", err)
        self.assertIn("daily_chat_quick", err)


if __name__ == "__main__":
    unittest.main()
