"""会话分类、CHAT_QUICK 推理与 Bridge 单轮 JSON 路径的完整性检查。"""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from lifers_brain.markov_lm import train_from_text


def _write_min_brain_root(root: Path) -> None:
    (root / "config").mkdir(parents=True, exist_ok=True)
    (root / "weights").mkdir(parents=True, exist_ok=True)
    (root / "memory").mkdir(parents=True, exist_ok=True)
    (root / "logs").mkdir(parents=True, exist_ok=True)
    (root / "state").mkdir(parents=True, exist_ok=True)
    stack = {
        "version": 1,
        "runtime": {"role": "brain"},
        "brain": {
            "model": "markov",
            "sandbox": False,
            "session_max_turns": 8,
            "memory_db": "memory/test_longterm.sqlite3",
            "weights": {
                "markov": "weights/lifers_markov.json",
                "transformer": "weights/lifers_transformer.json",
            },
            "deep_steward": {"enabled": False},
        },
        "human_sim": {"enabled": False},
        "instincts": {"enabled": False},
        "openclaw": {"enabled": False},
        "llm_ops": {"enabled": False},
        "organ_system": {"enabled": False},
        "physiology_sim": {"enabled": False},
    }
    (root / "config" / "stack.json").write_text(
        json.dumps(stack, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    corpus = "中文对话测试。今天天气不错。用户提出问题时要简洁回答。\n" * 40
    train_from_text(corpus).save(root / "weights" / "lifers_markov.json")


class SessionDialogueTests(unittest.TestCase):
    def setUp(self) -> None:
        self._prev = dict(os.environ)

    def tearDown(self) -> None:
        os.environ.clear()
        os.environ.update(self._prev)

    def test_classify_chat_quick(self) -> None:
        from lifers_brain.taskflow.classify import classify_task
        from lifers_brain.taskflow.kinds import TaskKind

        self.assertEqual(classify_task("你好", False), TaskKind.CHAT_QUICK)
        self.assertEqual(classify_task("search  Python", False), TaskKind.WEB_SEARCH)

    def test_chat_quick_skips_learn_by_default(self) -> None:
        from lifers_brain.agent import AgentConfig, LifersAgent
        from lifers_brain.taskflow.orchestrator import run_lifers_turn

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
            root = Path(d)
            _write_min_brain_root(root)
            os.environ["LIFERS_QUICK_CHAT_LEARN"] = "0"
            os.environ["LIFERS_TASKFLOW"] = "1"
            os.environ["LIFERS_MICRO_THINK_EVERY"] = "999"
            os.environ["LIFERS_QUICK_WEB"] = "0"
            agent = LifersAgent(AgentConfig(root_dir=root, model="markov", sandbox=True))
            n0 = agent.longterm.count_all()
            r = run_lifers_turn(agent, "今天适合做什么？")
            self.assertTrue(r.strip())
            n1 = agent.longterm.count_all()
            self.assertEqual(n0, n1, "CHAT_QUICK 默认不应写入 longterm")
            del agent

    def test_bridge_turn_json_smoke(self) -> None:
        from lifers_brain.bridge_turn import lifers_turn_from_json_body

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
            root = Path(d)
            _write_min_brain_root(root)
            os.environ["LIFERS_TASKFLOW"] = "1"
            os.environ["LIFERS_FORCE_LOCAL_ONLY"] = "1"
            os.environ["LIFERS_QUICK_CHAT_LEARN"] = "0"
            os.environ["LIFERS_MICRO_THINK_EVERY"] = "999"
            os.environ["LIFERS_QUICK_WEB"] = "0"
            os.environ["SANDBOX"] = "1"
            body = json.dumps({"text": "用一句话介绍你自己。", "contextFiles": []}, ensure_ascii=False)
            out = lifers_turn_from_json_body(root, body)
            self.assertTrue(out.get("ok"), msg=out.get("error"))
            self.assertTrue((out.get("text") or "").strip())

    def test_chat_message_coerce(self) -> None:
        from lifers_brain.chat_messages import coerce_messages, transcript_to_messages

        raw = [{"role": "user", "content": "hi"}, {"role": "oops", "content": "x"}, "bad", {"role": "assistant", "content": "ok"}]
        m = coerce_messages(raw)
        self.assertEqual(len(m), 2)
        t = transcript_to_messages("sys", "u", "a")
        self.assertEqual(len(t), 3)


if __name__ == "__main__":
    unittest.main()
