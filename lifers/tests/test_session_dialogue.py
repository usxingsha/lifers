"""会话分类、CHAT_QUICK 推理与 Bridge 单轮 JSON 路径的完整性检查。"""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from lifers.markov_lm import train_from_text


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
        from lifers.taskflow.classify import classify_task
        from lifers.taskflow.kinds import TaskKind

        self.assertEqual(classify_task("你好", False), TaskKind.CHAT_QUICK)
        self.assertEqual(classify_task("search  Python", False), TaskKind.WEB_SEARCH)

    def test_chat_quick_cn_greeting_no_markov_load(self) -> None:
        """中文寒暄走快路径，避免超大 markov JSON 整文件加载。"""
        from lifers.agent import AgentConfig, LifersAgent

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
            root = Path(d)
            _write_min_brain_root(root)
            os.environ["LIFERS_QUICK_CHAT_LEARN"] = "0"
            os.environ["LIFERS_TASKFLOW"] = "1"
            os.environ["LIFERS_MICRO_THINK_EVERY"] = "999"
            os.environ["LIFERS_QUICK_WEB"] = "0"
            os.environ["LIFERS_QUICK_TEMPLATE_SHORTCUTS"] = "1"
            agent = LifersAgent(AgentConfig(root_dir=root, model="markov", sandbox=True))
            r = agent.quick_chat("你好！")
            self.assertIn("Lifers", r)
            self.assertIn("中文", r)
            del agent

    def test_markov_json_size_cap_message(self) -> None:
        """超过 LIFERS_MARKOV_JSON_MAX_BYTES 时 generate 直接返回说明，不 json.load 整文件。"""
        from lifers.agent import AgentConfig, LifersAgent, LocalBrain

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
            root = Path(d)
            _write_min_brain_root(root)
            big = root / "weights" / "lifers_markov.json"
            big.write_text("x" * 8000, encoding="utf-8")
            os.environ["LIFERS_MARKOV_JSON_MAX_BYTES"] = "5000"
            brain = LocalBrain(AgentConfig(root_dir=root, model="markov", sandbox=True))
            out = brain.generate("你好", max_out_chars=80)
            self.assertIn("Markov", out)
            self.assertIn("上限", out)

    def test_chat_quick_skips_learn_by_default(self) -> None:
        from lifers.agent import AgentConfig, LifersAgent
        from lifers.taskflow.orchestrator import run_lifers_turn

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
        from lifers.bridge_turn import lifers_turn_from_json_body

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

    def test_bridge_turn_trivial_ascii_no_hang(self) -> None:
        """极短非中文输入应走确定性本地句，不依赖 web_search（即使 LIFERS_QUICK_WEB=1）。"""
        from lifers.bridge_turn import lifers_turn_from_json_body

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
            root = Path(d)
            _write_min_brain_root(root)
            os.environ["LIFERS_TASKFLOW"] = "1"
            os.environ["LIFERS_FORCE_LOCAL_ONLY"] = "1"
            os.environ["LIFERS_QUICK_CHAT_LEARN"] = "0"
            os.environ["LIFERS_MICRO_THINK_EVERY"] = "999"
            os.environ["LIFERS_QUICK_WEB"] = "1"
            os.environ["LIFERS_AGENTS_UI_BRIDGE"] = "1"
            os.environ["SANDBOX"] = "0"
            try:
                body = json.dumps({"text": "123", "contextFiles": []}, ensure_ascii=False)
                out = lifers_turn_from_json_body(root, body)
                self.assertTrue(out.get("ok"), msg=out.get("error"))
                text = (out.get("text") or "").strip()
                self.assertIn("收到", text)
                self.assertIn("123", text)
                self.assertIn("【本轮·生成锚】", text)
            finally:
                os.environ.pop("LIFERS_AGENTS_UI_BRIDGE", None)

    def test_quick_time_footer_respects_off(self) -> None:
        from lifers.agent import AgentConfig, LifersAgent

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
            root = Path(d)
            _write_min_brain_root(root)
            os.environ["LIFERS_QUICK_CHAT_LEARN"] = "0"
            os.environ["LIFERS_QUICK_WEB"] = "0"
            os.environ["LIFERS_QUICK_TEMPLATE_SHORTCUTS"] = "1"
            os.environ["LIFERS_AGENTS_UI_BRIDGE"] = "1"
            os.environ["LIFERS_QUICK_TIME_FOOTER"] = "0"
            try:
                agent = LifersAgent(AgentConfig(root_dir=root, model="markov", sandbox=True))
                r = agent.quick_chat("你好！")
                self.assertNotIn("【本轮·生成锚】", r)
                del agent
            finally:
                os.environ.pop("LIFERS_AGENTS_UI_BRIDGE", None)
                os.environ.pop("LIFERS_QUICK_TIME_FOOTER", None)

    def test_quick_chat_inference_logs_route_and_full_pack(self) -> None:
        """CHAT_QUICK 默认走完整栈上下文装配；stderr 可见 inference context_pack + 路由原因。"""
        import io
        from contextlib import redirect_stderr

        from lifers.agent import AgentConfig, LifersAgent

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
            root = Path(d)
            _write_min_brain_root(root)
            os.environ["LIFERS_QUICK_CHAT_LEARN"] = "0"
            os.environ["LIFERS_QUICK_WEB"] = "0"
            os.environ["LIFERS_QUICK_TEMPLATE_SHORTCUTS"] = "0"
            agent = LifersAgent(AgentConfig(root_dir=root, model="markov", sandbox=True))
            buf = io.StringIO()
            with redirect_stderr(buf):
                agent.quick_chat(
                    "今天适合做点什么？",
                    dialogue_route_reason="daily_chat_quick",
                    dialogue_route_notes_zh="日常对话",
                )
            err = buf.getvalue()
            self.assertIn("LIFERS_PROGRESS inference", err)
            self.assertIn("context_pack", err)
            self.assertIn("daily_chat_quick", err)
            self.assertIn("full_stack_pack", err)

    def test_chat_message_coerce(self) -> None:
        from lifers.chat_messages import coerce_messages, transcript_to_messages

        raw = [{"role": "user", "content": "hi"}, {"role": "oops", "content": "x"}, "bad", {"role": "assistant", "content": "ok"}]
        m = coerce_messages(raw)
        self.assertEqual(len(m), 2)
        t = transcript_to_messages("sys", "u", "a")
        self.assertEqual(len(t), 3)


if __name__ == "__main__":
    unittest.main()
