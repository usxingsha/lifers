"""推理路径全覆盖：栈上下文装配、路由元数据、legacy/strict 环境变量、taskflow 端到端。"""
from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path

from lifers.markov_lm import train_from_text


def _brain_fixture(root: Path) -> None:
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


class InferenceComprehensiveTests(unittest.TestCase):
    def setUp(self) -> None:
        self._prev = dict(os.environ)

    def tearDown(self) -> None:
        os.environ.clear()
        os.environ.update(self._prev)

    def test_log_inference_emits_valid_json_line(self) -> None:
        from lifers.inference_pipeline import log_inference

        buf = io.StringIO()
        with redirect_stderr(buf):
            log_inference("unit_test", foo="bar", n=1)
        line = buf.getvalue().strip().split("\n")[-1]
        self.assertTrue(line.startswith("LIFERS_PROGRESS inference "))
        payload = json.loads(line[len("LIFERS_PROGRESS inference ") :])
        self.assertEqual(payload["stage"], "unit_test")
        self.assertEqual(payload["foo"], "bar")

    def test_stack_context_body_matches_context_pack_system_prefix(self) -> None:
        from lifers.agent import AgentConfig, LifersAgent

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
            root = Path(d)
            _brain_fixture(root)
            os.environ["LIFERS_QUICK_WEB"] = "0"
            agent = LifersAgent(AgentConfig(root_dir=root, model="markov", sandbox=True))
            body = agent._stack_context_body()
            pack = agent._context_pack("hello", [], [])
            self.assertIn(body.strip(), pack)
            self.assertIn("LONGTERM_RECALL:", pack)
            self.assertIn("TOOL_OBSERVATIONS:", pack)

    def test_quick_inference_pack_contains_route_and_recall_sections(self) -> None:
        from lifers.agent import AgentConfig, LifersAgent

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
            root = Path(d)
            _brain_fixture(root)
            agent = LifersAgent(AgentConfig(root_dir=root, model="markov", sandbox=True))
            agent._quick_route_reason = "assistant_meta_intent"
            agent._quick_route_notes_zh = "元问题"
            try:
                recalled = agent.longterm.search("测", k=3)
                pack = agent._quick_chat_inference_pack("能做什么", recalled)
            finally:
                agent._quick_route_reason = ""
                agent._quick_route_notes_zh = ""
            self.assertIn("DIALOGUE_ROUTE:", pack)
            self.assertIn("assistant_meta_intent", pack)
            self.assertIn("LONGTERM_RECALL:", pack)
            self.assertIn("CHAT_QUICK path", pack)

    def test_quick_clip_tighter_default_for_transformer(self) -> None:
        from lifers.agent import AgentConfig, LifersAgent

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
            root = Path(d)
            _brain_fixture(root)
            agent = LifersAgent(AgentConfig(root_dir=root, model="markov", sandbox=True))
            agent.brain.model = "transformer"
            os.environ.pop("LIFERS_QUICK_PACK_MAX_CHARS", None)
            blob = "Q" * 25_000
            clipped = agent._clip_quick_inference_prompt(blob)
            self.assertLess(len(clipped), 13_000)
            self.assertIn("截断", clipped)

    def test_quick_session_tail_limits_pack_for_transformer(self) -> None:
        from lifers.agent import AgentConfig, LifersAgent

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
            root = Path(d)
            _brain_fixture(root)
            agent = LifersAgent(AgentConfig(root_dir=root, model="markov", sandbox=True))
            agent.brain.model = "transformer"
            os.environ.pop("LIFERS_QUICK_SESSION_CONTEXT_CHARS", None)
            for _ in range(120):
                agent.session.add_turn("user", "行" * 120)
                agent.session.add_turn("assistant", "答" * 120)
            pack = agent._quick_chat_inference_pack("收到", [])
            self.assertIn("收到", pack)
            self.assertLess(len(pack), 25_000)

    def test_quick_stack_body_clip_transformer(self) -> None:
        from lifers.agent import AgentConfig, LifersAgent

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
            root = Path(d)
            _brain_fixture(root)
            agent = LifersAgent(AgentConfig(root_dir=root, model="markov", sandbox=True))
            agent.brain.model = "transformer"
            os.environ.pop("LIFERS_QUICK_STACK_BODY_CHARS", None)
            agent._stack_context_body = lambda: "Z" * 20_000
            pack = agent._quick_chat_inference_pack("x", [])
            self.assertIn("截断", pack)
            self.assertLess(len(pack), 12_000)

    def test_ni_zuo_shen_me_meta_template(self) -> None:
        """「你做什么」须走 assistant_meta_intent，即时模板，不卡 Markov。"""
        from lifers.agent import AgentConfig, LifersAgent
        from lifers.taskflow.orchestrator import run_lifers_turn

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
            root = Path(d)
            _brain_fixture(root)
            os.environ["LIFERS_QUICK_WEB"] = "0"
            os.environ["LIFERS_QUICK_TEMPLATE_SHORTCUTS"] = "0"
            os.environ["LIFERS_TASKFLOW"] = "1"
            agent = LifersAgent(AgentConfig(root_dir=root, model="markov", sandbox=True))
            out = run_lifers_turn(agent, "你做什么")
            self.assertIn("Lifers", out)
            self.assertIn("常见用法", out)

    def test_run_lifers_turn_meta_emits_route_and_inference(self) -> None:
        from lifers.agent import AgentConfig, LifersAgent
        from lifers.taskflow.orchestrator import run_lifers_turn

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
            root = Path(d)
            _brain_fixture(root)
            os.environ["LIFERS_QUICK_CHAT_LEARN"] = "0"
            os.environ["LIFERS_QUICK_WEB"] = "0"
            os.environ["LIFERS_QUICK_TEMPLATE_SHORTCUTS"] = "0"
            os.environ["LIFERS_TASKFLOW"] = "1"
            agent = LifersAgent(AgentConfig(root_dir=root, model="markov", sandbox=True))
            buf = io.StringIO()
            with redirect_stderr(buf):
                out = run_lifers_turn(agent, "能做什么")
            err = buf.getvalue()
            self.assertTrue(out.strip())
            self.assertIn("Lifers", out)
            self.assertIn("dialogue_route", err)
            self.assertIn("assistant_meta_intent", err)
            self.assertIn("LIFERS_PROGRESS inference", err)
            self.assertIn("taskflow_route", err)
            self.assertIn("meta_capability_by_route", err)

    def test_meta_route_can_opt_into_local_brain(self) -> None:
        """assistant_meta_intent + LIFERS_QUICK_META_USE_LOCAL_BRAIN=1 时仍走本地生成（慢路径）。"""
        from lifers.agent import AgentConfig, LifersAgent
        from lifers.taskflow.orchestrator import run_lifers_turn

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
            root = Path(d)
            _brain_fixture(root)
            os.environ["LIFERS_QUICK_CHAT_LEARN"] = "0"
            os.environ["LIFERS_QUICK_WEB"] = "0"
            os.environ["LIFERS_QUICK_TEMPLATE_SHORTCUTS"] = "0"
            os.environ["LIFERS_TASKFLOW"] = "1"
            os.environ["LIFERS_QUICK_META_USE_LOCAL_BRAIN"] = "1"
            agent = LifersAgent(AgentConfig(root_dir=root, model="markov", sandbox=True))
            buf = io.StringIO()
            with redirect_stderr(buf):
                out = run_lifers_turn(agent, "能做什么")
            err = buf.getvalue()
            self.assertTrue(out.strip())
            self.assertNotIn("meta_capability_by_route", err)
            self.assertIn("context_pack", err)
            self.assertIn("generate_local", err)

    def test_legacy_prompt_env_switch(self) -> None:
        from lifers.agent import AgentConfig, LifersAgent

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
            root = Path(d)
            _brain_fixture(root)
            os.environ["LIFERS_QUICK_WEB"] = "0"
            os.environ["LIFERS_QUICK_TEMPLATE_SHORTCUTS"] = "0"
            os.environ["LIFERS_QUICK_LEGACY_PROMPT"] = "1"
            agent = LifersAgent(AgentConfig(root_dir=root, model="markov", sandbox=True))
            buf = io.StringIO()
            with redirect_stderr(buf):
                r = agent.quick_chat("解释一下这个词：测试")
            self.assertTrue(r.strip())
            self.assertIn("legacy_narrow", buf.getvalue())

    def test_remote_stub_skips_local_when_env_on(self) -> None:
        """LIFERS_REMOTE_CHAT 开启且无 key 时应回落本地（与既有 remote_infer 行为一致）。"""
        from lifers.agent import AgentConfig, LifersAgent

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
            root = Path(d)
            _brain_fixture(root)
            os.environ.pop("LIFERS_CHAT_API_KEY", None)
            os.environ.pop("OPENAI_API_KEY", None)
            os.environ["LIFERS_REMOTE_CHAT"] = "1"
            os.environ["LIFERS_QUICK_WEB"] = "0"
            os.environ["LIFERS_QUICK_TEMPLATE_SHORTCUTS"] = "1"
            agent = LifersAgent(AgentConfig(root_dir=root, model="markov", sandbox=True))
            buf = io.StringIO()
            with redirect_stderr(buf):
                r = agent.quick_chat("你好")
            self.assertTrue(r.strip())
            err = buf.getvalue()
            self.assertIn("remote_infer skipped", err)
            self.assertIn("greeting_template", err)


if __name__ == "__main__":
    unittest.main()
