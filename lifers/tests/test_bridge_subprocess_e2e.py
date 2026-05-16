"""子进程 stdin/stdout 调用 agent_bridge_once：与扩展 Bridge 同路径的端到端烟测。"""
from __future__ import annotations

import json
import os
import subprocess
import sys
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
            "sandbox": True,
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


class BridgeSubprocessE2ETests(unittest.TestCase):
    def setUp(self) -> None:
        self._prev = dict(os.environ)

    def tearDown(self) -> None:
        os.environ.clear()
        os.environ.update(self._prev)

    def test_agent_bridge_once_subprocess_json(self) -> None:
        script = Path(__file__).resolve().parents[1] / "scripts" / "agent_bridge_once.py"
        self.assertTrue(script.is_file(), msg=f"missing {script}")

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
            root = Path(d)
            _write_min_brain_root(root)
            body = json.dumps(
                {"text": "你好，请用一句话说明你能帮我做什么。", "contextFiles": []},
                ensure_ascii=False,
            ).encode("utf-8")
            repo_root = Path(__file__).resolve().parents[1]
            env = {
                **os.environ,
                "PYTHONUTF8": "1",
                "PYTHONIOENCODING": "utf-8",
                "LIFERS_ROOT": str(root),
                # 与扩展一致：LIFERS_ROOT=数据/权重根；包 lifers 在仓库根下，由 PYTHONPATH 指向该根。
                "PYTHONPATH": str(repo_root),
                "SANDBOX": "1",
                "MODEL": "lifers",
                "LIFERS_FORCE_LOCAL_ONLY": "1",
                "LIFERS_TASKFLOW": "1",
                "LIFERS_QUICK_CHAT_LEARN": "0",
                "LIFERS_MICRO_THINK_EVERY": "999",
                "LIFERS_MAX_SPEED": "1",
                "LIFERS_QUICK_WEB": "0",
            }
            cp = subprocess.run(
                [sys.executable, "-u", str(script)],
                input=body + b"\n",
                cwd=str(root),
                env=env,
                capture_output=True,
                timeout=120,
            )
            self.assertEqual(cp.returncode, 0, msg=cp.stderr.decode("utf-8", errors="replace")[:2000])
            line = cp.stdout.decode("utf-8", errors="replace").strip().splitlines()[-1]
            out = json.loads(line)
            self.assertTrue(out.get("ok"), msg=out.get("error"))
            self.assertTrue((out.get("text") or "").strip())


if __name__ == "__main__":
    unittest.main()
