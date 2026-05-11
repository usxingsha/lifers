"""GUI 宿主 HTTP：/health、/api/editor-settings、POST /api/bridge（临时 brain 根）。"""
from __future__ import annotations

import json
import os
import tempfile
import unittest
import urllib.request
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
            "sandbox": True,
            "session_max_turns": 8,
            "memory_db": "memory/t_gui.sqlite3",
            "weights": {"markov": "weights/lifers_markov.json", "transformer": "weights/lifers_transformer.json"},
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


class GuiHostHttpTests(unittest.TestCase):
    def setUp(self) -> None:
        self._prev = dict(os.environ)

    def tearDown(self) -> None:
        os.environ.clear()
        os.environ.update(self._prev)

    def test_http_health_settings_bridge(self) -> None:
        from tools.lifers_gui_host.host import build_httpd, serve_background

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
            brain = Path(d) / "brain"
            brain.mkdir(parents=True)
            _write_min_brain_root(brain)
            repo = brain.parent
            httpd = build_httpd(brain, repo, "127.0.0.1", 0)
            port = httpd.server_address[1]
            serve_background(httpd)
            try:
                base = f"http://127.0.0.1:{port}"
                with urllib.request.urlopen(base + "/health", timeout=5) as h:
                    jh = json.loads(h.read().decode("utf-8"))
                self.assertTrue(jh.get("ok"))

                with urllib.request.urlopen(base + "/api/editor-settings", timeout=5) as s:
                    js = json.loads(s.read().decode("utf-8"))
                self.assertTrue(js.get("ok"))
                self.assertIn("fontSizePx", js.get("theme", {}))

                body = json.dumps({"text": "你好", "contextFiles": []}, ensure_ascii=False).encode("utf-8")
                req = urllib.request.Request(
                    base + "/api/bridge",
                    data=body,
                    headers={"Content-Type": "application/json; charset=utf-8"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=60) as br:
                    out = json.loads(br.read().decode("utf-8"))
                self.assertTrue(out.get("ok"), msg=out.get("error"))
                self.assertTrue((out.get("text") or "").strip())
            finally:
                httpd.shutdown()
                httpd.server_close()


if __name__ == "__main__":
    unittest.main()
