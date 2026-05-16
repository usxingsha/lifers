"""check_lifers_llm_ready.py：无权重失败、有 markov 则成功。"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class LlmBootstrapCheckTests(unittest.TestCase):
    def test_script_fails_without_weights(self) -> None:
        script = Path(__file__).resolve().parents[1] / "scripts" / "check_lifers_llm_ready.py"
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
            root = Path(d)
            (root / "config").mkdir(parents=True)
            (root / "weights").mkdir(parents=True)
            (root / "config" / "stack.json").write_text("{}", encoding="utf-8")
            env = os.environ.copy()
            env["LIFERS_ROOT"] = str(root)
            p = subprocess.run([sys.executable, str(script)], cwd=str(root.parent), env=env, capture_output=True, text=True)
            self.assertEqual(p.returncode, 1)
            data = json.loads(p.stdout)
            self.assertFalse(data.get("ok"))

    def test_script_ok_with_markov_only(self) -> None:
        script = Path(__file__).resolve().parents[1] / "scripts" / "check_lifers_llm_ready.py"
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
            root = Path(d)
            (root / "weights").mkdir(parents=True)
            (root / "weights" / "lifers_markov.json").write_text("{}", encoding="utf-8")
            env = os.environ.copy()
            env["LIFERS_ROOT"] = str(root)
            p = subprocess.run([sys.executable, str(script)], cwd=str(root.parent), env=env, capture_output=True, text=True)
            self.assertEqual(p.returncode, 0, msg=p.stdout + p.stderr)
            data = json.loads(p.stdout)
            self.assertTrue(data.get("ok"))


if __name__ == "__main__":
    unittest.main()
